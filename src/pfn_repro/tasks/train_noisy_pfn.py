"""Training the two-headed PFN on the noisy BNN prior (issue #3): `prior=noisy_bnn`, `surrogate=two_head_pfn`.

Same recipe as `train_pfn` (AdamW + warmup-cosine, checkpoint/resume, deadline fitting), with the
latent + noisy NLL of `training/noisy_trainer.py` and held-out validation per noise regime. The
checkpoint loads with `checkpoints.load_pfn_file` (its `surrogate` key names `TwoHeadPFN`) and carries
the prior config and the sha256 of the latent ECDF it was trained on.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from ..checkpoints import sha256
from ..priors.noisy_bnn import ASSETS, NoisyBNNPrior
from ..tracking.sink import Run
from ..training.noisy_trainer import NoisyPFNTrainer, build_regime_validation_callback
from ..utils.seeding import seed_everything
from .common import device_of, log_to, to_dict


def train_noisy_pfn(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    dev = device_of(t.device)
    seed_everything(cfg.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    prior_cfg = to_dict(cfg.prior)
    assert prior_cfg.pop("name") == "noisy_bnn", "train_noisy_pfn needs prior=noisy_bnn"
    prior = NoisyBNNPrior(batch_size=t.batch_size, x_dim=t.x_dim, seed=cfg.seed, device=dev, **prior_cfg)
    ecdf_file = ASSETS / prior.ecdf_key()
    model = hydra.utils.instantiate(cfg.surrogate, max_x_dim=prior.d).to(dev)
    model_config = {k: v for k, v in to_dict(cfg.surrogate).items() if k != "_target_"}
    model_config["max_x_dim"] = prior.d
    run.log_summary(n_params=sum(p.numel() for p in model.parameters()))
    run.set_tags(ecdf_file=ecdf_file.name, ecdf_sha256=sha256(ecdf_file))

    v = t.validation
    val = build_regime_validation_callback(
        prior_cfg,
        t.x_dim,
        prior.ecdf_sorted,
        context_sizes=tuple(v.context_sizes),
        n_functions=v.n_functions,
        n_points=v.n_points,
        every_n_steps=v.every_n_steps,
        device=dev,
    )
    ckpt = Path(run.dir) / "checkpoint.pt"
    tags = run.read().get("tags", {})
    trainer = NoisyPFNTrainer(
        prior=prior,
        model=model,
        seed=cfg.seed,
        model_config=model_config,
        callbacks=[val],
        on_log=log_to(run),
        checkpoint_path=ckpt,
        resume=ckpt.exists(),
        extra_checkpoint_metadata={
            "surrogate": cfg.surrogate._target_,
            "prior": {"name": "noisy_bnn", "x_dim": t.x_dim, **prior_cfg},
            "ecdf_file": ecdf_file.name,
            "ecdf_sha256": sha256(ecdf_file),
            "sweep_id": tags.get("sweep_id"),
            "git_commit": tags.get("git_commit"),
        },
        **to_dict(t.trainer),
    )
    history = trainer.run()["history"]
    last = {k: v[-1] for k, v in history.items() if k != "step" and v}
    run.log_summary(**{f"final/{k.replace('/', '_')}": float(v) for k, v in last.items()})
    return float(last.get("nll/train", float("nan")))
