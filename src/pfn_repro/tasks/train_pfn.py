"""Pretraining a surrogate on a prior: `prior=bnn|mf`, `surrogate=pfn|tabpfn` (pfn-acqpolicy's train_pfn).

A fresh prior batch every step, bar-distribution NLL, AdamW + warmup-cosine (`training/trainer.py`),
held-out validation per input dimension and context size. The checkpoint is written into the run
directory (`checkpoint.pt`, rewritten every `save_every` steps) and resumed from there when the member
is re-executed (`just sweep-resume`). Register a finished one in `checkpoints.py` to use it by name.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from ..priors import build_prior
from ..tracking.sink import Run
from ..training.trainer import PFNTrainer
from ..utils.seeding import seed_everything
from .common import device_of, episode_validation, log_to, to_dict


def train_pfn(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    dev = device_of(t.device)
    seed_everything(cfg.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    prior_cfg = to_dict(cfg.prior)
    prior = build_prior(
        batch_size=t.batch_size,
        x_dim=t.x_dim,
        seed=cfg.seed,
        device=dev,
        variable_dim_min=t.variable_dim_min,
        **prior_cfg,
    )
    model = hydra.utils.instantiate(cfg.surrogate, max_x_dim=prior.d).to(dev)
    model_config = {k: v for k, v in to_dict(cfg.surrogate).items() if k != "_target_"}
    model_config["max_x_dim"] = prior.d
    run.log_summary(n_params=sum(p.numel() for p in model.parameters()))

    ckpt = Path(run.dir) / "checkpoint.pt"
    trainer = PFNTrainer(
        prior=prior,
        model=model,
        seed=cfg.seed,
        model_config=model_config,
        callbacks=[episode_validation(prior_cfg, t, dev)],
        on_log=log_to(run),
        checkpoint_path=ckpt,
        resume=ckpt.exists(),
        extra_checkpoint_metadata={
            "surrogate": cfg.surrogate._target_,
            "prior": prior_cfg,
            "sweep_id": run.read().get("tags", {}).get("sweep_id"),
            "git_commit": run.read().get("tags", {}).get("git_commit"),
        },
        **to_dict(t.trainer),
    )
    history = trainer.run()["history"]
    last = {k: v[-1] for k, v in history.items() if k != "step" and v}
    run.log_summary(**{f"final/{k.replace('/', '_')}": float(v) for k, v in last.items()})
    return float(last.get("nll/train", float("nan")))
