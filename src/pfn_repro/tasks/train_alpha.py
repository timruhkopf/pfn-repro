"""alpha-PFN (Rakotoarison et al., ICML 2026) on the BNN prior, both stages initialized from a trained PFN
(pfn-acqpolicy's train_es).

  stage=conditioned  q(y | x, D, I), I in {none, x*, f*, both}
  stage=alpha        a bar distribution over alpha~ = H[q(y|x,D)] - H[q(y|x,D,I)] for one variant
                     (pes: I = x*, i.e. Entropy Search; mes: f*; jes: both); its mean is the
                     amortized acquisition value (maximize it)

Checkpoints are given by registry name (`checkpoints.py`) or as a path to a file trained here.
"""

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from ..acquisitions.alpha_pfn.conditioned import OptimumConditionedPFN
from ..acquisitions.alpha_pfn.episodes import sample_episode
from ..acquisitions.alpha_pfn.targets import VARIANTS, fit_borders, info_gain
from ..acquisitions.alpha_pfn.trainers import AlphaPFNTrainer, ConditionedPFNTrainer
from ..acquisitions.alpha_pfn.validation import build_alpha_validation, build_conditioned_validation
from ..checkpoints import REGISTRY, load, load_conditioned_file, load_ecdf, load_pfn_file
from ..objectives.bar_distribution import BarDistribution
from ..priors.bnn import BNNPrior
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .common import device_of, log_to, to_dict


def _load(ref: str, device: str, conditioned: bool = False) -> tuple[Any, dict]:
    if ref in REGISTRY:
        return load(ref, device)
    return (load_conditioned_file if conditioned else load_pfn_file)(Path(ref), device)


def calibrate_borders(prior, conditioned, info, t, n_bins, seed) -> torch.Tensor:
    """Quantile borders from `calibration_batches` draws of alpha~ at random context sizes."""
    g = torch.Generator().manual_seed(seed + 1)
    samples = []
    for i in range(t.calibration_batches):
        n = int(torch.randint(t.trainer.min_train, t.trainer.max_train + 1, (1,), generator=g))
        ep = sample_episode(prior, n, t.trainer.n_test, t.trace, t.optimum, seed=-1 - i)
        samples.append(info_gain(conditioned, ep, info).flatten())
    return fit_borders(torch.cat(samples), n_bins)


def train_alpha(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    dev = device_of(t.device)
    seed_everything(cfg.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    base, base_ckpt = _load(t.base_checkpoint, dev)
    for p in base.parameters():  # a trainable copy of the frozen base
        p.requires_grad_(True)
    base.train()
    family = {k: v for k, v in to_dict(cfg.prior).items() if k != "name"}
    ecdf = load_ecdf(dev)
    prior = BNNPrior(
        batch_size=t.batch_size,
        x_dim=base.max_x_dim,
        variable_dim_min=1,
        seed=cfg.seed,
        ecdf_sorted=ecdf,
        device=dev,
        **family,
    )
    trace, optimum = to_dict(t.trace), to_dict(t.optimum)
    v = t.validation
    vkw: dict[str, Any] = dict(
        dims=list(v.dims),
        context_sizes=list(v.context_sizes),
        ecdf=ecdf,
        prior_kwargs=family,
        trace=trace,
        optimum=optimum,
        n_functions=v.n_functions,
        n_points=v.n_points,
        every_n_steps=v.every_n_steps,
        device=dev,
    )
    ckpt = Path(run.dir) / "checkpoint.pt"
    meta = {
        "kind": t.stage,
        "base_checkpoint": t.base_checkpoint,
        "base_prior": base_ckpt.get("prior"),
        "trace": trace,
        "optimum": optimum,
        "surrogate": "pfn_repro.surrogates.pfn.PFN",
        "git_commit": run.read().get("tags", {}).get("git_commit"),
    }
    common: dict[str, Any] = dict(
        prior=prior,
        seed=cfg.seed,
        model_config=base_ckpt["config"],
        on_log=log_to(run),
        checkpoint_path=ckpt,
        resume=ckpt.exists(),
        **to_dict(t.trainer),
    )

    if t.stage == "conditioned":
        model = OptimumConditionedPFN(base).to(dev)
        trainer = ConditionedPFNTrainer(
            model=model,
            trace=trace,
            optimum=optimum,
            episodes_per_function=t.episodes_per_function,
            callbacks=[build_conditioned_validation(**vkw)],
            extra_checkpoint_metadata=meta,
            **common,
        )
    elif t.stage == "alpha":
        info = VARIANTS[t.variant]
        conditioned, _ = _load(t.conditioned_checkpoint, dev, conditioned=True)
        model = base
        if t.reinit_head:
            model.out_head.reset_parameters()
        model.bar_dist = BarDistribution(
            calibrate_borders(prior, conditioned, info, t, base.bar_dist.num_bars, cfg.seed)
        ).to(dev)
        meta.update(variant=t.variant, info=info, conditioned_checkpoint=t.conditioned_checkpoint)
        trainer = AlphaPFNTrainer(
            model=model,
            conditioned=conditioned,
            info=info,
            trace=trace,
            optimum=optimum,
            episodes_per_function=t.episodes_per_function,
            callbacks=[build_alpha_validation(conditioned, info, **vkw)],
            extra_checkpoint_metadata=meta,
            **common,
        )
    else:
        raise ValueError(f"task.stage must be conditioned|alpha, got {t.stage}")

    run.log_summary(n_params=sum(p.numel() for p in model.parameters()))
    history = trainer.run()["history"]
    last = {k: v[-1] for k, v in history.items() if k != "step" and v}
    run.log_summary(**{f"final/{k.replace('/', '_')}": float(x) for k, x in last.items()})
    return float(last.get("nll/train", float("nan")))
