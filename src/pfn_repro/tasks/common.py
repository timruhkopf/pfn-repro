"""Small helpers shared by the pipelines."""

from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from ..checkpoints import load_ecdf
from ..priors import build_prior
from ..tracking.sink import Run
from ..training.callbacks import Callback
from ..training.validation import build_dim_validation_callback


def to_dict(cfg: DictConfig | Any) -> dict[str, Any]:
    """A config node as a plain dict (typed, unlike OmegaConf.to_container)."""
    out = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(out, dict)
    return {str(k): v for k, v in out.items()}


def device_of(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")  # never fall back silently
    return name


def log_to(run: Run) -> Callable[[int, dict], None]:
    return lambda step, metrics: run.log_metrics({k: float(v) for k, v in metrics.items() if v == v}, step)


def episode_validation(prior_cfg: dict[str, Any], t: Any, device: str) -> Callback:
    """Fixed held-out episodes per input dimension and context size, drawn once.

    BNN prior: the ported `training/validation.py` (same seeds as the original runs). MF prior: the same
    idea on fixed-d MF episodes (`nll/val_d{d}_n{n}`, d = number of hyperparameters)."""
    v = t.validation
    if prior_cfg["name"] == "bnn":
        family = {k: val for k, val in prior_cfg.items() if k != "name"}
        return build_dim_validation_callback(
            list(v.dims),
            load_ecdf(device),
            family,
            context_sizes=tuple(v.context_sizes),
            n_functions=v.n_functions,
            n_points=v.n_points,
            every_n_steps=v.every_n_steps,
            device=device,
        )
    n_max = max(v.context_sizes)
    sets = {}
    for d in v.dims:
        prior = build_prior(
            batch_size=v.n_functions, x_dim=d, seed=10_000 + d, device=device, variable_dim_min=None, **prior_cfg
        )
        sets[d] = (prior.sample_episode(n_max, v.n_points), prior.active_dim)

    def probe(step: int, trainer: Any) -> dict[str, float]:
        out = {}
        with torch.no_grad():
            for d, ((x_tr, y_tr, x_te, y_te), nf) in sets.items():
                for n in v.context_sizes:
                    logits = trainer.model(x_tr[:, :n], y_tr[:, :n], x_te, n_features=nf).float()
                    out[f"nll/val_d{d}_n{n}"] = trainer.bar_dist(logits, y_te).mean().item()
                    out[f"mse/val_d{d}_n{n}"] = (trainer.bar_dist.mean(logits) - y_te).square().mean().item()
        return out

    return Callback(name="", fn=probe, every_n_steps=v.every_n_steps)
