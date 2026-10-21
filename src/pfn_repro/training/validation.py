# Ported from pfn-acqpolicy@980fad9 src/pfn_acqpolicy/callbacks/dim_validation.py, unchanged apart from import paths.
"""Held-out validation per dimension and context size: `nll/val_d{d}_n{n}`, `mse/val_d{d}_n{n}`.

One fixed set per d, drawn once at build time (fixed functions *and* points), so
the curves have no resampling noise. Every d shares the training prior's ECDF:
a per-d ECDF would rescale each d to fill [0, 1] and hide cross-d difficulty.
Context size n uses the first n context points, so all n share the test points.
The NLL is a density NLL (log bin width included), comparable across n_bins.
"""
import torch

from pfn_repro.training.callbacks import Callback
from pfn_repro.priors.bnn import BNNPrior


def build_dim_validation_callback(dims: list[int], ecdf_sorted: torch.Tensor, prior_kwargs: dict,
                                  context_sizes: tuple[int, ...] = (5, 20, 50), n_functions: int = 64,
                                  n_points: int = 200, seed: int = 10_000, every_n_steps: int | None = None,
                                  device: str = "cpu") -> Callback:
    n_max = max(context_sizes)
    sets = {}
    for d in dims:
        prior = BNNPrior(batch_size=n_functions, x_dim=d, variable_dim_min=None, seed=seed + d,
                         ecdf_sorted=ecdf_sorted, device=device, **prior_kwargs)
        sets[d] = prior.sample_episode(n_max, n_points)

    def probe(step: int, trainer) -> dict:
        out = {}
        with torch.no_grad():
            for d, (x_tr, y_tr, x_te, y_te) in sets.items():
                nf = torch.full((n_functions,), d, device=x_tr.device)
                for n in context_sizes:
                    logits = trainer.model(x_tr[:, :n], y_tr[:, :n], x_te, n_features=nf).float()
                    out[f"nll/val_d{d}_n{n}"] = trainer.bar_dist(logits, y_te).mean().item()
                    out[f"mse/val_d{d}_n{n}"] = (trainer.bar_dist.mean(logits) - y_te).square().mean().item()
        return out

    return Callback(name="", fn=probe, every_n_steps=every_n_steps)
