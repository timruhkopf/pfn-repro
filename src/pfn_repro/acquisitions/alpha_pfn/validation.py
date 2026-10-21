# Ported from pfn-acqpolicy@980fad9 src/pfn_acqpolicy/es/validation.py, unchanged apart from import paths.
"""Held-out validation for the two stages: fixed episodes per (d, n), drawn once.

Stage 1, `nll/val_d{d}_n{n}_{case}`, case in none/x/f/xf. Expected: x, f, xf
below none (the optimum is informative), and none no worse than the base PFN.
Stage 2, per (d, n): `nll`, `r2` = 1 - MSE(mean, alpha~) / Var(alpha~), and `rho`,
the Spearman correlation of the predicted mean with alpha~ across query points
within a context (what acquisition maximization uses), averaged over functions.
alpha~ is a single noisy draw per (D, x), so the best achievable r2/rho is < 1.
"""
import torch

from pfn_repro.training.callbacks import Callback
from pfn_repro.acquisitions.alpha_pfn.episodes import sample_episode
from pfn_repro.acquisitions.alpha_pfn.targets import info_gain
from pfn_repro.acquisitions.alpha_pfn.conditioned import CASES, case_flags
from pfn_repro.priors.bnn import BNNPrior


def _episodes(dims, context_sizes, ecdf, prior_kwargs, n_functions, n_points, trace, optimum, seed, device):
    out = {}
    for d in dims:
        prior = BNNPrior(batch_size=n_functions, x_dim=d, variable_dim_min=None, seed=seed + d,
                         ecdf_sorted=ecdf, device=device, **prior_kwargs)
        for n in context_sizes:
            out[d, n] = sample_episode(prior, n, n_points, trace, optimum, seed=seed + 31 * d + n)
    return out


def _spearman_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ra, rb = a.argsort(-1).argsort(-1).float(), b.argsort(-1).argsort(-1).float()
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)


def build_conditioned_validation(dims, context_sizes, ecdf, prior_kwargs, trace, optimum, n_functions=64,
                                 n_points=200, seed=20_000, every_n_steps=None, device="cpu") -> Callback:
    eps = _episodes(dims, context_sizes, ecdf, prior_kwargs, n_functions, n_points, trace, optimum, seed, device)

    def probe(step, trainer):
        out = {}
        with torch.no_grad():
            for (d, n), ep in eps.items():
                for case in CASES:
                    logits = trainer.model(ep.x_tr, ep.y_tr, ep.x_te, ep.x_opt, ep.f_opt,
                                           *case_flags(case, n_functions, ep.y_tr.device), ep.n_features)
                    out[f"nll/val_d{d}_n{n}_{case}"] = trainer.bar_dist(logits.float(), ep.y_te).mean().item()
        return out

    return Callback(name="", fn=probe, every_n_steps=every_n_steps)


def build_alpha_validation(conditioned, info, dims, context_sizes, ecdf, prior_kwargs, trace, optimum,
                           n_functions=64, n_points=200, seed=30_000, every_n_steps=None, device="cpu") -> Callback:
    eps = _episodes(dims, context_sizes, ecdf, prior_kwargs, n_functions, n_points, trace, optimum, seed, device)
    targets = {k: info_gain(conditioned, ep, info) for k, ep in eps.items()}

    def probe(step, trainer):
        out = {}
        with torch.no_grad():
            for (d, n), ep in eps.items():
                t = targets[d, n]
                logits = trainer.model(ep.x_tr, ep.y_tr, ep.x_te, n_features=ep.n_features).float()
                mean = trainer.bar_dist.mean(logits)
                out[f"nll/val_d{d}_n{n}"] = trainer.bar_dist(logits, t).mean().item()
                out[f"r2/val_d{d}_n{n}"] = 1 - ((mean - t).square().mean() / t.var()).item()
                out[f"rho/val_d{d}_n{n}"] = _spearman_rows(mean, t).mean().item()
        return out

    return Callback(name="", fn=probe, every_n_steps=every_n_steps)
