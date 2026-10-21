# Ported from pfn-acqpolicy@980fad9 src/pfn_acqpolicy/evaluation/predictive.py, unchanged apart from import paths.
"""Per-test-point quality of a bar-distribution predictive, beyond NLL.

All functions take logits [..., nb] and targets y [...] and return per-point values,
so callers can average over points, then bootstrap over functions.

  nll        density NLL (log bin width included; comparable across bin counts)
  sq_err     (posterior mean - y)^2: point accuracy, what mean-based decisions see
  crps       continuous ranked probability score, int (F(t) - 1[t >= y])^2 dt: a proper
             score in y's units, bounded, insensitive to the bin resolution that caps NLL
  pit        F(y), the probability integral transform: uniform iff calibrated
  std        predictive std (sharpness)
Per function (over its test points):
  spearman        rank correlation of the posterior mean with y (what an argmax-based
                  acquisition needs: the right order, not the right level)
  select_regret   y[argmin mean] - min y over the test points: the regret of trusting
                  the posterior mean to pick the best of the queried points
"""
import torch


def _cdf_at(bd, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """CDF of the piecewise-uniform density at t (broadcast against p's leading dims)."""
    j = (torch.searchsorted(bd.borders, t.contiguous(), right=True) - 1).clamp(0, bd.num_bars - 1)
    cum = torch.cat([torch.zeros_like(p[..., :1]), p.cumsum(-1)], -1)
    frac = ((t - bd.borders[j]) / bd.bucket_widths[j]).clamp(0, 1)
    return (cum.gather(-1, j) + p.gather(-1, j) * frac).clamp(0, 1)


def pit(bd, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits.float(), -1)
    return _cdf_at(bd, p, y[..., None])[..., 0]


def crps(bd, logits: torch.Tensor, y: torch.Tensor, n_grid: int = 2000) -> torch.Tensor:
    """Midpoint rule on n_grid points over the support (exact CDF at each point)."""
    p = torch.softmax(logits.float(), -1)
    lo, hi = float(bd.borders[0]), float(bd.borders[-1])
    t = lo + (hi - lo) * (torch.arange(n_grid, device=p.device) + 0.5) / n_grid
    F = _cdf_at(bd, p, t.expand(*p.shape[:-1], n_grid))
    step = (t[None] >= y.reshape(-1, 1)).reshape(*y.shape, n_grid).float()
    return (F - step).square().mean(-1) * (hi - lo)


def point_metrics(bd, logits: torch.Tensor, y: torch.Tensor) -> dict:
    logits = logits.float()
    mean = bd.mean(logits)
    return {"nll": bd(logits, y), "sq_err": (mean - y).square(), "crps": crps(bd, logits, y),
            "pit": pit(bd, logits, y), "std": bd.variance(logits).clamp(min=0).sqrt(), "mean": mean}


def function_metrics(mean: torch.Tensor, y: torch.Tensor) -> dict:
    """mean, y [B, N] -> per function [B]."""
    def ranks(v):
        return v.argsort(-1).argsort(-1).float()
    rm, ry = ranks(mean), ranks(y)
    rm, ry = rm - rm.mean(-1, keepdim=True), ry - ry.mean(-1, keepdim=True)
    rho = (rm * ry).sum(-1) / (rm.square().sum(-1) * ry.square().sum(-1)).sqrt().clamp(min=1e-12)
    pick = y.gather(-1, mean.argmin(-1, keepdim=True))[..., 0]
    return {"spearman": rho, "select_regret": pick - y.min(-1).values}
