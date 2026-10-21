# Ported from pfn-acqpolicy@980fad9 src/pfn_acqpolicy/branching/bo.py (the score functions only).
"""Standard acquisitions from a PFN's bar-distribution logits (minimization; higher score = better).

  pi     P(Y < y_inc)                     exploitative
  ei     E[max(y_inc - Y, 0)]             balanced
  ucb    -(q-quantile), q = 1 - ucb_quantile: optimistic for minimization
  mean   -posterior mean                  pure exploitation
  std    posterior variance               pure uncertainty sampling
  mes    max-value entropy search (Wang & Jegelka 2017) from the bar distributions alone: p(f*) from the
         pool's marginals assumed independent, f* <= y_inc, 16 stratified quantiles; per point
         H[p(y|x)] - E_f* H[p(y|x) truncated to y >= f*], in closed form per bin
A number after the name sets the hyperparameter (`parse_arm`): `ei0.01` = margin 0.01 (on the ECDF
scale), `ucb0.99` = quantile 0.99. The amortized alternative to mes/es is `acquisitions.alpha_pfn`.
"""

import re

import torch

ARMS = ("pi", "ei", "ucb", "mean", "std", "mes")
MES_SAMPLES = 16


def parse_arm(name: str) -> tuple[str, float | None]:
    """'ei0.01@2' -> ('ei', 0.01): base acquisition and its hyperparameter (None = default).
    The number is the margin for pi/ei and the quantile for ucb."""
    m = re.fullmatch(r"([a-z]+?)([0-9.]+)?", name.split("@")[0])
    if m is None:
        raise ValueError(name)
    return m.group(1), (float(m.group(2)) if m.group(2) else None)


def arm_scores(
    name: str, bd, logits: torch.Tensor, inc: torch.Tensor, ucb_quantile: float, mes_fs: torch.Tensor | None = None
) -> torch.Tensor:
    """Higher = better. logits [M, P, nb], inc [M] -> [M, P]. `mes_fs` [M, n]: MES's f* samples, fixed
    (refinement scores a few candidates, whose own marginals would give the wrong p(f*))."""
    name, hp = parse_arm(name)
    xi = hp or 0.0  # PI/EI margin: improvement over y_inc - xi
    if name == "pi":
        return bd.pi(logits, (inc - xi)[:, None])
    if name == "ei":
        return bd.ei(logits, (inc - xi)[:, None])
    if name == "ucb":
        return -bd.ucb(logits, rest_prob=1.0 - (ucb_quantile if hp is None else hp))
    if name == "mean":
        return -bd.mean(logits)
    if name == "std":
        return bd.variance(logits)
    if name == "mes":
        return mes_scores(bd, logits, inc, fs=mes_fs)
    raise ValueError(name)


def fstar_quantiles(bd, logits: torch.Tensor, inc: torch.Tensor, n: int = MES_SAMPLES) -> torch.Tensor:
    """Stratified quantiles of min_x Y_x over the pool, the Y_x independent (MES's usual
    approximation), capped at the incumbent. logits [M, P, nb], inc [M] -> [M, n]."""
    p = torch.softmax(logits.float(), -1)
    cdf = torch.cat([torch.zeros_like(p[..., :1]), p.cumsum(-1)], -1).clamp(0, 1)  # [M,P,nb+1] at borders
    log_surv = torch.log1p(-cdf.clamp(max=1 - 1e-7)).sum(1)  # [M,nb+1]
    cdf_min = (1 - log_surv.exp()).contiguous()  # increasing in the border
    u = ((torch.arange(n, device=logits.device) + 0.5) / n).expand(cdf_min.shape[0], n).contiguous()
    j = (torch.searchsorted(cdf_min, u, right=True) - 1).clamp(0, bd.num_bars - 1)
    c0, c1 = cdf_min.gather(1, j), cdf_min.gather(1, j + 1)
    lo = bd.borders[j]
    f = lo + bd.bucket_widths[j] * ((u - c0) / (c1 - c0).clamp(min=1e-12)).clamp(0, 1)
    return torch.minimum(f, inc[:, None])


def mes_scores(
    bd, logits: torch.Tensor, inc: torch.Tensor, n: int = MES_SAMPLES, fs: torch.Tensor | None = None
) -> torch.Tensor:
    """H[Y_x] - mean_k H[Y_x | Y_x >= f*_k] for the piecewise-uniform bar density.
    Truncation keeps each bin's density and drops the mass below f*, so with
    q_j = p_j * frac_j (frac_j = part of bin j above f*), Z = sum q_j:
    H_trunc = -(1/Z) sum_j q_j log(p_j / w_j) + log Z. logits [M,P,nb], inc [M] -> [M,P]."""
    fs = fstar_quantiles(bd, logits, inc, n) if fs is None else fs  # [M,n] (given: from the pool)
    p = torch.softmax(logits.float(), -1)
    plw = p * (torch.log(p.clamp(min=1e-30)) - torch.log(bd.bucket_widths))  # p_j log(p_j/w_j)
    h = -plw.sum(-1)  # [M,P]
    tail_p = p.flip(-1).cumsum(-1).flip(-1)  # sum_{i>=j}
    tail_plw = plw.flip(-1).cumsum(-1).flip(-1)
    zero = torch.zeros_like(p[..., :1])
    tail_p1, tail_plw1 = torch.cat([tail_p[..., 1:], zero], -1), torch.cat([tail_plw[..., 1:], zero], -1)
    j0 = (torch.searchsorted(bd.borders, fs.contiguous(), right=True) - 1).clamp(0, bd.num_bars - 1)  # [M,n]
    frac = ((bd.borders[j0 + 1] - fs) / bd.bucket_widths[j0]).clamp(0, 1)  # [M,n]
    idx = j0[:, None].expand(-1, p.shape[1], -1)  # [M,P,n]
    q0 = p.gather(-1, idx) * frac[:, None]
    z = (q0 + tail_p1.gather(-1, idx)).clamp(min=1e-12)
    num = frac[:, None] * plw.gather(-1, idx) + tail_plw1.gather(-1, idx)
    h_trunc = -num / z + torch.log(z)
    return h - h_trunc.mean(-1)
