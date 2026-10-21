# Ported from pfn-acqpolicy@980fad9 src/pfn_acqpolicy/es/episodes.py, unchanged apart from import paths.
"""Training episodes for the alpha-PFN stages: BO-like traces plus the function's minimum.

Traces follow alpha-PFN's Algorithm 1 (Appendix E; their ablation: uniform
contexts cause a domain shift in BO that grows with d). Each point is local with
a probability rising linearly along the trace, otherwise global:
- local: N(center, sigma^2) clipped to [0, 1]. The center is the incumbent for
  context points and a uniformly chosen context point for query points;
- global: U(-eps/2, 1 + eps/2) clipped, which puts mass on the box faces.
The paper doesn't give alpha_0, alpha_L, sigma or eps; ours are in the config.
One deviation for speed: context points are generated in `n_rounds` blocks, with
the incumbent updated between blocks instead of after every point.

x*/f* (minimization, noise-free, ECDF scale): best of a scrambled-Sobol set plus
the trace points, then Adam refinement through the differentiable prior (as in
`branching/instances.py::compute_ground_truth`, smaller budget). Coordinates on
irrelevant dims are redrawn uniformly: f is constant along them, so any value
is an equally valid minimizer.
"""
from dataclasses import dataclass

import torch

from pfn_repro.priors.bnn import BNNPrior


@dataclass
class Episode:
    x_tr: torch.Tensor   # [B, n, D]
    y_tr: torch.Tensor   # [B, n]
    x_te: torch.Tensor   # [B, m, D]
    y_te: torch.Tensor   # [B, m]
    x_opt: torch.Tensor  # [B, D]
    f_opt: torch.Tensor  # [B]
    n_features: torch.Tensor  # [B]


def _uniform(shape, lo, hi, prior):
    return lo + (hi - lo) * torch.rand(shape, device=prior.device, generator=prior.generator)


def _local_or_global(center, alpha, sigma, eps, prior):
    """center: [B, k, D], alpha: [k] local probability per position, sigma: [B, 1, 1]."""
    B, k, D = center.shape
    local = (center + sigma * torch.randn(center.shape, device=prior.device, generator=prior.generator)).clamp(0, 1)
    glob = _uniform((B, k, D), -eps / 2, 1 + eps / 2, prior).clamp(0, 1)
    is_local = torch.rand((B, k, 1), device=prior.device, generator=prior.generator) < alpha.view(1, k, 1)
    return torch.where(is_local, local, glob) * prior.active_dim_mask[:, None, :]


def sample_trace(prior: BNNPrior, n_train: int, n_test: int, alpha_range=(0.0, 0.8), sigma_range=(0.01, 0.2),
                 eps: float = 0.2, n_rounds: int = 8):
    B, D, dev = prior.B, prior.d, prior.device
    L = n_train + n_test
    alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * torch.arange(L, device=dev) / L
    lo, hi = torch.log(torch.tensor(sigma_range, device=dev))
    sigma = torch.exp(_uniform((B, 1, 1), lo, hi, prior))
    bounds = torch.linspace(0, n_train, min(n_rounds, n_train) + 1).round().long().tolist()
    xs, ys = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if not xs:
            x = _local_or_global(torch.zeros(B, b - a, D, device=dev), torch.zeros(b - a, device=dev), sigma, eps, prior)
        else:
            X, Y = torch.cat(xs, 1), torch.cat(ys, 1)
            inc = X[torch.arange(B), Y.argmin(1)][:, None].expand(B, b - a, D)
            x = _local_or_global(inc, alpha[a:b], sigma, eps, prior)
        xs.append(x)
        ys.append(prior.evaluate(x))
    x_tr, y_tr = torch.cat(xs, 1), torch.cat(ys, 1)
    pick = torch.randint(0, n_train, (B, n_test), device=dev, generator=prior.generator)
    centers = torch.gather(x_tr, 1, pick[..., None].expand(B, n_test, D))
    x_te = _local_or_global(centers, alpha[n_train:], sigma, eps, prior)
    return x_tr, y_tr, x_te, prior.evaluate(x_te)


def find_minimum(prior: BNNPrior, extra_x: torch.Tensor | None = None, n_dense: int = 4096, n_refine: int = 4,
                 refine_steps: int = 0, lr: float = 0.01, zoom_rounds: int = 3, n_zoom: int = 512,
                 zoom_sigma: float = 0.05, seed: int = 0, chunk: int = 2048):
    """-> x_opt [B, D], f_opt [B] of the current prior batch (batch-uniform active d).
    `zoom_rounds`: Gaussian samples around the best point, sigma shrinking 3x per round;
    one batched forward each, much cheaper on a GPU than Adam steps through the
    32-layer masked BNN (kernel-launch bound)."""
    B, D, dev = prior.B, prior.d, prior.device
    d = int(prior.active_dim[0])
    dense = torch.zeros(n_dense, D, device=dev)
    dense[:, :d] = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed).draw(n_dense).to(dev)
    cand = dense[None].expand(B, -1, -1)
    if extra_x is not None:
        cand = torch.cat([cand, extra_x], 1)
    with torch.no_grad():
        vals = torch.cat([prior.evaluate(cand[:, s:s + chunk], noise=False) for s in range(0, cand.shape[1], chunk)], 1)
    top = vals.topk(n_refine, dim=1, largest=False).indices
    x0 = torch.gather(cand, 1, top[..., None].expand(B, n_refine, D))
    x = x0.clone().requires_grad_(True)
    mask = prior.active_dim_mask[:, None, :]
    opt = torch.optim.Adam([x], lr=lr)
    with torch.enable_grad():  # callers may sit in no_grad (validation sets)
        for _ in range(refine_steps):
            opt.zero_grad()
            prior.evaluate((x * mask).clamp(0, 1), noise=False).sum().backward()
            opt.step()
    with torch.no_grad():
        xr = (x * mask).clamp(0, 1)
        allx, allf = torch.cat([xr, x0], 1), torch.cat([prior.evaluate(xr, noise=False), vals.gather(1, top)], 1)
        j = allf.argmin(1)
        x_opt, f_opt = allx[torch.arange(B), j], allf[torch.arange(B), j]
        mask1 = prior.active_dim_mask[:, None, :]
        for r in range(zoom_rounds):
            noise = torch.randn((B, n_zoom, D), device=dev, generator=prior.generator)
            xz = ((x_opt[:, None] + zoom_sigma / 3 ** r * noise) * mask1).clamp(0, 1)
            fz = prior.evaluate(xz, noise=False)
            jz = fz.argmin(1)
            better = fz[torch.arange(B), jz] < f_opt
            x_opt = torch.where(better[:, None], xz[torch.arange(B), jz], x_opt)
            f_opt = torch.where(better, fz[torch.arange(B), jz], f_opt)
        irrelevant = (prior.relevant_mask == 0) & (prior.active_dim_mask == 1)
        x_opt = torch.where(irrelevant, _uniform((B, D), 0.0, 1.0, prior), x_opt)
    return x_opt, f_opt


def sample_episode(prior: BNNPrior, n_train: int, n_test: int, trace: dict, optimum: dict, seed: int = 0,
                   minimum: tuple[torch.Tensor, torch.Tensor] | None = None) -> Episode:
    """A trace on each function and each function's minimum. `minimum=None`: fresh
    functions (prior.reset()) and a new search; otherwise the (x_opt, f_opt) of the
    prior's *current* functions is reused (the search dominates the cost). Either
    way a trace point replaces the minimum if it is lower."""
    if minimum is None:
        prior.reset()
        minimum = find_minimum(prior, seed=seed, **optimum)
    x_opt, f_opt = minimum
    x_tr, y_tr, x_te, y_te = sample_trace(prior, n_train, n_test, **trace)
    with torch.no_grad():
        X = torch.cat([x_tr, x_te], 1)
        F = prior.evaluate(X, noise=False)
        j = F.argmin(1)
        r = torch.arange(prior.B, device=X.device)
        better = F[r, j] < f_opt
        x_opt = torch.where(better[:, None], X[r, j], x_opt)
        f_opt = torch.where(better, F[r, j], f_opt)
    return Episode(x_tr, y_tr, x_te, y_te, x_opt, f_opt, prior.active_dim)
