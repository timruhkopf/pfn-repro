"""`FastMFPrior`: `MFPrior` with the per-curve work batched over the whole batch, same episodes.

Profile (laptop, batch 64, n_test 256, x_dim 3): `MFPrior` spends ~1.2 s per `sample_episode`, two thirds
of it in `AllocationPrior.parse_allocation_into_sequence`, which evaluates every curve of every task in its
own numpy call (~90 calls per task, each a dozen tiny ufuncs), and the parameter linking (scipy ppfs) once
per task. Here the random draws happen exactly as in `MFPrior._episode`, in the same order and from the same
global RNGs (allocation, configs, BNN forward, query fidelities, observation noise), but the deterministic
work is done once for all tokens of all tasks: one linker call (all tasks of a hp_dim share the latent
ECDF) and one evaluation of the curve mixture with per-token parameters.

Two more exact shortcuts: `LeanMLP` samples and evaluates the latent BNN (`bnn.MLP` via `BNNPrior.sample_mlp`)
without `nn.Module` overhead, making the same torch/numpy RNG calls in the same order (including nn.Linear's
default init, which `MLP.reset_parameters` overwrites); `TabulatedLinker` replaces the linker's scipy ppfs by
lookups, since their argument is a function of the latent ECDF index (one of len(ECDF) + 1 values).

`exact_stream=False` (used by experiment pfn_mf) drops two pieces of work that only advance torch's RNG: the
throwaway nn.Linear init draws, and the latent BNN's rows of configurations the episode never uses (each
config's row gets its own i.i.d. noise given the weights, so the used rows' distribution is unchanged). The
episode distribution is the same and every numpy draw (allocation, configs, fidelities, noise, y0/ymax) is
identical; only the torch stream, and so the latent BNN's weights and outputs, differ from MFPrior's.

The observation noise is drawn as one standard-normal vector per task and scaled per token: numpy's legacy
`normal(size=k, scale=s)` is `0 + s * gauss` with gauss read from the same stream, so concatenating the
per-curve draws gives the same numbers. The curve values could in principle differ from the per-curve path
in the last float64 bits (ufuncs on arrays vs scalars, a 4-term sum instead of `w @ y`); after the cast to
float32 they are identical in every case tested (tests/unit/test_priors.py asserts exact equality).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import expon, gamma, norm

from .curves import CurveModels
from .dims import FidelityPrior
from .episodes import N_LATENT, MFPrior

_BASIS = (CurveModels.power_law, CurveModels.exponential, CurveModels.logarithmic, CurveModels.hill)
_EPS_GRAD = 10**-9


def mixture(x, y0, yinf, w, alpha, xsat, prec, rpsat):
    """`VectorizedParameterLinker.weighted_curve_model_vectorized` with one parameter set per point.

    x, y0, yinf: [N]; w, alpha, xsat, prec, rpsat: [N, 4]. -> [N]."""
    out = np.zeros_like(x)
    eps = np.array([_EPS_GRAD, 2 * _EPS_GRAD])
    for i, fn in enumerate(_BASIS):
        xt = np.where(x < xsat[:, i], x, rpsat[:, i] * (x - xsat[:, i]) + xsat[:, i])
        args = (y0, yinf, prec[:, i], xsat[:, i], alpha[:, i])
        y_eps = fn(eps[:, None], *(a[None] for a in args))  # [2, N]: the slope at 0 of each curve
        grad = (y_eps[1] - y_eps[0]) / _EPS_GRAD
        tail = y0 * np.exp(np.clip(xt * (grad + _EPS_GRAD) / (y0 + 1e-15), -700, 700))
        out = out + w[:, i] * np.where(xt > 0, fn(xt, *args), tail)
    return out


class LeanMLP:
    """`BNNPrior.sample_mlp(n_in, n_out)` (bnn.py): the same draws, the same forward, as plain tensors."""

    def __init__(self, n_in: int, n_out: int, exact_stream: bool = True):
        n_layers = np.random.randint(8, 16)  # noqa: NPY002  (sample_mlp's draws, in its order)
        n_hidden = np.random.randint(36, 150)  # noqa: NPY002
        init_std = np.random.uniform(0.089, 0.193)  # noqa: NPY002
        sparseness = 0.145
        self.pre_noise = np.random.uniform(0.0003, 0.0014)  # noqa: NPY002
        self.out_noise = np.random.uniform(0.0004, 0.0013)  # noqa: NPY002
        dims = [n_in] + [n_hidden] * (n_layers - 1) + [n_out]
        self.ws = [torch.empty(o, i) for i, o in zip(dims[:-1], dims[1:], strict=True)]
        self.bs = [torch.empty(o) for o in dims[1:]]
        with torch.no_grad():
            # nn.Linear.__init__'s default init, then MLP.reset_parameters' linear.reset_parameters(): both are
            # overwritten below, but they advance torch's RNG. They are one contiguous block of uniform draws
            # (one per element, whatever the bounds), so one call consumes the same stream.
            if exact_stream:
                torch.empty(2 * sum(w.numel() + b.numel() for w, b in zip(self.ws, self.bs, strict=True))).uniform_()
            for w, b in zip(self.ws, self.bs, strict=True):
                w.normal_(0, init_std)
                b.normal_(0, init_std)
            for w in self.ws[1:-1]:
                w /= (1.0 - sparseness) ** (1 / 2)
                w *= torch.bernoulli(torch.ones_like(w) * (1.0 - sparseness))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for w, b in zip(self.ws[:-1], self.bs[:-1], strict=True):
            x = F.linear(x, w, b)
            x = x + torch.randn_like(x) * self.pre_noise
            x = torch.tanh(x)
        x = F.linear(x, self.ws[-1], self.bs[-1])
        return x + torch.randn_like(x) * self.out_noise


class TabulatedLinker:
    """`VectorizedParameterLinker.__call__` (curves.py) with every ppf looked up by ECDF index."""

    def __init__(self, y_samples: np.ndarray):
        self.y_samples = y_samples
        n = len(y_samples)
        self.eps = 0.5 / n
        u_shift = (1.0 - 2.0 * self.eps) * (np.arange(n + 1) / n) + self.eps
        self.gamma1 = gamma.ppf(u_shift, a=1)
        self.norm = {ls: norm.ppf(u_shift, loc=ls[0], scale=ls[1]) for ls in ((1, 1), (0, 1), (-4, 1), (0.5, 0.5), (-5, 1))}
        self.expon = expon.ppf(u_shift, scale=1)

    def __call__(self, bnn_outputs, y0, ymax):
        i = np.searchsorted(self.y_samples, bnn_outputs, side="left")
        u_raw = i / len(self.y_samples)
        Yinf = (ymax - y0) * u_raw[:, 0] + y0
        w_raw = self.gamma1[i[:, 1:5]]
        w = w_raw / w_raw.sum(axis=1, keepdims=True)
        a1 = np.exp(self.norm[1, 1][i[:, 5]])
        a2 = np.exp(self.norm[0, 1][i[:, 6]])
        a3 = 1.0 + np.exp(self.norm[-4, 1][i[:, 7]])
        a4 = np.exp(self.norm[0.5, 0.5][i[:, 8]])
        alpha = np.stack([a1, a2, a3, a4], axis=1)
        Xsat_max = 10 ** self.norm[0, 1][i[:, 9]]
        Xsat_rel = self.gamma1[i[:, 10:14]]
        Xsat = (Xsat_max[:, np.newaxis] * Xsat_rel) / np.max(Xsat_rel, axis=1, keepdims=True)
        PREC = 1.0 / (10 ** (-3.0 + 3.0 * u_raw[:, 14:18]))
        Rpsat = 1.0 - self.expon[i[:, 18:22]]
        sigma = np.exp(self.norm[-5, 1][i[:, 22]])
        return y0, Yinf, w, alpha, Xsat, PREC, Rpsat, sigma


class FastMFPrior(MFPrior):
    _linkers: dict[tuple[int, int], TabulatedLinker] = {}

    def __init__(self, *args, exact_stream: bool = True, **kwargs):
        self.exact_stream = exact_stream
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        """MFPrior.reset with each task's latent BNN drawn as a LeanMLP (the same draws)."""
        lo = self.variable_dim_min or self.x_dim
        self.hp_dim = int(torch.randint(lo, self.x_dim + 1, (1,), generator=self.generator))
        from .bnn import BNNPrior as LatentBNN
        from .task import MultiFidelityTask

        LatentBNN.ensure_ecdf_loaded(self.hp_dim, N_LATENT)
        with self._seeded():
            self.n_levels = self.fixed_levels or max(2, FidelityPrior().n_levels)
            if self.hp_dim not in self._tasks:
                self._tasks[self.hp_dim] = [MultiFidelityTask(self.hp_dim, N_LATENT) for _ in range(self.B)]
            for task in self._tasks[self.hp_dim]:
                task.model = LeanMLP(self.hp_dim, N_LATENT, self.exact_stream)  # pyright: ignore[reportAttributeAccessIssue]  (sample_task, lean)
                task.sample_y0_ymax()
        n_real = self.hp_dim + 1
        self.active_dim = torch.full((self.B,), n_real, dtype=torch.long, device=self.device)
        self.active_dim_mask = (torch.arange(self.d, device=self.device) < n_real).float().expand(self.B, -1)

    def _linker(self, task) -> TabulatedLinker:
        y = task.linker.y_samples
        key = (self.hp_dim, len(y))
        if key not in self._linkers or self._linkers[key].y_samples is not y:
            self._linkers[key] = TabulatedLinker(y)
        return self._linkers[key]

    def sample_episode(self, n_train: int, n_test: int):
        """-> x_tr [B, n_train, d], y_tr [B, n_train], x_te [B, n_test, d], y_te [B, n_test]."""
        seq = n_train + n_test
        tasks = self._tasks[self.hp_dim]
        parts = []
        with self._seeded():
            for task in tasks:  # every random draw, in MFPrior._episode's order
                parts.append(self._draw(task, seq, n_train))
        x, y = self._evaluate(tasks, parts, seq)
        x, y = x.to(self.device), y.to(self.device)
        return x[:, :n_train], y[:, :n_train], x[:, n_train:], y[:, n_train:]

    def _draw(self, task, seq: int, n_train: int) -> dict:
        from .allocation import AllocationPrior

        alloc = AllocationPrior(seq, self.n_levels)
        cutoff, epochs_per_curve, ordering = alloc.sample_abstract_allocation(n_train)
        configs = np.random.uniform(size=(seq, self.hp_dim))  # noqa: NPY002  (as in MFPrior._episode)
        x = torch.from_numpy(configs).float()
        if self.exact_stream:
            latent = task.model(x).numpy()  # LeanMLP or bnn.MLP
        else:  # only the configurations in the episode (unused rows: zeros, never read)
            used = np.unique(ordering)
            latent = np.zeros((seq, N_LATENT), dtype=np.float32)
            latent[used] = task.model(x[used]).numpy()
        # k-th appearance of each curve in the sequence
        by_curve = np.argsort(ordering, kind="stable")
        first = np.searchsorted(ordering[by_curve], np.arange(seq))
        k = np.empty(seq, dtype=int)
        k[by_curve] = np.arange(seq) - first[ordering[by_curve]]
        # fidelities: observations are levels 1..cutoff; queries are drawn per curve, in curve order
        fidelity = (k + 1) / self.n_levels
        for cid in np.flatnonzero(cutoff < epochs_per_curve):
            n_q = epochs_per_curve[cid] - cutoff[cid]
            q = np.random.choice(np.arange(cutoff[cid] + 1, self.n_levels + 1), size=n_q, replace=False)  # noqa: NPY002
            tok = by_curve[first[cid] + cutoff[cid] : first[cid] + epochs_per_curve[cid]]
            fidelity[tok] = q / self.n_levels
        z = np.empty(seq)
        z[by_curve] = np.random.normal(size=seq)  # noqa: NPY002  (the per-curve noise draws, concatenated)
        # curve id: 0 for a query of a curve not seen before (first appearance at or after n_train)
        first_pos = by_curve[first[ordering]]  # position of each token's curve's first appearance
        new_query = (first_pos >= n_train) & (np.arange(seq) == first_pos)
        curve_id = np.where(new_query, 0, ordering + 1)
        return dict(ordering=ordering, configs=configs, latent=latent, fidelity=fidelity, z=z, curve_id=curve_id,
                    y0=task.y0, ymax=task.ymax)

    def _evaluate(self, tasks, parts: list[dict], seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(parts)
        cat = lambda key: np.concatenate([p[key] for p in parts])  # noqa: E731
        rep = lambda key: np.repeat([p[key] for p in parts], seq)  # noqa: E731
        # one linker call: every task of this hp_dim links through the same latent ECDF
        y0, yinf, w, alpha, xsat, prec, rpsat, sigma = self._linker(tasks[0])(cat("latent"), rep("y0"), rep("ymax"))
        token = cat("ordering") + np.repeat(np.arange(B) * seq, seq)  # row of each token's curve
        y = mixture(cat("fidelity"), y0[token], yinf[token], w[token], alpha[token], xsat[token], prec[token],
                    rpsat[token])
        y = np.clip(y + sigma[token] * cat("z"), 0.0, 1.0)
        config = cat("configs")[token]
        x = np.concatenate([config, cat("fidelity")[:, None]], 1).reshape(B, seq, -1)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y.reshape(B, seq)).float()
        pad = torch.zeros(B, seq, self.d - x.shape[-1] - int(self.include_curve_id))
        cols = [x, pad]
        if self.include_curve_id:  # last column, as in MFPrior
            cols.append(torch.from_numpy(np.stack([p["curve_id"] for p in parts])).float()[..., None] / seq)
        return torch.cat(cols, -1), 1.0 - y if self.minimize else y
