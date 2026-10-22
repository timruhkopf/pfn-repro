"""`FastMFPrior`: `MFPrior` with the per-curve work batched over the whole batch, same episodes.

Profile (laptop, batch 64, n_test 256, x_dim 3): `MFPrior` spends ~1.2 s per `sample_episode`, two thirds
of it in `AllocationPrior.parse_allocation_into_sequence`, which evaluates every curve of every task in its
own numpy call (~90 calls per task, each a dozen tiny ufuncs), and the parameter linking (scipy ppfs) once
per task. Here the random draws happen exactly as in `MFPrior._episode`, in the same order and from the same
global RNGs (allocation, configs, BNN forward, query fidelities, observation noise), but the deterministic
work is done once for all tokens of all tasks: one linker call (all tasks of a hp_dim share the latent
ECDF) and one evaluation of the curve mixture with per-token parameters.

The observation noise is drawn as one standard-normal vector per task and scaled per token: numpy's legacy
`normal(size=k, scale=s)` is `0 + s * gauss` with gauss read from the same stream, so concatenating the
per-curve draws gives the same numbers. The curve values could in principle differ from the per-curve path
in the last float64 bits (ufuncs on arrays vs scalars, a 4-term sum instead of `w @ y`); after the cast to
float32 they are identical in every case tested (tests/unit/test_priors.py asserts exact equality).
"""

import numpy as np
import torch

from .curves import CurveModels
from .episodes import MFPrior

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


class FastMFPrior(MFPrior):
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
        with torch.no_grad():
            latent = task.model(torch.from_numpy(configs).float()).numpy()
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
        y0, yinf, w, alpha, xsat, prec, rpsat, sigma = tasks[0].linker(cat("latent"), rep("y0"), rep("ymax"))
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
