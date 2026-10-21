"""`MFPrior`: the multi-fidelity learning-curve prior behind the same episode interface as `priors.bnn.BNNPrior`,
so `training.trainer.PFNTrainer` and every surrogate can train on it unchanged.

One episode per batch item, built exactly like ifBO's sequences (`allocation.AllocationPrior`): a
sequence of (config, fidelity) tokens drawn from a few learning curves; the first `n_train` are the
context (each curve observed from its first fidelity level up to its cutoff), the rest are queries at
later fidelities of seen curves or of new ones. Inputs are packed as `[config (d dims), fidelity]`, so
`n_features = d + 1` real columns (zero-padded up to `x_dim + 1`); an optional curve-id column
(ifBO's id token) can be appended. y is the curve's performance in [0, 1], flipped to `1 - perf` with
`minimize=True` (this package minimizes everywhere).
"""

import contextlib

import numpy as np
import torch

from .allocation import AllocationPrior
from .bnn import BNNPrior as LatentBNN
from .dims import FidelityPrior
from .task import MultiFidelityTask

N_LATENT = 23  # outputs of the curve-parameter BNN (curves.VectorizedParameterLinker's schema)


class MFPrior:
    def __init__(
        self,
        batch_size: int,
        x_dim: int = 3,
        variable_dim_min: int | None = 1,
        n_levels: int | None = 50,
        minimize: bool = True,
        include_curve_id: bool = False,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.B = batch_size
        self.x_dim = x_dim
        self.variable_dim_min = variable_dim_min
        self.fixed_levels = n_levels  # None: FidelityPrior draws the number of levels per reset
        self.minimize = minimize
        self.include_curve_id = include_curve_id
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)
        self.d = x_dim + 1 + int(include_curve_id)  # input width (BNNPrior.d's meaning)
        self._tasks: dict[int, list[MultiFidelityTask]] = {}
        self.reset()

    @contextlib.contextmanager
    def _seeded(self):
        """The ported code draws from numpy's and torch's global RNGs: seed both from our generator, and
        restore the caller's torch state afterwards."""
        s = int(torch.randint(0, 2**31 - 1, (1,), generator=self.generator))
        with torch.random.fork_rng():
            np.random.seed(s)  # noqa: NPY002  (the ported prior uses the global numpy RNG)
            torch.manual_seed(s)
            yield

    def reset(self) -> None:
        """New functions: a batch-uniform number of hyperparameters, a new task per batch item."""
        lo = self.variable_dim_min or self.x_dim
        self.hp_dim = int(torch.randint(lo, self.x_dim + 1, (1,), generator=self.generator))
        LatentBNN.ensure_ecdf_loaded(self.hp_dim, N_LATENT)  # before seeding: generated once, own seed
        with self._seeded():
            self.n_levels = self.fixed_levels or max(2, FidelityPrior().n_levels)
            if self.hp_dim not in self._tasks:
                self._tasks[self.hp_dim] = [MultiFidelityTask(self.hp_dim, N_LATENT) for _ in range(self.B)]
            for task in self._tasks[self.hp_dim]:
                task.sample_task()
        n_real = self.hp_dim + 1
        self.active_dim = torch.full((self.B,), n_real, dtype=torch.long, device=self.device)
        self.active_dim_mask = (torch.arange(self.d, device=self.device) < n_real).float().expand(self.B, -1)

    def sample_episode(self, n_train: int, n_test: int):
        """-> x_tr [B, n_train, d], y_tr [B, n_train], x_te [B, n_test, d], y_te [B, n_test]."""
        seq = n_train + n_test
        xs, ys = [], []
        with self._seeded():
            episodes = [self._episode(task, seq, n_train) for task in self._tasks[self.hp_dim]]
        for x, y in episodes:
            curve_id, fidelity, config = x[:, :1], x[:, 1:2], x[:, 2:]
            cols = [config, fidelity] + ([curve_id / seq] if self.include_curve_id else [])
            packed = torch.cat(cols, 1)
            pad = torch.zeros(seq, self.d - packed.shape[1])
            if self.include_curve_id:  # keep the id last: real columns first, then padding, then id
                packed = torch.cat([config, fidelity, pad, cols[-1]], 1)
            else:
                packed = torch.cat([packed, pad], 1)
            xs.append(packed)
            ys.append(1.0 - y if self.minimize else y)
        x, y = torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)
        return x[:, :n_train], y[:, :n_train], x[:, n_train:], y[:, n_train:]

    def _episode(self, task: MultiFidelityTask, seq: int, n_train: int) -> tuple[torch.Tensor, torch.Tensor]:
        alloc = AllocationPrior(seq, self.n_levels)
        allocation = alloc.sample_abstract_allocation(n_train)
        configs = np.random.uniform(size=(seq, self.hp_dim))  # noqa: NPY002  (as in ifBO; seeded by _seeded)
        curves = task.get_marginal_curve(torch.from_numpy(configs).float())
        return alloc.parse_allocation_into_sequence(configs, curves, self.hp_dim, n_train, allocation)
