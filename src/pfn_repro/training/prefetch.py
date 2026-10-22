"""`PrefetchTrainer`: `PFNTrainer` with the prior's batches generated ahead in worker processes.

For priors whose sampling costs more than a training step (the MF prior: ~0.5 s per batch of 64 on one
core, against ~0.06 s for a pfn_large-sized step). Each step's batch is a pure function of (seed, step,
n_train): the worker reseeds its own prior from `SeedSequence([seed, step])` before `reset()` and
`sample_episode()`, so the episodes do not depend on the number of workers or on which worker made them,
and a resumed run gets the same batches as an uninterrupted one. The distribution is the prior's; the
stream differs from the serial trainer's (which advances one prior generator through all steps).

The n_train of the next steps is known ahead: `PFNTrainer.run` draws exactly one `randint` per step from
its generator, so a copy of that generator replays them.
"""

import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

import numpy as np
import torch

from .trainer import PFNTrainer

_prior: Any = None


def _init_worker(prior_kwargs: dict[str, Any]) -> None:
    global _prior
    torch.set_num_threads(1)
    from ..priors import build_prior

    _prior = build_prior(**prior_kwargs, device="cpu")
    tasks = getattr(_prior, "_tasks", None)
    if tasks is not None:  # MF prior: create every hp_dim's tasks now, so no episode pays for (or depends on) it
        from ..priors.mf.episodes import N_LATENT
        from ..priors.mf.task import MultiFidelityTask

        for d in range((_prior.variable_dim_min or _prior.x_dim), _prior.x_dim + 1):
            if d not in tasks:
                _prior.hp_dim = d
                with _prior._seeded():
                    tasks[d] = [MultiFidelityTask(d, N_LATENT) for _ in range(_prior.B)]


def step_seed(seed: int, step: int) -> int:
    return int(np.random.SeedSequence([seed, step]).generate_state(1)[0])


def make_batch(seed: int, step: int, n_train: int, n_test: int) -> tuple[torch.Tensor, ...]:
    """The batch of `step` (in a worker): x_tr, y_tr, x_te, y_te, active_dim."""
    _prior.generator.manual_seed(step_seed(seed, step))
    _prior.reset()
    return (*_prior.sample_episode(n_train, n_test), _prior.active_dim)


class PrefetchTrainer(PFNTrainer):
    def __init__(self, *args: Any, prior_kwargs: dict[str, Any], workers: int = 8, depth: int | None = None,
                 **kwargs: Any):
        """prior_kwargs: `build_prior`'s arguments (without device) for the workers' priors."""
        super().__init__(*args, **kwargs)
        self.prior_kwargs = prior_kwargs
        self.workers = workers
        self.depth = depth or 2 * workers
        self._pool: ProcessPoolExecutor | None = None
        self._pending: dict[int, tuple[int, Future]] = {}

    def _submit(self, step: int, n_train: int) -> None:
        assert self._pool is not None
        self._pending[step] = (n_train, self._pool.submit(make_batch, self.seed, step, n_train, self.n_test))

    def sample_batch(self, step: int, n_train: int, generator: torch.Generator) -> tuple[dict, torch.Tensor]:
        if self._pool is None:
            ctx = mp.get_context("spawn")  # no fork after CUDA init
            self._pool = ProcessPoolExecutor(self.workers, ctx, initializer=_init_worker,
                                             initargs=(self.prior_kwargs,))
        ahead = torch.Generator()
        ahead.set_state(generator.get_state())  # replays run()'s n_train draws of the next steps
        for s in range(step + 1, min(step + 1 + self.depth, self.n_steps)):
            n = int(torch.randint(self.min_train, self.max_train + 1, (1,), generator=ahead).item())
            if s not in self._pending:
                self._submit(s, n)
        n_pending, future = self._pending.pop(step, (None, None))
        if future is None or n_pending != n_train:  # first step, or a schedule change (resume): make it now
            self._submit(step, n_train)
            n_pending, future = self._pending.pop(step)
        for s in [s for s in self._pending if s < step]:
            self._pending.pop(s)[1].cancel()
        dev = next(self.model.parameters()).device
        x_tr, y_tr, x_te, y_te, nf = (t.to(dev) for t in future.result())
        return dict(x_train=x_tr, y_train=y_tr, x_test=x_te, n_features=nf), y_te

    def run(self) -> dict:
        try:
            return super().run()
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None
