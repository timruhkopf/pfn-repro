# Ported from pfn-acqpolicy@980fad9 src/pfn_acqpolicy/trainer/es_trainer.py, unchanged apart from import paths.
"""The two alpha-PFN training stages, as `PFNTrainer` subclasses (same loop, AMP,
checkpointing and resume; only the batch differs). Both draw BO-like traces and
the function's minimum (`es/episodes.py`).

- Stage 1 (`ConditionedPFNTrainer`): NLL of y at the query points, with I drawn
  per function uniformly from {none, x*, f*, both} (the paper: equally likely).
- Stage 2 (`AlphaPFNTrainer`): NLL of alpha~ = H[q(y|x,D)] - H[q(y|x,D,I)] under the
  frozen stage-1 model, for one variant I (PES: x*). The model is a plain PFN
  whose bar distribution lives on alpha~'s support.

`episodes_per_function` > 1 reuses a function batch and its minimum for that many
consecutive steps (fresh trace and context size each step): the x* search costs
more than the rest of a step. A resumed run redraws the batch at the resume step,
so unlike `PFNTrainer` it is not bit-identical to an uninterrupted run.
"""
import torch

from pfn_repro.acquisitions.alpha_pfn.episodes import sample_episode
from pfn_repro.acquisitions.alpha_pfn.targets import info_gain
from pfn_repro.acquisitions.alpha_pfn.conditioned import OptimumConditionedPFN
from pfn_repro.training.trainer import PFNTrainer


class _EpisodeTrainer(PFNTrainer):
    def __init__(self, *args, trace: dict, optimum: dict, episodes_per_function: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace, self.optimum, self.episodes_per_function = trace, optimum, episodes_per_function
        self._minimum = None

    def episode(self, step, n_train):
        reuse = self._minimum if step % self.episodes_per_function else None
        ep = sample_episode(self.prior, n_train, self.n_test, self.trace, self.optimum, seed=step, minimum=reuse)
        self._minimum = (ep.x_opt, ep.f_opt)
        return ep


class ConditionedPFNTrainer(_EpisodeTrainer):
    def sample_batch(self, step, n_train, generator):
        ep = self.episode(step, n_train)
        case = torch.randint(0, 4, (self.prior.B,), generator=generator).to(ep.y_tr.device)
        batch = dict(x_train=ep.x_tr, y_train=ep.y_tr, x_test=ep.x_te, x_opt=ep.x_opt, f_opt=ep.f_opt,
                     has_x=(case == 1) | (case == 3), has_f=(case == 2) | (case == 3), n_features=ep.n_features)
        return batch, ep.y_te


class AlphaPFNTrainer(_EpisodeTrainer):
    def __init__(self, *args, conditioned: OptimumConditionedPFN, info: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditioned, self.info = conditioned, info

    def sample_batch(self, step, n_train, generator):
        ep = self.episode(step, n_train)
        target = info_gain(self.conditioned, ep, self.info)
        return dict(x_train=ep.x_tr, y_train=ep.y_tr, x_test=ep.x_te, n_features=ep.n_features), target
