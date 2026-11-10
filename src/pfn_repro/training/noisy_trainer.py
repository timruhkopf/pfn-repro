"""Training the two-headed PFN on the noisy prior (issue #3): latent NLL on f(x_test) + noisy NLL on y(x_test).

Both targets are drawn by the generator: the latent f is privileged (never a model input), which is
legitimate because the loss is a proper scoring rule whose minimizer is the posterior law of f given D
(the same argument that makes ordinary PFN training legitimate). Context y are noisy observations.
"""

from typing import Any

import torch

from ..priors.noisy_bnn import REGIMES, NoisyBNNPrior
from .callbacks import Callback
from .trainer import PFNTrainer


class NoisyPFNTrainer(PFNTrainer):
    def __init__(self, *args: Any, latent_weight: float = 1.0, noisy_weight: float = 1.0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.latent_weight, self.noisy_weight = latent_weight, noisy_weight

    def sample_batch(self, step: int, n_train: int, generator: torch.Generator) -> tuple[dict, torch.Tensor]:
        self.prior.reset()
        e = self.prior.sample_noisy_episode(n_train, self.n_test)
        batch = dict(x_train=e.x_train, y_train=e.y_train, x_test=e.x_test, n_features=self.prior.active_dim)
        return batch, torch.stack([e.f_test, e.y_test], -1)

    def forward(self, batch: dict):
        return self.model.forward_both(**batch)

    def _nlls(self, logits, target) -> tuple[torch.Tensor, torch.Tensor]:
        lat, noi = logits
        return (self.model.bar_dist(lat.float(), target[..., 0]).mean(),
                self.model.noisy_bar_dist(noi.float(), target[..., 1]).mean())

    def loss(self, logits, target) -> torch.Tensor:
        nll_lat, nll_noi = self._nlls(logits, target)
        return self.latent_weight * nll_lat + self.noisy_weight * nll_noi

    def train_metrics(self, logits, target) -> dict:
        nll_lat, nll_noi = self._nlls(logits, target)
        mse = (self.model.bar_dist.mean(logits[0].float()) - target[..., 0]).square().mean()
        return {"nll_latent/train": nll_lat.item(), "nll_noisy/train": nll_noi.item(), "mse_latent/train": mse.item()}


def build_regime_validation_callback(prior_kwargs: dict, x_dim: int, ecdf_sorted: torch.Tensor,
                                     context_sizes: tuple[int, ...] = (5, 20, 50), n_functions: int = 64,
                                     n_points: int = 200, seed: int = 20_000, every_n_steps: int | None = None,
                                     device: str = "cpu") -> Callback:
    """Fixed held-out episodes per noise regime (functions, contexts and points drawn once):
    `nll_latent/val_{regime}_n{n}`, `nll_noisy/val_{regime}_n{n}`. Context size n uses the first n
    context points (a prefix of the generating sequence, so BO-like contexts stay BO-like)."""
    n_max = max(context_sizes)
    sets = {}
    for i, regime in enumerate(REGIMES):
        prior = NoisyBNNPrior(batch_size=n_functions, x_dim=x_dim, seed=seed + i, device=device,
                              ecdf_sorted=ecdf_sorted, **prior_kwargs)
        prior.reset(regime=regime)
        sets[regime] = (prior.sample_noisy_episode(n_max, n_points), prior.active_dim)

    def probe(step: int, trainer: Any) -> dict:
        out = {}
        m = trainer.model
        with torch.no_grad():
            for regime, (e, nf) in sets.items():
                for n in context_sizes:
                    lat, noi = m.forward_both(e.x_train[:, :n], e.y_train[:, :n], e.x_test, n_features=nf)
                    out[f"nll_latent/val_{regime}_n{n}"] = m.bar_dist(lat.float(), e.f_test).mean().item()
                    out[f"nll_noisy/val_{regime}_n{n}"] = m.noisy_bar_dist(noi.float(), e.y_test).mean().item()
        return out

    return Callback(name="", fn=probe, every_n_steps=every_n_steps)
