"""Priors: distributions over functions that the surrogates are trained on.

Both priors expose the same episode interface, which `training.trainer.PFNTrainer` consumes:
`reset()` (new functions), `sample_episode(n_train, n_test) -> x_tr, y_tr, x_te, y_te` (inputs in
[0, 1], real features packed first and zero-padded to width `d`), `active_dim` ([B] real feature
count, passed to the surrogate as `n_features`), `B`, `d` and `generator` (a `torch.Generator`
whose state the trainer checkpoints).

- `bnn.BNNPrior`: BO functions from random deep tanh networks, y on the training ECDF scale (PFNs4BO).
- `mf.MFPrior`: learning curves over (configuration, fidelity) (ifBO's FT-PFN prior); `build_prior` returns
  `mf.fast.FastMFPrior`, which gives the same episodes faster.
"""

from typing import Any


def build_prior(
    name: str,
    batch_size: int,
    x_dim: int,
    seed: int,
    device: str = "cpu",
    variable_dim_min: int | None = 1,
    **params: Any,
):
    """A prior from its config (`configs/prior/<name>.yaml`). x_dim: the maximal number of inputs of the
    function family (BNN: x dims; MF: hyperparameters, the fidelity comes on top)."""
    if name == "bnn":
        from ..checkpoints import load_ecdf
        from .bnn import BNNPrior

        return BNNPrior(
            batch_size=batch_size,
            x_dim=x_dim,
            variable_dim_min=variable_dim_min,
            seed=seed,
            ecdf_sorted=load_ecdf(device),
            device=device,
            **params,
        )
    if name == "mf":
        from .mf.fast import FastMFPrior  # MFPrior's episodes, batched (priors/mf/fast.py)

        return FastMFPrior(
            batch_size=batch_size, x_dim=x_dim, variable_dim_min=variable_dim_min, seed=seed, device=device, **params
        )
    raise ValueError(f"unknown prior {name!r}")
