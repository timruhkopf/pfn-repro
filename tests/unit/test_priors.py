import numpy as np
import pytest
import torch

from pfn_repro.priors import build_prior
from pfn_repro.priors.mf.allocation import AllocationPrior
from pfn_repro.priors.mf.bnn import BNNPrior as MFBNN


@pytest.fixture(autouse=True)
def small_mf_ecdf(tmp_path, monkeypatch):
    """The MF prior's latent ECDF: 200 draws into a temporary cache instead of 10,000 into data/."""
    monkeypatch.setenv("PFN_REPRO_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(MFBNN, "N_datasets", 200)
    monkeypatch.setattr(MFBNN, "_ecdfs", {})


@pytest.mark.parametrize("name", ["bnn", "mf"])
def test_episode_interface(name: str) -> None:
    prior = build_prior(name, batch_size=3, x_dim=2, seed=0, variable_dim_min=1)
    x_tr, y_tr, x_te, y_te = prior.sample_episode(7, 5)
    assert x_tr.shape == (3, 7, prior.d) and x_te.shape == (3, 5, prior.d)
    assert y_tr.shape == (3, 7) and y_te.shape == (3, 5)
    assert bool(((x_tr >= 0) & (x_tr <= 1)).all())
    assert bool(((y_tr >= 0) & (y_tr <= 1)).all())  # ECDF scale / performance
    n_real = int(prior.active_dim[0])
    assert bool((x_tr[..., n_real:] == 0).all())  # real features packed first, the rest zero


@pytest.mark.parametrize("name", ["bnn", "mf"])
def test_seeded_priors_are_reproducible(name: str) -> None:
    a = build_prior(name, batch_size=2, x_dim=2, seed=7).sample_episode(5, 3)
    b = build_prior(name, batch_size=2, x_dim=2, seed=7).sample_episode(5, 3)
    for u, v in zip(a, b, strict=True):
        torch.testing.assert_close(u, v)


def test_mf_context_is_freeze_thaw_shaped() -> None:
    """Each curve's observations start at the first fidelity level (ifBO's allocation)."""
    prior = build_prior("mf", batch_size=2, x_dim=1, seed=0, variable_dim_min=None, n_levels=10)
    x_tr, _, _, _ = prior.sample_episode(30, 10)
    fid = x_tr[..., 1]  # [config, fidelity]
    assert bool((fid > 0).all() and (fid <= 1).all())
    assert bool(torch.isclose(fid.min(), torch.tensor(0.1)))


def test_allocation_counts() -> None:
    np.random.seed(0)  # noqa: NPY002  (the ported allocation prior uses the global RNG)
    alloc = AllocationPrior(seq_len=200, n_levels=10)
    cutoff, epochs, ordering = alloc.sample_abstract_allocation(single_eval_pos=80)
    assert cutoff.sum() == 80 and epochs.sum() == 200
    assert bool((cutoff <= epochs).all())
    assert len(ordering) == 200
