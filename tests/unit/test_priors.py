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


@pytest.mark.parametrize(("n_levels", "curve_id"), [(10, False), (None, True)])
def test_fast_mf_prior_gives_the_same_episodes(n_levels: int | None, curve_id: bool) -> None:
    """FastMFPrior batches MFPrior's deterministic work; every episode stays exactly the same."""
    from pfn_repro.priors.mf import MFPrior
    from pfn_repro.priors.mf.fast import FastMFPrior

    kw = dict(batch_size=4, x_dim=2, variable_dim_min=1, n_levels=n_levels, include_curve_id=curve_id, seed=3)
    slow, fast = MFPrior(**kw), FastMFPrior(**kw)
    for n_train in (1, 6, 40):
        slow.reset()
        fast.reset()
        for u, v in zip(slow.sample_episode(n_train, 30), fast.sample_episode(n_train, 30), strict=True):
            assert torch.equal(u, v)


def test_prefetch_batches_depend_only_on_seed_and_step() -> None:
    """A worker's batch for a step does not depend on what the worker made before."""
    from pfn_repro.training import prefetch

    prefetch._init_worker(dict(name="mf", batch_size=2, x_dim=2, seed=0, variable_dim_min=1))
    a = prefetch.make_batch(0, 5, 7, 4)
    for step in (1, 2, 3):
        prefetch.make_batch(0, step, 9, 4)
    b = prefetch.make_batch(0, 5, 7, 4)
    for u, v in zip(a, b, strict=True):
        assert torch.equal(u, v)
