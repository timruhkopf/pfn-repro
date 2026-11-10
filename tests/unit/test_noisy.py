"""Noisy BNN prior, two-headed PFN, their trainer and the stage-0 probes (issue #3)."""

import numpy as np
import pytest
import torch

from pfn_repro.checkpoints import load_pfn_file
from pfn_repro.priors.noisy_bnn import FAMILIES, REGIMES, NoisyBNNPrior
from pfn_repro.surrogates import SURROGATES
from pfn_repro.surrogates.two_head_pfn import TwoHeadPFN
from pfn_repro.tasks.eval_noisy_pfn import balanced_accuracy, fit_logistic, fit_ridge
from pfn_repro.training.noisy_trainer import NoisyPFNTrainer


def prior(batch: int = 8, d: int = 2, seed: int = 0, **kw) -> NoisyBNNPrior:
    return NoisyBNNPrior(batch_size=batch, x_dim=d, seed=seed, **kw)


def test_episode_shapes_and_bounds() -> None:
    p = prior()
    e = p.sample_noisy_episode(20, 30)
    assert e.x_train.shape == (8, 20, 2) and e.x_test.shape == (8, 30, 2)
    assert bool(((e.x_train >= 0) & (e.x_train <= 1)).all() and ((e.x_test >= 0) & (e.x_test <= 1)).all())
    assert bool(((e.f_train >= 0) & (e.f_train <= 1)).all())
    assert torch.allclose(p.latent(e.x_train), e.f_train)


def test_seeded_prior_is_reproducible() -> None:
    a, b = prior(seed=3).sample_noisy_episode(10, 5), prior(seed=3).sample_noisy_episode(10, 5)
    assert torch.equal(a.y_train, b.y_train) and torch.equal(a.x_test, b.x_test)


def test_noise_free_regime_observes_the_latent() -> None:
    p = prior()
    p.reset(regime="free")
    e = p.sample_noisy_episode(10, 10)
    assert torch.equal(e.y_train, e.f_train) and torch.equal(e.y_test, e.f_test)


@pytest.mark.parametrize("family", FAMILIES)
def test_unit_noise_has_robust_scale_one_and_mean_zero(family: str) -> None:
    p = prior(batch=4)
    p.reset(regime="homo", family=family)
    u = torch.rand(4, 200_000, generator=torch.Generator().manual_seed(0))
    eps = p.unit_noise(u)
    q25, q75 = eps.quantile(0.25, dim=1), eps.quantile(0.75, dim=1)
    assert torch.allclose((q75 - q25) / 1.349, torch.ones(4), atol=0.02)
    assert torch.allclose(eps.mean(1), torch.zeros(4), atol=0.03)  # tails beyond 1e-4 are truncated


def test_noise_scale_follows_the_snr() -> None:
    p = prior(batch=16)
    p.reset(regime="homo", family="gauss", snr=4.0)
    x = torch.rand(16, 50_000, 2, generator=torch.Generator().manual_seed(1))
    resid = p.observe(x) - p.latent(x)
    robust = (resid.quantile(0.75, dim=1) - resid.quantile(0.25, dim=1)) / 1.349
    assert torch.allclose(robust, p.gap / 4.0, rtol=0.05)


def test_heteroscedastic_scale_varies_and_homoscedastic_does_not() -> None:
    p = prior(batch=8)
    x = torch.rand(8, 500, 2, generator=torch.Generator().manual_seed(2))
    p.reset(regime="homo")
    s = p.noise_scale(x)
    assert torch.allclose(s.std(1), torch.zeros(8), atol=1e-6)
    p.reset(regime="hetero")
    s = p.noise_scale(x)
    assert bool((s.max(1).values / s.min(1).values > 1.0).all())
    assert bool((s.max(1).values / s.min(1).values <= np.exp(2 * p.hetero_amp_max * 6)).all())


def test_contamination_only_in_the_contaminated_family() -> None:
    p = prior(batch=4)
    p.reset(regime="homo", family="gauss")
    assert bool((p.contam_rate == 0).all())
    p.reset(regime="homo", family="contaminated")
    assert bool(((p.contam_rate >= 0.01) & (p.contam_rate <= 0.1)).all())


def test_common_random_numbers_fix_the_noise() -> None:
    p = prior(batch=4)
    p.reset(regime="hetero", family="student_t")
    x = torch.rand(4, 7, 2)
    u = torch.rand(4, 7, 3)
    assert torch.equal(p.observe(x, u=u), p.observe(x, u=u))


def test_relevance_gives_at_least_one_relevant_input() -> None:
    p = prior(batch=256, d=4, n_ref=16)
    assert bool(((p.d_eff >= 1) & (p.d_eff <= 4)).all())
    assert bool(((p.relevance == 1.0).sum(1) == p.d_eff).all())
    assert set(p.d_eff.tolist()) == {1, 2, 3, 4}


def test_regime_and_family_frequencies() -> None:
    p = prior(
        batch=4000, n_ref=16, depth_range=(8, 9), width_range=(32, 33), ecdf_sorted=torch.linspace(-2, 2, 50)[None]
    )
    reg = np.bincount(p.regime.numpy(), minlength=3) / 4000
    fam = np.bincount(p.family.numpy(), minlength=4) / 4000
    assert np.allclose(reg, [0.2, 0.4, 0.4], atol=0.03) and np.allclose(fam, [0.6, 0.2, 0.1, 0.1], atol=0.03)


def test_contexts_contain_replicates_and_local_points() -> None:
    p = prior(batch=64, p_bo_context=1.0, p_replicates=1.0, replicate_frac_max=0.3)
    e = p.sample_noisy_episode(40, 4)
    dup = (torch.cdist(e.x_train, e.x_train) == 0).float().sum((1, 2)) - 40  # exact pairs off the diagonal
    assert float((dup > 0).float().mean()) > 0.5


def two_head(d: int = 2) -> TwoHeadPFN:
    torch.manual_seed(0)
    return TwoHeadPFN(max_x_dim=d, d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins=32)


def test_two_head_interface_and_invariances() -> None:
    p = prior(batch=3)
    e = p.sample_noisy_episode(12, 9)
    m = two_head()
    lat, noi = m.forward_both(e.x_train, e.y_train, e.x_test, p.active_dim)
    assert lat.shape == (3, 9, 32) and noi.shape == (3, 9, m.noisy_bar_dist.num_bars)
    assert torch.equal(m(e.x_train, e.y_train, e.x_test, p.active_dim), lat)
    perm = torch.randperm(12)
    assert torch.allclose(m(e.x_train[:, perm], e.y_train[:, perm], e.x_test, p.active_dim), lat, atol=1e-5)
    kv = m.context_kv(e.x_train, e.y_train, p.active_dim)
    lat2, noi2 = m.heads_from_context_kv(kv, e.x_test, p.active_dim)
    assert torch.allclose(lat, lat2, atol=1e-5) and torch.allclose(noi, noi2, atol=1e-5)
    # test points do not see each other
    x2 = e.x_test.clone()
    x2[:, 0] = 0.5
    assert torch.allclose(m(e.x_train, e.y_train, x2, p.active_dim)[:, 1:], lat[:, 1:], atol=1e-6)
    assert float(m.noisy_bar_dist.borders[0]) == -1.0 and float(m.noisy_bar_dist.borders[-1]) == 2.0


def test_trainer_reduces_both_nlls_and_checkpoint_loads(tmp_path) -> None:
    p = prior(batch=16)
    m = two_head()
    path = tmp_path / "ckpt.pt"
    out = NoisyPFNTrainer(
        prior=p,
        model=m,
        seed=0,
        n_steps=120,
        min_train=2,
        max_train=15,
        n_test=32,
        lr=3e-3,
        warmup_steps=10,
        log_every=20,
        checkpoint_path=path,
        model_config=dict(max_x_dim=2, d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins=32),
        extra_checkpoint_metadata={"surrogate": "pfn_repro.surrogates.two_head_pfn.TwoHeadPFN"},
    ).run()
    h = out["history"]
    assert h["nll_latent/train"][-1] < h["nll_latent/train"][0]
    assert h["nll_noisy/train"][-1] < h["nll_noisy/train"][0]
    loaded, _ = load_pfn_file(path)
    assert isinstance(loaded, TwoHeadPFN)
    assert "pfn_repro.surrogates.two_head_pfn.TwoHeadPFN" in SURROGATES


def test_probes_recover_planted_signal_and_nulls_stay_at_chance() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    X = np.c_[y + rng.normal(0, 0.3, 2000), rng.normal(size=(2000, 3))]
    pred = fit_logistic(X[:1000], y[:1000], X[1000:], 3)
    assert balanced_accuracy(y[1000:], pred) > 0.9
    pred_null = fit_logistic(X[:1000, 1:], y[:1000], X[1000:, 1:], 3)
    assert balanced_accuracy(y[1000:], pred_null) < 0.45
    t = X[:, 0] * 2 + rng.normal(0, 0.1, 2000)
    r = fit_ridge(X[:1000], t[:1000], X[1000:])
    assert np.corrcoef(r, t[1000:])[0, 1] > 0.95


def test_regimes_listed_once() -> None:
    assert REGIMES == ("free", "homo", "hetero") and len(set(FAMILIES)) == 4
