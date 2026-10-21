"""Structural properties every surrogate must have: permutation invariance over the context, no leakage
between test points, invisible padding (rows and features), and the shared output interface."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from pfn_repro.surrogates import PFN, TabPFN

SMALL = dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins=16)


def _models(max_x_dim: int = 4):
    torch.manual_seed(0)
    return {
        "pfn": PFN(max_x_dim=max_x_dim, **SMALL).eval(),
        "tabpfn": TabPFN(max_x_dim=max_x_dim, n_thinking_rows=3, **SMALL).eval(),
    }


MODELS = _models()


def _batch(seed: int, B: int = 2, N: int = 6, M: int = 5, D: int = 3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, N, D, generator=g), torch.rand(B, N, generator=g), torch.rand(B, M, D, generator=g)


@pytest.mark.parametrize("name", MODELS)
@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_context_permutation_invariance(name: str, seed: int) -> None:
    m = MODELS[name]
    x_tr, y_tr, x_te = _batch(seed)
    nf = torch.tensor([3, 2])
    perm = torch.randperm(x_tr.shape[1], generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(
        m(x_tr, y_tr, x_te, n_features=nf), m(x_tr[:, perm], y_tr[:, perm], x_te, n_features=nf), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("name", MODELS)
def test_no_leakage_between_test_points(name: str) -> None:
    m = MODELS[name]
    x_tr, y_tr, x_te = _batch(1)
    nf = torch.tensor([3, 3])
    moved = x_te.clone()
    moved[:, 0] += 1.0
    torch.testing.assert_close(m(x_tr, y_tr, x_te, n_features=nf)[:, 1:], m(x_tr, y_tr, moved, n_features=nf)[:, 1:])


@pytest.mark.parametrize("name", MODELS)
def test_padded_features_are_invisible(name: str) -> None:
    m = MODELS[name]
    x_tr, y_tr, x_te = _batch(2)
    nf = torch.tensor([3, 2])
    junk_tr, junk_te = x_tr.clone(), x_te.clone()
    junk_tr[1, :, 2], junk_te[1, :, 2] = 99.0, -5.0  # item 1 has 2 real features
    torch.testing.assert_close(m(x_tr, y_tr, x_te, n_features=nf)[1], m(junk_tr, y_tr, junk_te, n_features=nf)[1])


@pytest.mark.parametrize("name", MODELS)
def test_padded_context_rows_are_invisible(name: str) -> None:
    m = MODELS[name]
    x_tr, y_tr, x_te = _batch(3)
    nf = torch.tensor([3, 3])
    mask = torch.ones(2, x_tr.shape[1], dtype=torch.bool)
    mask[1, 4:] = False
    padded = m(x_tr, y_tr, x_te, n_features=nf, train_key_padding_mask=mask)[1]
    torch.testing.assert_close(
        padded, m(x_tr[1:, :4], y_tr[1:, :4], x_te[1:], n_features=nf[1:])[0], atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("name", MODELS)
def test_output_interface(name: str) -> None:
    m = MODELS[name]
    x_tr, y_tr, x_te = _batch(4)
    logits = m(x_tr, y_tr, x_te, n_features=torch.tensor([3, 3]))
    assert logits.shape == (2, 5, 16)
    assert m.bar_dist.mean(logits).shape == (2, 5)


def test_thinking_rows_are_parameters() -> None:
    a, b = TabPFN(max_x_dim=2, n_thinking_rows=0, **SMALL), TabPFN(max_x_dim=2, n_thinking_rows=4, **SMALL)
    extra = sum(p.numel() for p in b.parameters()) - sum(p.numel() for p in a.parameters())
    assert extra == 4 * 3 * SMALL["d_model"]  # rows x (features + target) x d_model


@pytest.mark.parametrize("cls", [PFN, TabPFN])
def test_learns_a_simple_function(cls) -> None:
    """A few hundred steps on y = x fit better than the start: the surrogate is trainable end to end."""
    torch.manual_seed(0)
    m = cls(max_x_dim=1, **SMALL)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)

    def loss() -> torch.Tensor:
        x_tr, x_te = torch.rand(16, 8, 1, generator=g), torch.rand(16, 8, 1, generator=g)
        return m.bar_dist(m(x_tr, x_tr[..., 0], x_te), x_te[..., 0]).mean()

    with torch.no_grad():
        start = loss().item()
    for _ in range(150):
        opt.zero_grad()
        loss().backward()
        opt.step()
    with torch.no_grad():
        assert loss().item() < start - 0.5
