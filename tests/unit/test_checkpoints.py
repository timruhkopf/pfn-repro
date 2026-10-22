"""The registered checkpoints load and give the same outputs as when they were registered.

The golden values are the contract with dependent repositories: if one changes, behaviour changed, and
that is a breaking change of this package (see docs/dependency_guide.md).
"""

import pytest
import torch

from pfn_repro.checkpoints import ECDF, REGISTRY, available, load, load_ecdf, locate, sha256

GOLDEN_MEAN = {  # posterior mean on the fixed input below
    "pfn_smoke": [0.789634, 0.683714, 0.674171, 0.676689, 0.798535, 0.510211, 0.718502, 0.65644],
    "pfn_large": [0.862092, 0.686324, 0.462559, 0.620298, 0.524391, 0.630577, 0.571432, 0.511699],
    "alpha_pes_large": [0.11717, 0.086715, 0.214902, 0.090865, 0.069107, 0.13975, 0.037202, 0.087072],
    "tabpfn_bnn_pilot": [0.850763, 0.67664, 0.486804, 0.678669, 0.515473, 0.703518, 0.573948, 0.460899],
}


def _fixed_input():
    g = torch.Generator().manual_seed(123)
    return torch.rand(2, 8, 3, generator=g), torch.rand(2, 8, generator=g), torch.rand(2, 4, 3, generator=g)


@pytest.mark.parametrize("name", sorted(GOLDEN_MEAN))
def test_golden_outputs(name: str) -> None:
    if not available()[name]:
        pytest.skip(f"{name} is not on this machine (large checkpoints are not in git)")
    model, _ = load(name)
    x_tr, y_tr, x_te = _fixed_input()
    logits = model(x_tr, y_tr, x_te, n_features=torch.tensor([3, 2])).float()
    torch.testing.assert_close(
        model.bar_dist.mean(logits).flatten(), torch.tensor(GOLDEN_MEAN[name]), atol=1e-5, rtol=1e-5
    )


def test_packaged_files_match_registry() -> None:
    assert sha256(locate(ECDF.name)) == ECDF.sha256
    assert sha256(locate("pfn_smoke")) == REGISTRY["pfn_smoke"].sha256
    ecdf = load_ecdf()
    assert ecdf.shape[0] == 1
    assert bool((ecdf[:, 1:] >= ecdf[:, :-1]).all())  # sorted


def test_loaded_models_are_frozen() -> None:
    model, ckpt = load("pfn_smoke")
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert ckpt["config"]["max_x_dim"] == 6


def test_tampered_file_is_refused(tmp_path, monkeypatch) -> None:
    import shutil

    shutil.copy(locate("pfn_smoke"), tmp_path / "pfn_large.pt")  # wrong bytes under a registered name
    monkeypatch.setenv("PFN_REPRO_CKPT_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        load("pfn_large")


def test_missing_file_says_where_to_put_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PFN_REPRO_CKPT_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="PFN_REPRO_CKPT_ROOT"):
        load("alpha_pes_large")
