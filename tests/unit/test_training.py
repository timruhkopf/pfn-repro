import pytest
import torch

from pfn_repro.priors import build_prior
from pfn_repro.surrogates import PFN
from pfn_repro.training.trainer import PFNTrainer


class Killed(Exception):
    pass


def _train(path, kill_at: int | None = None):
    torch.manual_seed(0)
    prior = build_prior("bnn", batch_size=4, x_dim=2, seed=0)
    model = PFN(max_x_dim=2, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)

    def on_log(step: int, metrics: dict) -> None:
        if step == kill_at:
            raise Killed  # the process dies after the step-10 save, before the next one

    PFNTrainer(
        prior=prior,
        model=model,
        seed=0,
        n_steps=20,
        min_train=2,
        max_train=6,
        n_test=8,
        warmup_steps=2,
        log_every=5,
        save_every=10,
        resume=path.exists(),
        checkpoint_path=path,
        model_config={"max_x_dim": 2},
        on_log=on_log,
    ).run()
    return model


@pytest.mark.slow
def test_resume_continues_the_same_run(tmp_path) -> None:
    """Killed after its step-10 checkpoint and resumed, a run ends exactly where an uninterrupted one does."""
    straight = _train(tmp_path / "a.pt")
    with pytest.raises(Killed):
        _train(tmp_path / "b.pt", kill_at=15)
    resumed = _train(tmp_path / "b.pt")
    for p, q in zip(straight.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(p, q)


@pytest.mark.parametrize(("prior", "surrogate"), [("bnn", "tabpfn"), ("mf", "pfn"), ("mf", "tabpfn")])
def test_every_prior_trains_every_surrogate(prior: str, surrogate: str, tmp_path, monkeypatch) -> None:
    """The concerns compose: any prior with the episode interface trains any surrogate (a few steps)."""
    from pfn_repro.priors.mf.bnn import BNNPrior as LatentBNN
    from pfn_repro.surrogates import TabPFN

    monkeypatch.setenv("PFN_REPRO_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(LatentBNN, "N_datasets", 100)
    monkeypatch.setattr(LatentBNN, "_ecdfs", {})
    p = build_prior(prior, batch_size=2, x_dim=2, seed=0)
    cls = PFN if surrogate == "pfn" else TabPFN
    model = cls(max_x_dim=p.d, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    history = PFNTrainer(
        prior=p, model=model, seed=0, n_steps=3, min_train=2, max_train=5, n_test=4, warmup_steps=1, log_every=1
    ).run()["history"]
    assert all(v == v for v in history["nll/train"])  # finite
