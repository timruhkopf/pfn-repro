# pfn-repro

> Reproducible PFN surrogates, priors and amortized acquisitions for Bayesian optimization, with pinned
> checkpoints, to be used as a dependency by research repos.

**How we work:** [docs/process.md](docs/process.md) · **Using it from another repo:** [docs/dependency_guide.md](docs/dependency_guide.md) · **Checkpoints:** [docs/checkpoints.md](docs/checkpoints.md) · **Changes:** [CHANGELOG.md](CHANGELOG.md)

## What is in it

| Concern | Module | Contents |
|---|---|---|
| priors | `pfn_repro.priors` | `bnn.BNNPrior` (PFNs4BO-style BO functions, ECDF-scaled y), `mf.MFPrior` (ifBO's learning-curve prior) |
| surrogates | `pfn_repro.surrogates` | `PFN` (row tokens, train↔train + test→train attention), `TabPFN` (TabPFN-v2-style cell tokens, feature + item attention, thinking rows) |
| acquisitions | `pfn_repro.acquisitions` | `standard` (PI, EI, UCB, MES, std from the bar distribution), `alpha_pfn` (amortized Entropy Search: conditioned PFN, targets, trainers) |
| objectives | `pfn_repro.objectives` | `BarDistribution` (piecewise-uniform output distribution: NLL, mean, quantiles, EI/PI) |
| checkpoints | `pfn_repro.checkpoints` | registry pinned by sha256: `pfn_smoke`, `pfn_large`, `alpha_conditioned_large`, `alpha_pes_large` |

Every prior trains every surrogate (`training.trainer.PFNTrainer`), and every surrogate has the same call signature.

```python
from pfn_repro.checkpoints import load, make_prior
from pfn_repro.acquisitions.standard import arm_scores

pfn, _ = load("pfn_large")                                 # frozen, sha256-verified
prior = make_prior(x_dim=2, batch_size=4, seed=0)          # BNN prior on the checkpoints' ECDF scale
x_tr, y_tr, x_te, _ = prior.sample_episode(10, 256)
logits = pfn(x_tr, y_tr, x_te, n_features=prior.active_dim)
ei = arm_scores("ei", pfn.bar_dist, logits, y_tr.min(1).values, ucb_quantile=0.95)   # [4, 256]
```

## Notebooks

| Notebook | What |
|---|---|
| `01_*_bnn_prior_1d` | the BNN prior in 1-d: draws, noise, the ECDF scale, how to use it |
| `02_*_mf_prior_1d` | the multi-fidelity prior: learning curves over a 1-d configuration, how a context is allocated |
| `03_*_pfn_quality` | `pfn_large` vs `pfn_smoke`: NLL, calibration, ranking, cost; 1-d PPD heatmaps |
| `04_*_alpha_pfn` | the alpha-PFN pair: training, gates, 1-d behaviour, agreement with standard acquisitions |

## Quickstart

```bash
git clone https://github.com/timruhkopf/pfn-repro.git && cd pfn-repro
just setup cpu local
just test                    # golden outputs need the large checkpoints, else they skip
uv run pfn_repro experiment=train_smoke prior=mf surrogate=tabpfn      # a few training steps (scratch)
```

The large checkpoints are not in git; see [docs/checkpoints.md](docs/checkpoints.md).

## Citation

```bibtex
@misc{ruhkopf2026pfnrepro,
  title  = {pfn-repro: reproducible PFN surrogates, priors and amortized acquisitions},
  author = {Ruhkopf, Tim},
  year   = {2026},
  url    = {https://github.com/timruhkopf/pfn-repro}
}
```

Builds on PFNs4BO (Müller et al. 2023), ifBO (Rakotoarison et al. 2024), TabPFN v2 (Hollmann et al.
2025) and alpha-PFN (Rakotoarison et al. 2026). MIT, see [LICENSE](LICENSE).
