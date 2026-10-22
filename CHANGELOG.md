# Changelog

Semantic versioning on the public surface: import paths, registry names, golden outputs
(docs/dependency_guide.md). Sections: Breaking / Added / Fixed.

## v0.2.0 (2026-10-22)

### Added
- Checkpoint `tabpfn_bnn_pilot`: the TabPFN-v2-style surrogate trained overnight on the BNN prior
  (issue #1; sweep `pilot_20261021-142330` at 6567285, 100,115 steps, MLflow run 315b763c). A pilot-stage
  model: its comparison with `pfn_large` (`eval_pfn`) is pending. Golden outputs pinned.

## v0.1.0 (2026-10-21)

### Added
- Ported from pfn-acqpolicy@980fad9: the PFN (`surrogates.pfn`), BNN prior (`priors.bnn`), bar
  distribution (`objectives`), trainer, alpha-PFN stages (`acquisitions.alpha_pfn`), standard
  acquisitions, predictive metrics.
- The multi-fidelity prior from Meta_FTPFN (`priors.mf`, decision 0002).
- A TabPFN-v2-style surrogate with feature/item attention and thinking rows (`surrogates.tabpfn`).
- Checkpoint registry with sha256 pins: `pfn_smoke` (packaged), `pfn_large`, `alpha_conditioned_large`,
  `alpha_pes_large`; golden-output tests.
- Pipelines: `train_pfn` (any prior × surrogate), `train_alpha`, `eval_pfn`, `eval_alpha`.
