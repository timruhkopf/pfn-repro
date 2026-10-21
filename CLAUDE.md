# pfn-repro

Reproducible PFN surrogates, priors and amortized acquisitions with pinned checkpoints, used as a dependency
by research repos (pfn-acqpolicy, …).

@docs/agent_rules.md

## Current phase

v0.1.0: the port from pfn-acqpolicy@980fad9 and Meta_FTPFN. Its sanity check: `eval_pfn_large` and
`eval_alpha_large` reproduce the original numbers (log entry `research_log/entries/*_eval-pfn-large.md`).

## Project rules

- This is a library: its public surface (import paths, registry names, golden outputs in
  `tests/unit/test_checkpoints.py`) is versioned. A change that alters it is breaking: bump, CHANGELOG
  entry, release (docs/dependency_guide.md).
- FROZEN, never edit (the checkpoints depend on them): `surrogates/pfn.py`, `priors/bnn.py`,
  `objectives/bar_distribution.py`, `acquisitions/alpha_pfn/conditioned.py`. Other ported files keep
  their source header; they are excluded from ruff/pyright to stay diffable.
- Registered checkpoint names never change meaning; new weights get a new name (docs/checkpoints.md).
- y is on the training prior's ECDF scale: every BNN evaluation prior uses `load_ecdf()` / `make_prior()`.
- Always pass `n_features` to surrogates. Stay inside a checkpoint's training context range
  (`REGISTRY[name].train_n`); runs beyond it are robustness variants.
- Minimization everywhere (the MF prior flips performance to `1 - perf`).
