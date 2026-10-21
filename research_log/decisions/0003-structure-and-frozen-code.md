# 0003: Package structure, frozen code, TabPFN-v2-style surrogate

**Structure (separation of concerns).**
- `priors/`: what functions look like: `bnn` (BO functions) and `mf` (learning curves). Both have
  one episode interface.
- `surrogates/`: map (context, query) to bar-distribution logits: `pfn.PFN` (PFNs4BO-style, row
  tokens) and `tabpfn.TabPFN` (TabPFN-v2-style, cell tokens).
- `acquisitions/`: turn predictions into choices: `standard` (PI/EI/UCB/MES/std from the bar
  distribution) and `alpha_pfn` (amortized Entropy Search: conditioned PFN, episodes, targets, trainers).
- `objectives/`: the output distribution and its loss: `bar_distribution`.
- `training/` (the shared trainer, callbacks, validation), `evaluation/` (predictive metrics),
  `checkpoints.py` (the registry). `tasks/` holds the pipelines only.

**Frozen code.** `surrogates/pfn.py`, `priors/bnn.py`, `objectives/bar_distribution.py` and
`acquisitions/alpha_pfn/conditioned.py` back the registered checkpoints. They are kept verbatim from
pfn-acqpolicy@980fad9 apart from import paths, and are excluded from ruff/pyright so they stay diffable
against the source. The golden-output test pins their behaviour. The other ported files carry their
source in a header; the trainer's type hints were widened so it trains any prior × surrogate.

**TabPFN v2.** Own compact implementation (`surrogates/tabpfn.py`), not a wrapper around PriorLabs'
`tabpfn` package: no external dependency or licence question, and it uses the same interface as the PFN.
Deviations from TabPFN v2:
- learned per-feature-index embeddings instead of random ones (in BO a column keeps its meaning);
- no feature grouping;
- no preprocessing ensemble.

Thinking rows are learned rows prepended to the context. Tests check permutation invariance, no
test-test leakage and invisible padding, as for the PFN. No checkpoint is trained yet.
