# References

Annotated literature for this project. BibTeX for every entry lives in `references.bib` under the same key; `just refs-check` verifies that both files agree.

**Status values:** `to-read` · `skimmed` · `read` · `cited` · `to-verify` (id or BibTeX not yet checked against the source)

**Entry template**

```markdown
### [key] Authors (Year). *Title*. Venue. DOI or arXiv id
- **Status:**
- **Claim:** what the paper shows, in one sentence, in our own words.
- **Relevance:** supports / contradicts / baseline / method source / dataset / evaluation — and why.
- **Used in:** configs, code modules, research log entries that depend on it.
- **Notes:** setup details we must match (splits, metric definitions, seeds, compute).
```

---

## Problem & motivation

<!-- Papers that establish why the question matters. -->

## Closest related work

<!-- What the final paper must position itself against. One line per paper on how we differ. -->

## Method building blocks

<!-- Sources of components we use or adapt. Link the module under "Used in". -->

### [bergna2026decoupledpfnsidentifiableepistemicaleatoric] Bergna, Depeweg, Hernández-Lobato (2026). *Decoupled PFNs: Identifiable Epistemic-Aleatoric Decomposition via Structured Synthetic Priors*. arXiv:2605.06413
- **Status:** abstract read (arXiv abs page, 2026-11-10); full paper not yet read.
- **Claim:** the epistemic/aleatoric split is not identifiable from the posterior predictive alone, even an
  exact one. A synthetic prior with an explicit latent signal and noise function lets the generator
  supply query-level targets for both. A PFN trained on them with separate latent-signal and noise heads
  supports epistemic-only acquisition, which avoids the failure of total-variance exploration under
  noisy and heteroscedastic observations (best average rank in HPO and synthetic BO).
- **Relevance:** method source for the two-headed PFN and the noisy prior (issue #3).
- **Used in:** `priors/noisy_bnn.py`, `surrogates/two_head_pfn.py`, `training/noisy_trainer.py`,
  experiments `noisy_pfn` / `eval_noisy_pfn`.
- **Notes:** they derive the observation predictive by convolving the latent distribution with a learned
  (plug-in) noise variance. We train a separate noisy bar head instead: the exact predictive is a mixture
  over the noise level, and our noise families (Student-t, contamination, skew) are not Gaussian. Their
  signal prior is an RFF-GP / SCM mixture; ours is the PFNs4BO BNN with per-input relevance. Their noise
  settings (as summarized in our problem statement: base scale U(0.03, 0.12), heteroscedastic w.p. 0.8)
  are to be checked against the full paper.

## Baselines

<!-- Every baseline we run has a config in src/myproject/configs/ and is linked here. -->

## Datasets & benchmarks

<!-- Include license and the exact split/version we use. -->

## Evaluation & statistics

<!-- Metric definitions, statistical tests, reporting practices we follow. -->

## Tools & software

<!-- Software we should cite: PyTorch, Hydra, Optuna, MLflow, ... (use each project's recommended citation). -->

---

## Reading queue

<!-- Papers not yet placed in a section: key, one line on why it might matter. -->
