# Track record

**Append-only.** New entries go at the bottom; old entries are never edited, a later entry corrects an
earlier one explicitly. Times are local. Each entry names the commit it refers to, what was done, what
was **verified and how**, and numbers with the snapshot or artifact they come from.

Autonomous agent sessions start with their mandate (goal, deadline, which launches are allowed) and end
with deliverables and open questions.

## Entries

### 2026-10-21 14:30 — port done, launching the reproduction checks (012178a)

- Mandate (Tim): a standalone, reusable PFN repo from research-template: pfn_large and both alpha-PFN
  checkpoints, the PFN / alpha-PFN analysis notebooks (with 1-d PPD heatmaps), the BNN and MF priors
  (with 1-d notebooks), architecture, bar distribution, training pipeline; a dependency guide.
- Verified locally: all four checkpoints load and match their sha256; 32 unit tests (golden outputs,
  surrogate invariances, exact resume, every prior x surrogate trains); ruff and pyright clean.
- ulysses: `just remote-setup` (git over ssh, cu126 venv); checkpoints copied from
  pfn-acqpolicy/models and verified there.
- Next: `eval_pfn_large` and `eval_alpha_large` as sanity sweeps on ulysses (detached, CPU), then compare
  with the original eval_pfn_large artifacts.
