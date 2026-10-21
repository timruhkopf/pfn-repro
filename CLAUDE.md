# pfn-repro

Reproducible PFN surrogates, priors and amortized acquisitions (PFNs4BO-style PFN, TabPFN-v2-style surrogate, alpha-PFN), with pinned checkpoints, for reuse as a dependency.

@docs/agent_rules.md

## Current phase

<!-- Which phase/question is active, and what its gate is. Update when a phase is decided. -->
Phase 0: TODO.

## Project rules

<!-- Rules specific to this project, e.g. frozen components, validity ranges of models, data quirks. -->

## Commands

`just` lists all recipes. Common: `just test`, `just lint`, `just smoke`, `just sweep <backend> <stage> experiment=<name> ...`,
`just sweep-status <dir>`, `just snapshot <dir>`, `just decide <dir> <verdict> "<reason>"`.
Process and conventions: `docs/process.md`. Machine notes: `docs/compute.local.md` (untracked).
