# pfn-repro

<!-- Project-facing README: this is what a reviewer opens from your CV (~60 s).
     The research process lives in docs/process.md (owned by the template, updated by `copier update`). -->

[![CI](https://github.com/timruhkopf/pfn-repro/actions/workflows/ci.yml/badge.svg)](https://github.com/timruhkopf/pfn-repro/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Reproducible PFN surrogates, priors and amortized acquisitions (PFNs4BO-style PFN, TabPFN-v2-style surrogate, alpha-PFN), with pinned checkpoints, for reuse as a dependency.

<p align="center"><img src="reports/figures/main_result.png" width="600" alt="Main result"></p>

**Paper / preprint:** _link_ · **Cite:** [CITATION.cff](CITATION.cff) · **How we work:** [docs/process.md](docs/process.md) · **Results & decisions:** [research_log/](research_log/README.md)

## Status

<!-- Project-level gates. A phase starts only after the previous phase's verdict supports it. -->

| Phase | Question | Verdict | Evidence |
|---|---|---|---|
| 0 | Is there a signal worth building a method for? | _pending_ | [log entry](research_log/entries/) |
| 1 | ... | — | — |

## Claims

<!-- Every claim in the paper/abstract, with the evidence behind it. Filled at stage `final`. -->

| Claim | Snapshot | Notebook / figure |
|---|---|---|
| _Method X reduces Y by Z (95% CI ...) over N tasks_ | `results/<experiment>/<sweep_id>/` | `notebooks/03_...ipynb` |

## Quickstart

Runs end to end on a laptop CPU in a few minutes on the bundled sample data:

```bash
git clone https://github.com/timruhkopf/pfn-repro.git && cd pfn-repro
just setup cpu local     # uv sync --extra cpu, pre-commit hooks, .env from envs/local.env.example
just smoke               # uv run pfn_repro experiment=debug
```

Reproduce a reported result:

```bash
just reproduce results/<experiment>/<sweep_id>   # worktree at the sweep's tag, uv sync --frozen, rerun, compare
```

Requirements: [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just). Always pass a torch extra
(`cpu` or `cu126`); see [docs/process.md § Environment](docs/process.md#environment).

## Repository map

| Path | What |
|---|---|
| `src/pfn_repro/` | the package; one entry point `pfn_repro` (Hydra), configs ship inside it |
| `src/pfn_repro/configs/experiment/` | one file per research question (config only) |
| `research_log/` | lab book: entries with decisions, append-only track record, design decisions (backlog: GitHub issues) |
| `results/` | committed snapshots of finished sweeps; notebooks and figures read only these |
| `notebooks/` | executed analyses, outputs committed |
| `references.md` / `.bib` | annotated literature, keyed to BibTeX |

## Citation

```bibtex
@misc{ruhkopf2026pfn_repro,
  title  = {...},
  author = {...},
  year   = {2026},
  url    = {https://github.com/timruhkopf/pfn-repro}
}
```

## License and acknowledgements

MIT, see [LICENSE](LICENSE). Datasets are subject to their own licenses (`data/README.md`).
Funding, compute resources, and code this project builds on.
