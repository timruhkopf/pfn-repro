# Using pfn-repro from another repository

pfn-repro is a library with pinned checkpoints. Research repos (e.g. pfn-acqpolicy) depend on it. This
guide covers how to declare the dependency, how to develop both sides at once, and how changes reach
dependents without silently changing their results.

## The options, weighed

| Option | Declared as | Reproducible? | Dev loop | Use it for |
|---|---|---|---|---|
| **Git tag** (recommended default) | `pfn-repro = { git = "https://github.com/timruhkopf/pfn-repro", tag = "v0.1.0" }` | yes: `uv.lock` pins the commit, so the consumer's commit pins pfn-repro too | slow: change, tag, bump | every tracked run, and always for `confirm` / `final` |
| Git branch | `{ git = "…", branch = "feat/x" }` | per lock: `uv lock` re-resolves the branch head, so the lock (and provenance) moves when you re-lock | medium | trying a pfn-repro feature branch in a consumer branch, before it is tagged |
| Editable local checkout | `{ path = "../pfn-repro", editable = true }` | no: code outside the consumer's commit and lock | instant | scratch work on both repos at once; tracked runs check and record it (below) |
| git submodule / vendored copy | `pfn-repro` inside the consumer tree | yes (the consumer's commit pins it) | instant, but two histories in one tree | not recommended: it duplicates the code and invites divergent edits |
| Package index (PyPI / private) | `pfn-repro==0.1.0` | yes | slowest | not worth it for a handful of research repos |

**Recommendation.**
- Use a git tag by default.
- Switch to an editable checkout only on a consumer branch while you develop pfn-repro itself.
- Switch back to a tag before anything is reported.

The consumer's template enforces the last step. Tracked runs refuse editable dependencies that have
uncommitted changes, and record their commit in `sweep.json` and the notebook header. `confirm` and
`final` refuse editable dependencies altogether (`require_clean_dependencies` in the consumer's
`tracking/provenance.py`).

Two caveats:
- **The editable source isn't frozen.** A sweep's frozen worktree covers only the consumer's own code,
  not an editable dependency's. Pulling pfn-repro while a sweep runs changes what its remaining members
  import.
- **CI.** A path dependency doesn't exist on GitHub Actions, so the consumer's CI needs the git form.

## Setting it up in a consumer

```bash
# in the consumer (a repo generated from research-template)
uv add "pfn-repro @ git+https://github.com/timruhkopf/pfn-repro" --tag v0.1.0
uv sync --extra cpu            # or the GPU extra; pfn-repro uses the same torch extras
git add pyproject.toml uv.lock && git commit -m "Depend on pfn-repro v0.1.0"
```

The consumer's `uv.lock` resolves torch for both projects together. pfn-repro doesn't pin a torch
build itself; it uses the consumer's `cpu` / `cu126` extra.

**Checkpoints are not in the package** (except the smoke PFN and the ECDF). On each machine, put the
files in one directory and point the variable at it in the consumer's `.env`:

```bash
PFN_REPRO_CKPT_ROOT=/absolute/path/to/pfn-repro-checkpoints   # e.g. on ulysses: ~/PycharmProjects/pfn-repro/checkpoints
```

```python
from pfn_repro.checkpoints import load, make_prior
from pfn_repro.acquisitions.standard import arm_scores

pfn, ckpt = load("pfn_large")          # sha256-verified, frozen
prior = make_prior(x_dim=2, batch_size=8, seed=0)   # BNN prior on the checkpoints' ECDF scale
```

Record what you use. The checkpoint name and its sha256 (`REGISTRY[name].sha256`) belong in the
consumer's experiment config or log entry; `uv.lock` already pins the code.

## Developing pfn-repro and a consumer together

1. pfn-repro: issue + branch `feat/<N>-<slug>`, change, test (`just test`), push the branch.
2. Consumer, on its own branch:
   - scratch iteration: `uv add --editable ../pfn-repro`, then scratch runs and `pilot` sweeps;
   - sharing it with a GPU box or CI: switch to `uv add "pfn-repro @ git+…" --branch feat/<N>-<slug>`.
3. pfn-repro: merge the PR, bump the version, tag and release (see below).
4. Consumer: `uv add "pfn-repro @ git+…" --tag vX.Y.Z`, then `uv sync`, commit `pyproject.toml` + `uv.lock`.
   Then run `confirm` / `final`.

Don't develop pfn-repro code inside the consumer (e.g. a copied module "for now"). That is how two
versions of the same PFN diverge.

## Releasing changes, and how dependents learn about them

pfn-repro follows **semantic versioning** on its public surface:
- the import paths in `pfn_repro.{priors,surrogates,acquisitions,objectives,checkpoints}`;
- the registry names;
- the golden outputs in `tests/unit/test_checkpoints.py`.

| Change | Version | Examples |
|---|---|---|
| breaking | **major** (0.x: minor) | a registered checkpoint produces different outputs; a renamed module or argument; a changed prior default |
| additive | minor (0.x: patch) | a new checkpoint name, prior, surrogate or acquisition; a new optional argument |
| fix | patch | docs, a bug fix that doesn't change any golden output |

Rules that keep this honest:
- **Registered names never change meaning.** New weights get a new name (`pfn_large_v2`); the
  registry pins each file by sha256, so a swapped file is refused instead of used.
- **Frozen code stays frozen** (`surrogates/pfn.py`, `priors/bnn.py`, `objectives/bar_distribution.py`,
  `acquisitions/alpha_pfn/conditioned.py`): the checkpoints depend on them, and the golden-output test
  fails if they change.
- Every release gets a `CHANGELOG.md` entry with **Breaking / Added / Fixed** sections, a tag, and a
  GitHub release:

```bash
# in pfn-repro, on main, after the PR is merged
#   bump version in pyproject.toml; add the CHANGELOG entry
git commit -am "Release vX.Y.Z" && git tag -a vX.Y.Z -m "vX.Y.Z" && git push --follow-tags
gh release create vX.Y.Z --notes-file <(sed -n '/^## vX.Y.Z/,/^## v/p' CHANGELOG.md | sed '$d')
```

**On the consumer side:**
- **Get notified:** watch the GitHub repo for releases (Watch → Custom → Releases).
- **See what's available:** `uv tree --outdated | grep pfn-repro` shows a newer tag; nothing updates
  until you change the tag.
- **Updating:** read the CHANGELOG first. For a breaking change, re-run the consumer's sanity sweeps
  (and anything a decision depends on) before mixing results from both versions. The consumer's
  `summarize` already keeps different commits apart.
