"""Pinned checkpoints: a registry of names -> (file, sha256, kind, provenance), and their loaders.

Large checkpoints live outside git, in `PFN_REPRO_CKPT_ROOT` (default `<repo>/checkpoints`); small ones
(the smoke PFN and the training prior's ECDF) ship inside the package (`pfn_repro/assets/`). A file is
only ever loaded after its sha256 matches the registry, so a swapped file can't produce results under
an old name. New weights get a new name; a registered name never changes meaning.

    from pfn_repro.checkpoints import load, load_ecdf
    pfn, ckpt = load("pfn_large")              # frozen, eval mode
    ecdf = load_ecdf()                          # every evaluation prior must reuse it (see below)

The ECDF matters: every checkpoint here was trained on y = ECDF_train(raw BNN output), with the ECDF
pooled over the variable-d training prior (x_dim=6, variable_dim_min=1). Every evaluation prior must
reuse that exact ECDF (`ecdf_sorted=`), otherwise y lives on a different scale than the models expect.
"""

import functools
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .surrogates.pfn import PFN
from .utils.env import repo_root

ASSETS = Path(__file__).parent / "assets"
SOURCE_REPO = "pfn-acqpolicy"


@dataclass(frozen=True)
class Checkpoint:
    name: str
    file: str
    sha256: str
    kind: str  # pfn | conditioned (alpha-PFN stage 1) | alpha (alpha-PFN stage 2)
    train_n: tuple[int, int]  # context sizes seen in training: stay inside this range
    description: str
    source: dict[str, Any] = field(default_factory=dict)  # where and how it was trained
    in_package: bool = False


ECDF = Checkpoint(
    name="bnn_ecdf_d6",
    file="bnn_ecdf_d6_86acf732c579ed0d.pt",
    sha256="f63ff54b1065db372568c125c85164543eac2cdba0d3010e7b3d7f10aa8d5a54",
    kind="ecdf",
    train_n=(0, 0),
    description="ECDF of the raw BNN-prior output (x_dim=6, variable_dim_min=1), shape [1, n]",
    source={"repo": "AnytimeAcquisition", "cache_key": "86acf732c579ed0d"},
    in_package=True,
)

REGISTRY: dict[str, Checkpoint] = {
    c.name: c
    for c in (
        Checkpoint(
            name="pfn_smoke",
            file="pfn_variable_xdim_smoke.pt",
            sha256="683915cde582597d2b122ce93a6817df6233daff3cd87dc726404cff5d04ee23",
            kind="pfn",
            train_n=(3, 20),
            description="variable-d smoke PFN: d_model 64, 4 layers, 64 bins, 30k steps; never saw > 20 points",
            source={
                "repo": "AnytimeAcquisition",
                "git_commit": "d48bf610cc8ab58e2dbecda24bd7b0ad226510af",
                "mlflow_run_id": "d4615a19b6db4463b392ad80d75160ab",
                "step": 29999,
            },
            in_package=True,
        ),
        Checkpoint(
            name="pfn_large",
            file="pfn_large.pt",
            sha256="00c656d9ed82a8e4b31061d8575e1a2f972629464ac7d2df1ff675112b9ab171",
            kind="pfn",
            train_n=(1, 100),
            description="long-run PFN: 6.6M params (d_model 256, 8 layers, 1000 bins), 1.5M steps x 64 functions",
            source={
                "repo": SOURCE_REPO,
                "git_commit": "0b5683e8220e4947152544e45cf3cdc508ae3732",
                "mlflow_run_id": "9ce11fd39e0b47df86c0efe694dba306",
                "step": 1499999,
                "experiment": "pfn_large",
                "trained_on": "ulysses, 1x RTX 3090, ~24.5 h",
            },
        ),
        Checkpoint(
            name="alpha_conditioned_large",
            file="es_conditioned_large.pt",
            sha256="f677cc8e6016ac22c3e88ef361530c386a60275002b4c7d784ad51a0fbafed6c",
            kind="conditioned",
            train_n=(1, 100),
            description="alpha-PFN stage 1: q(y | x, D, I), I in {none, x*, f*, both}; fine-tuned from pfn_large",
            source={
                "repo": SOURCE_REPO,
                "git_commit": "4d92fd5d343e4166c2d2286eda73c39d0586536f",
                "mlflow_run_id": "e5da4b8a784a467c9a4629d6c0285fb5",
                "step": 156962,
                "experiment": "es_conditioned_large",
                "base": "pfn_large",
            },
        ),
        Checkpoint(
            name="alpha_pes_large",
            file="es_alpha_pes_large.pt",
            sha256="1428e4ed7dffa7dbb91c01122c4825306396b18522ecc00e5edd92dae80271ab",
            kind="alpha",
            train_n=(1, 100),
            description="alpha-PFN stage 2 (PES = Entropy Search): bar distribution over alpha~; its mean is the "
            "acquisition value (maximize)",
            source={
                "repo": SOURCE_REPO,
                "git_commit": "4d92fd5d343e4166c2d2286eda73c39d0586536f",
                "mlflow_run_id": "3d2b6747dc2b4141ad2e0e899cdc30c8",
                "step": 186298,
                "experiment": "es_alpha_pes_large",
                "base": "pfn_large",
                "conditioned": "alpha_conditioned_large",
                "variant": "pes",
            },
        ),
        Checkpoint(
            name="tabpfn_bnn_pilot",
            file="tabpfn_bnn_pilot.pt",
            sha256="d4141d53007ac0e7a8c83cbacfb8a9308a3662a54d77633b2230e143d26a58b4",
            kind="pfn",
            train_n=(1, 100),
            description="TabPFN-v2-style surrogate (5.0M params: d_model 192, 8 layers, 32 thinking rows, 1000 "
            "bins), overnight pilot: 100,115 steps x 64 functions, cosine schedule fitted to the deadline",
            source={
                "repo": "pfn-repro",
                "git_commit": "6567285e0b98efb6ab7fa7072931043d46009a3b",
                "mlflow_run_id": "315b763c6ca040cbb5e911f454fff3dc",
                "sweep_id": "pilot_20261021-142330",
                "step": 100114,
                "experiment": "tabpfn_bnn",
                "trained_on": "ulysses, 1x RTX 3090, 10.6 h (372 ms/step, bf16)",
                "stage": "pilot",
            },
        ),
    )
}


def checkpoint_dir() -> Path:
    """Where the large checkpoints are: PFN_REPRO_CKPT_ROOT, else <repo>/checkpoints (a source checkout)."""
    return Path(os.environ.get("PFN_REPRO_CKPT_ROOT") or repo_root() / "checkpoints")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@functools.lru_cache(maxsize=32)
def _verified(path: str, size: int, mtime: float, expected: str) -> bool:
    return sha256(path) == expected


def locate(name: str) -> Path:
    """Path of a registered file, after checking it exists and matches its sha256."""
    spec = ECDF if name == ECDF.name else REGISTRY[name]
    path = (ASSETS if spec.in_package else checkpoint_dir()) / spec.file
    if not path.exists():
        raise FileNotFoundError(
            f"{spec.name}: {path} is missing. Large checkpoints are not in git; copy {spec.file} there or set "
            f"PFN_REPRO_CKPT_ROOT (see docs/checkpoints.md)."
        )
    st = path.stat()
    if not _verified(str(path), st.st_size, st.st_mtime, spec.sha256):
        raise RuntimeError(f"sha256 mismatch for {path}: got {sha256(path)}, registry pins {spec.sha256}")
    return path


def available() -> dict[str, bool]:
    """Which registered checkpoints are present on this machine (without hashing them)."""
    return {n: ((ASSETS if c.in_package else checkpoint_dir()) / c.file).exists() for n, c in REGISTRY.items()}


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_pfn_file(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """-> (frozen surrogate, raw checkpoint dict) for an unregistered file, e.g. one you just trained.
    The surrogate class is `ckpt["surrogate"]` (checkpoints trained here), else `PFN` (older ones)."""
    from .surrogates import SURROGATES

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cls = SURROGATES[ckpt.get("surrogate", "pfn_repro.surrogates.pfn.PFN")]
    model = cls(**ckpt["config"]).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    bad = set(missing) - {"bar_dist.borders", "bar_dist.bucket_widths"}
    if bad or unexpected:
        raise RuntimeError(f"checkpoint/state_dict mismatch: missing={sorted(bad)} unexpected={sorted(unexpected)}")
    return _freeze(model), ckpt


def load_conditioned_file(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    from .acquisitions.alpha_pfn.conditioned import OptimumConditionedPFN

    ckpt = torch.load(path, map_location=device, weights_only=False)
    assert ckpt.get("kind") == "conditioned", f"{path} is not a stage-1 (conditioned) checkpoint"
    model = OptimumConditionedPFN(PFN(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    return _freeze(model), ckpt


def load(name: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """A registered checkpoint -> (frozen model in eval mode, raw checkpoint dict).

    kind pfn / alpha -> `PFN` (an alpha checkpoint's bar distribution lives on alpha~'s support);
    kind conditioned -> `OptimumConditionedPFN`."""
    spec = REGISTRY[name]
    path = locate(name)
    if spec.kind == "conditioned":
        return load_conditioned_file(path, device)
    return load_pfn_file(path, device)


def load_ecdf(device: str = "cpu") -> torch.Tensor:
    return torch.load(locate(ECDF.name), map_location=device)  # [1, n_ecdf]


def make_prior(x_dim: int, batch_size: int, seed: int, device: str = "cpu", **prior_kwargs: Any):
    """A fixed-d BNN prior batch on the checkpoints' training-prior family and ECDF."""
    from .priors.bnn import BNNPrior

    return BNNPrior(
        batch_size=batch_size,
        x_dim=x_dim,
        variable_dim_min=None,
        seed=seed,
        ecdf_sorted=load_ecdf(device),
        device=device,
        **prior_kwargs,
    )
