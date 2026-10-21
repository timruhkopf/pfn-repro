# ruff: noqa  (a one-off porting script, kept to show exactly what changed)
"""Regenerate pfn_repro/priors/mf/{curves,dims,allocation,bnn,task}.py from the Meta_FTPFN sources."""

import sys

S, P = sys.argv[1], sys.argv[2]
SRC = "# Ported from Meta_FTPFN@claude-init archive/src/ppfn/prior/multi_fidelity/mf_ftpfn_refactor/{f}"


def put(name, text):
    open(f"{P}/{name}", "w").write(text)


def rep(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b)


put("curves.py", SRC.format(f="bnn_link_fn.py") + ", unchanged.\n" + open(f"{S}/bnn_link_fn.py").read())
put("dims.py", SRC.format(f="other.py") + ", unchanged.\n" + open(f"{S}/other.py").read())
a = open(f"{S}/allocation_prior.py").read()
a = rep(
    a,
    "from ppfn.utils.deprecate import deprecated\n",
    """import functools
import warnings


def deprecated(reason):  # was ppfn/utils/deprecate.py
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(f"{func.__name__} is deprecated: {reason}", category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
""",
)
put("allocation.py", SRC.format(f="allocation_prior.py") + ", unchanged apart from inlining `deprecated`.\n" + a)

m = open(f"{S}/bnn__mlp.py").read()
m = rep(m, "import torch\nimport torch.nn as nn\nimport math\n\nfrom pfns4hpo.encoders import Normalize\n", "")
m = rep(
    m,
    """        # (x - m) / sigma to normalize BNN inputs
        self.normalizer = Normalize(0.5, math.sqrt(1 / 12))

""",
    "",
)
m = rep(
    m,
    """        # FIXME: check if the original mlp also did not use the normalizer!
        raise NotImplementedError('check if the original mlp also did not use the normalizer!')
        self.normalizer(x)
""",
    """        # No input normalization, as in the original (PFNs4HPO hpo_lc_pfn_bopfn_broken.py::MLP.forward):
        # the refactor's `self.normalizer(x)` discarded its result. docs/decisions/0002-mf-prior-port.md
""",
)
b = open(f"{S}/bnn__bnn_prior.py").read()
b = rep(
    b,
    "import threading\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\n\nfrom ppfn.prior.bnn.mlp import MLP\n",
    "",
)
b = rep(
    b,
    '''    output_samples = None  # Global cache for BNN output samples for ECDF fitting
    CACHE_DIR = Path(__file__).parent / "prior_ecdf"''',
    """    output_samples = None  # per instance now (see ensure_ecdf_loaded)
    _ecdfs: dict = {}  # (num_inputs, num_outputs, N_datasets, N_per_dataset) -> sorted samples""",
)
i = b.index("    @classmethod\n    def ensure_ecdf_loaded")
j = b.index("    def __init__(self, num_inputs, num_outputs, nn_cls=MLP):")
b = (
    b[:i]
    + '''    @classmethod
    def ensure_ecdf_loaded(cls, num_inputs, num_outputs=23):
        """Load or generate the latent ECDF for these settings, once per process and per settings.

        Changed from the original, which kept ONE class-level ECDF in the package directory: whichever
        num_inputs came first in a process then served every other num_inputs. Here it is cached per
        settings in <data root>/priors, and generated under its own fixed seed, so it neither depends
        on nor disturbs the caller's random state."""
        key = (num_inputs, num_outputs, cls.N_datasets, cls.N_per_dataset)
        if key in cls._ecdfs:
            return cls._ecdfs[key]
        with cls._lock:
            if key not in cls._ecdfs:
                cache = cache_dir()
                cache.mkdir(parents=True, exist_ok=True)
                file = cache / f"mf_bnn_ecdf_in{num_inputs}_out{num_outputs}_n{cls.N_datasets}x{cls.N_per_dataset}.npy"
                if file.exists():
                    cls._ecdfs[key] = np.load(file)
                else:
                    print(f"Generating the MF prior's latent ECDF for {num_inputs} inputs (cached in {file})...")
                    np_state = np.random.get_state()
                    with torch.random.fork_rng():
                        np.random.seed(12345 + num_inputs)
                        torch.manual_seed(12345 + num_inputs)
                        raw = cls._generate_ecdf_samples(cls.N_datasets, cls.N_per_dataset, num_inputs, num_outputs)
                    np.random.set_state(np_state)
                    cls._ecdfs[key] = np.sort(raw.numpy())
                    np.save(file, cls._ecdfs[key])
        return cls._ecdfs[key]

'''
    + b[j:]
)
b = rep(
    b,
    "        self.ensure_ecdf_loaded(num_inputs, num_outputs)\n",
    "        self.output_samples = self.ensure_ecdf_loaded(num_inputs, num_outputs)  # read by the linker\n",
)
b = rep(
    b,
    """                if i % 100 == 99:
                    print(f"{i + 1}/{N_datasets}")
""",
    "",
)
header = '''# Ported from Meta_FTPFN@claude-init src/ppfn/prior/bnn/{mlp,bnn_prior}.py. Changes: no pfns4hpo dependency
# (its Normalize was constructed but never applied); the ECDF is cached per settings in the data root
# and generated under its own seed. See docs/decisions/0002-mf-prior-port.md.
"""The BNN behind the multi-fidelity prior: maps a hyperparameter configuration to 23 latent outputs,
which `curves.VectorizedParameterLinker` turns into the parameters of a learning-curve mixture.

Not the same BNN as `priors/bnn.py` (the BO function prior): a different family, numpy's and torch's
global RNGs, and an ECDF over all 23 outputs that only serves the parameter linking.
"""

import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ...utils.env import repo_root, var


def cache_dir() -> Path:
    return Path(var("DATA_ROOT") or repo_root() / "data") / "priors"

'''
put("bnn.py", header + m + "\n" + b)
t = open(f"{S}/multifidelity_problem_prior.py").read()
t = rep(
    t,
    "from ppfn.prior.bnn.bnn_prior import BNNPrior\nfrom ppfn.prior.multi_fidelity.mf_ftpfn_refactor.bnn_link_fn import VectorizedParameterLinker\n",
    "from .bnn import BNNPrior\nfrom .curves import VectorizedParameterLinker\n",
)
t = rep(
    t,
    """        if ax is None:
            fig = plt.figure""",
    """        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure""",
)
put("task.py", SRC.format(f="multifidelity_problem_prior.py") + ", unchanged apart from import paths.\n" + t)
