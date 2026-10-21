"""Multi-fidelity (learning-curve) prior, ported from Meta_FTPFN (ifBO's FT-PFN prior, refactored).

- `task.MultiFidelityTask`: one synthetic HPO problem: a BNN maps a configuration to the parameters of a
  mixture of four learning-curve families (`curves.py`), evaluated at any fidelity in (0, 1].
- `allocation.AllocationPrior`: how a sequence's tokens are spread over curves and fidelities.
- `episodes.MFPrior`: both behind the BNN prior's episode interface (what the trainer consumes).
"""

from .episodes import MFPrior
from .task import MultiFidelityTask

__all__ = ["MFPrior", "MultiFidelityTask"]
