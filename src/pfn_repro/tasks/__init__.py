"""Pipeline registry: `task.name` in the config selects the function."""

from collections.abc import Callable

from omegaconf import DictConfig

from ..tracking.sink import Run
from .eval_alpha import eval_alpha
from .eval_pfn import eval_pfn
from .toy_regression import toy_regression
from .train_alpha import train_alpha
from .train_pfn import train_pfn

Pipeline = Callable[[DictConfig, Run], float | None]

TASKS: dict[str, Pipeline] = {
    "toy_regression": toy_regression,  # the template's own tests exercise the machinery with it
    "train_pfn": train_pfn,
    "train_alpha": train_alpha,
    "eval_pfn": eval_pfn,
    "eval_alpha": eval_alpha,
}
