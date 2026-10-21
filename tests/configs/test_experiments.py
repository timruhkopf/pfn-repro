"""Every experiment config composes and selects a registered pipeline."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module

import pfn_repro.main
from pfn_repro.tasks import TASKS

EXPERIMENTS = sorted(p.stem for p in (Path(pfn_repro.main.__file__).parent / "configs" / "experiment").glob("*.yaml"))


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_composes(name: str) -> None:
    with initialize_config_module(config_module="pfn_repro.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.task.name in TASKS
