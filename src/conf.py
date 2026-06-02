# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2020/10/21  15:22
# File      : conf.py
# Project   : causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Configuration loader: reads experiment config from JSON files and
resolves string references to actual Python callables (functions, classes).

Refactored: config dicts moved to ../configs/*.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from _func import (
    OBF,
    alpha1,
    alpha2,
    alpha3,
    alpha4,
    alternating_policy,
    egreedy_policy,
    fix_q0_policy,
    fix_q1_policy,
    obf_simulation,
    odf_egreedy_policy,
    random_policy,
    ttest_simulation,
    two_dimension_simulation,
)
from _monitor import Monitor, OBFMonitor, TestMonitor, TTestMonitor
from generation.generator import DataGenerator, TestDataGenerator

# -- project paths -----------------------------------------------------------
BASE_PATH = Path(__file__).parents[1]  # project root
CONFIG_DIR = BASE_PATH / "configs"

# -- registry: string name → callable ----------------------------------------
_CALLABLE_REGISTRY: dict[str, Any] = {
    # spending functions
    "alpha1": alpha1,
    "alpha2": alpha2,
    "alpha3": alpha3,
    "alpha4": alpha4,
    "OBF": OBF,
    # state transfer functions
    "two_dimension_simulation": two_dimension_simulation,
    "obf_simulation": obf_simulation,
    "ttest_simulation": ttest_simulation,
    # decision policies
    "alternating_policy": alternating_policy,
    "fix_q1_policy": fix_q1_policy,
    "fix_q0_policy": fix_q0_policy,
    "random_policy": random_policy,
    "egreedy_policy": egreedy_policy,
    "odf_egreedy_policy": odf_egreedy_policy,
    # data generators
    "DataGenerator": DataGenerator,
    "TestDataGenerator": TestDataGenerator,
    # monitors
    "Monitor": Monitor,
    "OBFMonitor": OBFMonitor,
    "TestMonitor": TestMonitor,
    "TTestMonitor": TTestMonitor,
}


def _resolve_callable(name: str) -> Any:
    """Resolve a string name to the registered callable."""
    try:
        return _CALLABLE_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown callable '{name}'. Available: {sorted(_CALLABLE_REGISTRY.keys())}"
        )


def _resolve_list(names: list[str]) -> list[Any]:
    """Resolve a list of string names to callables."""
    return [_resolve_callable(n) for n in names]


def load_config(version: str) -> dict[str, Any]:
    """Load an experiment configuration from a JSON file.

    Parameters
    ----------
    version : str
        Config key, e.g. ``"causal_rl"``, ``"t_test"``, ``"develop"``.

    Returns
    -------
    dict
        Configuration dictionary with resolved callables, suitable for
        direct use by ``main.causal_rl()`` and ``montecarlo.monte_carlo()``.
    """
    config_path = CONFIG_DIR / f"{version}.json"

    with open(config_path, "r") as f:
        raw = json.load(f)

    # Resolve string names → callables.
    # Use legacy singular key names for backward compat with main.py / montecarlo.py.
    resolved: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "spending_functions":
            resolved["spending_function"] = _resolve_list(value)
        elif key == "decision_making_policies":
            resolved["decision_making_policy"] = _resolve_list(value)
        elif key in ("state_transfer_func", "initial_policy", "generator", "monitor"):
            resolved[key] = _resolve_callable(value)
        elif key == "initial_state":
            resolved[key] = np.array(value)
        else:
            resolved[key] = value

    # add derived paths
    resolved["json_path"] = BASE_PATH / f"data/{resolved['version']}.json"

    return resolved


# -- backward-compatible default ---------------------------------------------
try:
    conf_dct = load_config("causal_rl")
except FileNotFoundError:
    conf_dct = {}
