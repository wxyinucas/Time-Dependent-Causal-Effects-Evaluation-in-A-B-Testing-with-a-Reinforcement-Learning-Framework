# -*- coding: utf-8 -*-
"""Regression tests: verify that refactoring does not change numerical results."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conf import load_config
from main import run_experiment

# Expected results for the "develop" config with seed(420).
# These values were captured BEFORE refactoring and must never change.
EXPECTED_DEVELOP = {
    "alpha1": {
        "0": {"counter": [0, 1, 3, 3, 3], "rates": [0.0, 0.005, 0.02, 0.035, 0.05]},
        "0.1": {"counter": [6, 0, 2, 3, 2], "rates": [0.03, 0.03, 0.04, 0.055, 0.065]},
        "0.2": {"counter": [2, 4, 1, 0, 3], "rates": [0.01, 0.03, 0.035, 0.035, 0.05]},
    },
    "alpha2": {
        "0": {"counter": [2, 1, 0, 2, 0], "rates": [0.01, 0.015, 0.015, 0.025, 0.025]},
        "0.1": {
            "counter": [3, 2, 1, 1, 2],
            "rates": [0.015, 0.025, 0.03, 0.035, 0.045],
        },
        "0.2": {
            "counter": [3, 3, 1, 0, 4],
            "rates": [0.015, 0.03, 0.035, 0.035, 0.055],
        },
    },
}


@pytest.fixture
def config():
    """Load develop config and ensure a clean json_path."""
    cfg = load_config("develop")
    cfg["json_path"].touch()
    return cfg


def test_develop_regression(config, tmp_path):
    """develop config output must match the pre-refactoring baseline."""
    # redirect output to a temp file to avoid side-effects
    config["json_path"] = tmp_path / "develop.json"
    config["json_path"].touch()

    np.random.seed(420)
    run_experiment(config)

    with open(config["json_path"]) as f:
        result = json.load(f)

    policies = result["1"]["alpha1"]["Alternating_time_interval design"]["result"]
    for delta_key, expected in EXPECTED_DEVELOP["alpha1"].items():
        actual = policies[delta_key]
        assert actual["counter"] == expected["counter"], (
            f"alpha1 δ={delta_key} counter mismatch"
        )
        assert actual["rates"] == expected["rates"], (
            f"alpha1 δ={delta_key} rates mismatch"
        )

    policies2 = result["1"]["alpha2"]["Alternating_time_interval design"]["result"]
    for delta_key, expected in EXPECTED_DEVELOP["alpha2"].items():
        actual = policies2[delta_key]
        assert actual["counter"] == expected["counter"], (
            f"alpha2 δ={delta_key} counter mismatch"
        )
        assert actual["rates"] == expected["rates"], (
            f"alpha2 δ={delta_key} rates mismatch"
        )


def test_all_configs_loadable():
    """All five config files must load without error."""
    for name in ["develop", "causal_rl", "t_test", "obf", "monte_carlo"]:
        cfg = load_config(name)
        assert cfg["spending_function"], f"{name}: spending_function missing"
        assert cfg["generator"], f"{name}: generator missing"
        assert cfg["monitor"], f"{name}: monitor missing"
        assert cfg["json_path"], f"{name}: json_path missing"


def test_make_poly_basis_no_eval():
    """make_poly_basis must not use eval()."""
    import inspect

    from estimation.basis import make_poly_basis

    source = inspect.getsource(make_poly_basis)
    assert "eval(" not in source, "make_poly_basis still contains eval()!"
    assert "lambda" in source

    # Verify it produces correct dimension
    basis = make_poly_basis(input_dimension=2, poly_degree=3)
    values = [f(np.array([1.0, 2.0])) for f in basis]
    assert len(values) == 1 + 2 * 3  # 1 constant + d * q
    assert values[0] == 1.0
    assert values[1] == 1.0  # s[0]^1
    assert values[2] == 2.0  # s[1]^1
