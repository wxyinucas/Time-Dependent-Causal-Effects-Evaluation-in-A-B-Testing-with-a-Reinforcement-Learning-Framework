# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2020/10/21  16:36
# File      : _monitor.py
# Project   : causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Backward-compatibility re-exports from ``monitoring.*`` and ``estimation.*``.

New code should import directly:

    from estimation.basis import make_poly_basis
    from monitoring.bootstrap import BootstrapMonitor
    from monitoring.obf import OBFMonitor
    from monitoring.ttest import TTestMonitor

The legacy names (``Monitor`` → ``BootstrapMonitor``) remain available here
so that ``main.py`` and ``montecarlo.py`` continue to work unchanged.
"""

from __future__ import annotations

import numpy as np

from estimation.basis import make_poly_basis  # noqa: F401
from monitoring.bootstrap import BootstrapMonitor as Monitor  # noqa: F401
from monitoring.obf import OBFMonitor  # noqa: F401
from monitoring.test import TestMonitor  # noqa: F401
from monitoring.ttest import TTestMonitor  # noqa: F401


class MonteCarloMonitor:
    """Monte Carlo evaluation monitor (kept here, not yet migrated)."""

    def __init__(self, weights, delta, **kwargs):
        self.weights = weights
        self.cum = {
            "ys1_sum": 0,
            "ys0_sum": 0,
            "counter": 0,
        }
        self.delta = delta
        self.res = ""

    def estimate_test(
        self, *, df_action_reward_state_0, df_action_reward_state_1, **kwargs
    ):
        counter = self.cum["counter"]
        ys1_sum = self.cum["ys1_sum"]
        ys0_sum = self.cum["ys0_sum"]

        _, _, y_s0 = (
            df_action_reward_state_0.iloc[:, 3:].to_numpy(),
            df_action_reward_state_0.iloc[:, 1].to_numpy(),
            df_action_reward_state_0.iloc[:, 2].to_numpy(),
        )
        _, _, y_s1 = (
            df_action_reward_state_1.iloc[:, 3:].to_numpy(),
            df_action_reward_state_1.iloc[:, 1].to_numpy(),
            df_action_reward_state_1.iloc[:, 2].to_numpy(),
        )

        ys0_sum += y_s0 @ self.weights
        ys1_sum += y_s1 @ self.weights
        counter += len(self.weights)

        mean_1 = ys1_sum / counter
        mean_0 = ys0_sum / counter

        self.cum.update({"ys1_sum": ys1_sum, "ys0_sum": ys0_sum, "counter": counter})
        self.res = (
            f"mean y of q1:{mean_1:.4f}, "
            f"mean y of q0: {mean_0:.4f}, diff:{mean_1 - mean_0:.4f}"
        )
        return self._test(mean_0, mean_1)

    def reset(self):
        self.cum = {"y_s0": np.array([]), "y_s1": np.array([])}

    def _test(self, mean0, mean1, eps=1e-3):
        return bool((mean1 - mean0) < eps)

    def show_res(self):
        print(self.res)


# -- for quick validation ----------------------------------------------------
if __name__ == "__main__":
    basis = make_poly_basis(2, 3)
    print([f([1, 2]) for f in basis])
    print("stop!")
