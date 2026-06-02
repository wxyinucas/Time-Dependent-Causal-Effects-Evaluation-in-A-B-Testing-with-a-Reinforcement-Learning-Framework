# -*- coding: utf-8 -*-
"""Two-sample t-test sequential monitor (comparison baseline)."""

from __future__ import annotations

import numpy as np
from statsmodels.stats.weightstats import ttest_ind


class TTestMonitor:
    """Repeated two-sample t-test at each interim analysis.

    Parameters
    ----------
    state_dimension : int
        Dimension of the state vector (unused; kept for API compatibility).
    """

    def __init__(self, state_dimension: int, *args: object, **kwargs: object):
        self.state_dimension = state_dimension
        self.y_pos: np.ndarray = np.array([])
        self.y_neg: np.ndarray = np.array([])

    def estimate_test(
        self,
        df_action_reward_state,
        test_time_list: list[int],
        time_alpha_pair_dct: dict[int, float],
        k: int,
    ) -> bool:
        """Perform t-test at stage *k*."""
        a_s = df_action_reward_state.iloc[:, 1].to_numpy()
        y_s = df_action_reward_state.iloc[:, 2].to_numpy()
        t = test_time_list[k]

        mask = a_s > 0
        self.y_pos = np.concatenate([self.y_pos, y_s[mask]])
        self.y_neg = np.concatenate([self.y_neg, y_s[~mask]])

        _stat, p, _df = ttest_ind(self.y_pos, self.y_neg, alternative="larger")
        return bool(p < time_alpha_pair_dct[t])

    def cal_q_arr(self, *args: object, **kwargs: object) -> np.ndarray:
        """Dummy Q-values for t-test (no Q-learning used)."""
        return np.array([1.0, 1.0])

    def reset(self) -> None:
        """Reset for a new replication."""
        self.y_pos = np.array([])
        self.y_neg = np.array([])
