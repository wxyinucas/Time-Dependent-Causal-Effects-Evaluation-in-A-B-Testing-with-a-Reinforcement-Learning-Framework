# -*- coding: utf-8 -*-
r"""O'Brien-Fleming sequential monitor.

A comparison baseline that uses the OBF boundary rather than the
bootstrap-calibrated CausalRL procedure.
"""

from __future__ import annotations

import numpy as np


class OBFMonitor:
    r"""O'Brien-Fleming group sequential test monitor.

    Parameters
    ----------
    threshold : float
        Critical value multiplier.
    K : int
        Number of interim analyses.
    """

    def __init__(self, **kwargs: object):
        self.threshold: float = kwargs["threshold"] * kwargs["K"]  # type: ignore[index]
        self.cum: dict[str, object] = {
            "mu_pos": 0.0,
            "mu_neg": 0.0,
            "t_pos": 0,
            "t_neg": 0,
            "y": np.array([]),
        }

    def estimate_test(
        self,
        df_action_reward_state,
        test_time_list: list[int],
        time_alpha_pair_dct: dict[int, float],
        k: int,
    ) -> bool:
        """OBF test at stage *k*."""
        threshold = self.threshold
        cum = self.cum

        a_s = df_action_reward_state.iloc[:, 1].to_numpy()
        y_s = df_action_reward_state.iloc[:, 2].to_numpy()

        mask = a_s > 0
        y_pos = y_s[mask]
        y_neg = y_s[~mask]
        n_pos = len(y_pos)
        n_neg = len(y_neg)

        y_all: np.ndarray = np.append(cum["y"], y_s)  # type: ignore[arg-type]
        tmp = y_all - y_all.mean()

        t_pos: int = cum["t_pos"] + n_pos  # type: ignore[operator]
        t_neg: int = cum["t_neg"] + n_neg  # type: ignore[operator]

        mu_pos = (cum["mu_pos"] * cum["t_pos"] + y_pos.sum()) / t_pos  # type: ignore[operator]
        mu_neg = (cum["mu_neg"] * cum["t_neg"] + y_neg.sum()) / t_neg  # type: ignore[operator]

        Dx = 1.0 / (t_pos + t_neg - 1) * (tmp * tmp).sum()
        D_sub = (1.0 / t_pos + 1.0 / t_neg) * Dx
        z = k * (mu_pos - mu_neg) ** 2 / D_sub

        self.cum.update(
            {
                "mu_pos": mu_pos,
                "mu_neg": mu_neg,
                "t_pos": t_pos,
                "t_neg": t_neg,
                "y": y_all,
            }
        )

        return bool(z > threshold)

    def reset(self) -> None:
        """Reset for a new replication."""
        self.cum = {
            "mu_pos": 0.0,
            "mu_neg": 0.0,
            "t_pos": 0,
            "t_neg": 0,
            "y": np.array([]),
        }
