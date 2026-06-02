# -*- coding: utf-8 -*-
r"""Decision-making policies for action selection in the MDP.

Each policy :math:`\pi(a_t \mid s_t, \ldots)` determines the next action
given the current (and possibly historical) state.

References
----------
Shi, Wang, Luo, Zhu, Ye & Song (2021), JASA.
"""

from __future__ import annotations

import numpy as np


def alternating_policy(**kwargs: object) -> int:
    """Alternating-time-interval design: switch action each time step.

    Returns ``1 - last_action``, guaranteeing balanced exposure.
    """
    last_action: int = kwargs["last_a"]  # type: ignore[assignment]
    if last_action not in (0, 1):
        raise ValueError(f"Action must be 0 or 1, got {last_action}")
    return 1 - last_action


def fix_q1_policy(**kwargs: object) -> int:
    """Always choose action 1 (treated)."""
    return 1


def fix_q0_policy(**kwargs: object) -> int:
    """Always choose action 0 (control)."""
    return 0


def random_policy(**kwargs: object) -> int:
    """Markov design: uniform random action selection."""
    return np.random.randint(0, 2)


def egreedy_policy(**kwargs: object) -> int:
    r"""Adaptive :math:`\varepsilon`-greedy policy.

    With probability :math:`1 - \varepsilon`, selects the action with the
    highest estimated Q-value; with probability :math:`\varepsilon`,
    selects uniformly at random.

    The Q-value array is masked so that only valid transitions from the
    current action are considered.
    """
    last_action: int = kwargs["last_action"]  # type: ignore[assignment]
    Q_arr: np.ndarray = kwargs["Q_arr"]  # type: ignore[assignment]
    epsilon: float = kwargs["epsilon"]  # type: ignore[assignment]

    L = len(Q_arr)
    # mask invalid transitions
    if last_action == 0:
        Q_arr[L // 2 :] = 0.0
    else:
        Q_arr[: L // 2] = 0.0

    if np.random.uniform(0, 1) > epsilon:
        argmax_arr = np.flatnonzero(Q_arr == Q_arr.max())
        return int(np.random.choice(argmax_arr))
    else:
        return int(np.random.choice(np.flatnonzero(Q_arr)))


def odf_egreedy_policy(**kwargs: object) -> int:
    """Adaptive design variant used in OBF comparison experiments.

    Unlike standard e-greedy, uses fixed switching probabilities
    (3ε / 5ε) rather than Q-value maximisation.
    """
    last_action: int = kwargs["last_action"]  # type: ignore[assignment]
    epsilon: float = kwargs["epsilon"]  # type: ignore[assignment]

    if last_action == 0:
        return 0 if np.random.uniform(0, 1) < 3 * epsilon else 1
    else:
        return 0 if np.random.uniform(0, 1) < 5 * epsilon else 1


# -- human-readable names for JSON config reference --------------------------
alternating_policy.name = "Alternating_time_interval design"
fix_q1_policy.name = "fix_q1_policy"
fix_q0_policy.name = "fix_q0_policy"
random_policy.name = "Markov design"
egreedy_policy.name = "Adaptive design"
odf_egreedy_policy.name = "Adaptive design"
