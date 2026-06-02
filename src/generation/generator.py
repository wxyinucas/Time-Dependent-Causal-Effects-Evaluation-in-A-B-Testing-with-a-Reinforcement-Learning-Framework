# -*- coding: utf-8 -*-
r"""MDP trajectory data generator.

Implements the data-generating process described in Section 3:

.. math::

    S_{t+1} &\sim P(\cdot \mid S_t, A_t) \\
    Y_t     &\sim P(\cdot \mid S_t, A_t) \\
    A_t     &\sim \pi(\cdot \mid S_t, \ldots)

The generator maintains a DataFrame of (t, a_t, y_t, s_t) and advances
one time step per ``run()`` call.

References
----------
Shi, Wang, Luo, Zhu, Ye & Song (2021), JASA.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


class DataGenerator:
    r"""MDP trajectory generator.

    Parameters
    ----------
    state_dimension : int
        Dimension :math:`p` of the state vector :math:`S_t`.
    max_observation_num : int
        Maximum number of time steps :math:`T`.
    state_transfer_func : Callable
        Function ``(s, a, delta) -> (y, s_next)``.
    decision_making_policy : Callable
        Policy function ``(**kwargs) -> a``.
    initial_policy : Callable
        Policy used in the first stage (before the first interim analysis).
    initial_action : int
        Action :math:`A_0`.
    initial_state : np.ndarray
        Initial state :math:`S_0`.
    """

    def __init__(
        self,
        *,
        state_dimension: int,
        max_observation_num: int,
        state_transfer_func: Callable[..., tuple[float, np.ndarray]],
        decision_making_policy: Callable[..., int],
        initial_policy: Callable[..., int],
        initial_action: int,
        initial_state: np.ndarray,
        **kwargs: Any,
    ):
        columns = ["t", "a", "y"] + [f"s{i}" for i in range(state_dimension)]
        self.df_action_reward_state = pd.DataFrame(
            np.zeros((max_observation_num, 3 + state_dimension)),
            columns=columns,
        )
        self.df_action_reward_state["t"] = np.arange(1, max_observation_num + 1)
        self.t_pointer: int = 0

        self.state_transfer_func = state_transfer_func
        self.decision_making_policy = decision_making_policy
        self.initial_policy = initial_policy

        self._init_a = initial_action
        self._init_s = initial_state

        if state_dimension != len(initial_state):
            raise ValueError(
                f"state_dimension={state_dimension} but initial_state "
                f"has length {len(initial_state)}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, **kwargs: Any) -> int:
        """Advance one time step and return the current time index."""
        prev_time = self.t_pointer
        curr_time = prev_time + 1
        self.t_pointer = curr_time

        prev_mask = self.df_action_reward_state["t"] == prev_time
        curr_mask = self.df_action_reward_state["t"] == curr_time

        if prev_time == 0:
            s = self._init_s
            y = 0.0
            a = self._init_a
        else:
            last_s = self.df_action_reward_state.loc[prev_mask].iloc[0, 3:]
            flag_init: bool = kwargs.get("flag_initial_policy", False)

            a = (
                self.initial_policy(last_a=kwargs.get("last_action", 0), **kwargs)
                if flag_init
                else self.decision_making_policy(
                    last_a=kwargs.get("last_action", 0), **kwargs
                )
            )

            delta: float = kwargs.get("delta", 0.0)
            y, s = self.state_transfer_func(last_s.to_numpy(), a, delta)

        self.df_action_reward_state.loc[
            curr_mask, self.df_action_reward_state.columns[3:]
        ] = s
        self.df_action_reward_state.loc[curr_mask, "y"] = y
        self.df_action_reward_state.loc[curr_mask, "a"] = a
        return curr_time

    def reset(self) -> None:
        """Reset trajectory for a new Monte Carlo replication."""
        self.df_action_reward_state.iloc[:, 1:] = 0
        self.t_pointer = 0


class TestDataGenerator:
    """Lightweight stub generator for development / debugging.

    Produces random normal rewards; does not model state transitions.
    """

    def __init__(self, **kwargs: Any):
        self.df_action_reward_state = pd.DataFrame(np.zeros((2000, 7)))
        self.t_pointer: int = 0

    def run(self, **kwargs: Any) -> int:
        """Advance one step with random normal reward."""
        prev_time = self.t_pointer
        curr_time = prev_time + 1
        self.t_pointer = curr_time

        self.df_action_reward_state.iloc[curr_time, 1] = curr_time
        self.df_action_reward_state.iloc[curr_time, 2] = np.random.normal(0, 1)
        return curr_time

    def reset(self) -> None:
        """Reset for a new replication."""
        self.df_action_reward_state = pd.DataFrame(np.zeros((2000, 7)))
        self.t_pointer = 0
