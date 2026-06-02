# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2021/5/16  10:19
# File      : montecarlo.py
# Project   : work_causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Monte Carlo evaluation experiment (standalone).

Refactored: ``run_monte_carlo()`` accepts a complete config dict.
"""

from __future__ import annotations

import sys
from itertools import product
from typing import Any

import numpy as np
from tqdm import tqdm

from _func import fix_q0_policy, fix_q1_policy, zip_test_time_and_spending_alpha
from _logger import logger
from _monitor import MonteCarloMonitor
from conf import load_config


def run_monte_carlo(conf_dct: dict[str, Any]) -> None:
    """Run Monte Carlo estimation of the long-term treatment effect.

    Generates two trajectories (always action 0 vs. always action 1)
    and compares the discounted cumulative rewards.

    Parameters
    ----------
    conf_dct : dict
        Configuration dict as returned by ``conf.load_config()``.
    """
    _c = conf_dct

    spending_function_lst = _c["spending_function"]
    max_observation_num: int = _c["max_observation_num"]
    total_alpha: float = _c["total_alpha"]
    iterations: int = _c["iterations"]

    state_transfer_func = _c["state_transfer_func"]
    decision_making_policy_lst = _c["decision_making_policy"]
    initial_action: int = _c["initial_action"]
    state_dimension: int = _c["state_dimension"]
    initial_state = _c["initial_state"]

    gamma_lst: list[float] = _c["gamma"]
    delta_lst: list[float] = _c["delta"]

    Generator = _c["generator"]

    saver = 0
    for q, K, gamma in product(_c["q"], _c["K"], gamma_lst):
        saver += 1

        for spending_function, decision_making_policy, delta in product(
            spending_function_lst, decision_making_policy_lst, delta_lst
        ):
            test_time_list, time_alpha_pair_dct = zip_test_time_and_spending_alpha(
                spending_function, total_alpha, K, max_observation_num
            )

            tra_len = int(np.log(1e-8) / np.log(gamma))
            weights = np.cumprod(np.ones(tra_len + 1) * gamma)

            dg0 = Generator(
                state_dimension=state_dimension,
                max_observation_num=tra_len + 1,
                state_transfer_func=state_transfer_func,
                decision_making_policy=fix_q0_policy,
                initial_policy=fix_q0_policy,
                initial_action=initial_action,
                initial_state=initial_state,
            )

            dg1 = Generator(
                state_dimension=state_dimension,
                max_observation_num=tra_len + 1,
                state_transfer_func=state_transfer_func,
                decision_making_policy=fix_q1_policy,
                initial_policy=fix_q1_policy,
                initial_action=initial_action,
                initial_state=initial_state,
            )

            m = MonteCarloMonitor(weights, delta)

            for _ in tqdm(range(iterations), desc=f"γ={gamma} δ={delta}"):
                flag_initial_policy = True
                Q_arr = np.array([0.1, 0.1])

                for t in range(1, tra_len + 1):
                    dg0.run(
                        delta=delta,
                        last_action=int(dg0.df_action_reward_state.iloc[-1, 1]),
                        flag_initial_policy=flag_initial_policy,
                        epsilon=0.1,
                        Q_arr=Q_arr,
                    )
                    dg1.run(
                        delta=delta,
                        last_action=int(dg1.df_action_reward_state.iloc[-1, 1]),
                        flag_initial_policy=flag_initial_policy,
                        epsilon=0.1,
                        Q_arr=Q_arr,
                    )

                m.estimate_test(
                    df_action_reward_state_0=dg0.df_action_reward_state,
                    df_action_reward_state_1=dg1.df_action_reward_state,
                )
                dg0.reset()
                dg1.reset()

            label = (
                "null"
                if m.estimate_test(
                    df_action_reward_state_0=dg0.df_action_reward_state,
                    df_action_reward_state_1=dg1.df_action_reward_state,
                )
                else "alter"
            )
            print(f"gamma:{gamma} delta: {delta}    {label}")
            m.show_res()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for Monte Carlo experiment."""
    if argv is None:
        argv = sys.argv

    version = argv[1] if len(argv) > 1 else "monte_carlo"
    config = load_config(version)

    logger.info(f"Monte Carlo experiment: {config['version']}")
    np.random.seed(420)
    run_monte_carlo(config)


if __name__ == "__main__":
    main()
