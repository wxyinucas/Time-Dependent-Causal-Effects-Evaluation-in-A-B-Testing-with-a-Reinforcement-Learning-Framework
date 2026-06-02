# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2020/10/21  15:42
# File      : main.py
# Project   : causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Entry point for CausalRL numerical experiments.

Usage
-----
    uv run python main.py              # run causal_rl config (default)
    uv run python main.py develop      # run develop config (fast debug)
    uv run python main.py t_test       # run t-test comparison
    uv run python main.py obf          # run OBF comparison
    uv run python main.py monte_carlo  # run Monte Carlo experiment

Refactored: ``causal_rl()`` now accepts a complete config dict.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from typing import Any

import numpy as np
from tqdm import tqdm

from _analyzer import plot_row
from _func import zip_test_time_and_spending_alpha
from _logger import logger
from _monitor import make_poly_basis
from conf import load_config


def run_experiment(conf_dct: dict[str, Any]) -> None:
    """Run one full experiment sweep over all parameter combinations.

    Parameters
    ----------
    conf_dct : dict
        Configuration dict as returned by ``conf.load_config()``.
    """
    _unpack = conf_dct  # shorthand

    spending_function_lst = _unpack["spending_function"]
    max_observation_num: int = _unpack["max_observation_num"]
    total_alpha: float = _unpack["total_alpha"]
    iterations: int = _unpack["iterations"]

    state_transfer_func = _unpack["state_transfer_func"]
    decision_making_policy_lst = _unpack["decision_making_policy"]
    initial_policy = _unpack["initial_policy"]
    initial_action: int = _unpack["initial_action"]
    state_dimension: int = _unpack["state_dimension"]
    initial_state = _unpack["initial_state"]

    q_lst: list[int] = _unpack["q"]
    K_lst: list[int] = _unpack["K"]
    gamma_lst: list[float] = _unpack["gamma"]
    delta_lst: list[float] = _unpack["delta"]
    B: int = _unpack["B"]

    Generator = _unpack["generator"]
    Monitor = _unpack["monitor"]
    json_path = _unpack["json_path"]
    json_path.touch()

    saver = 0
    for q, K, gamma in product(q_lst, K_lst, gamma_lst):
        saver += 1

        for spending_function, decision_making_policy, delta in product(
            spending_function_lst, decision_making_policy_lst, delta_lst
        ):
            test_time_list, time_alpha_pair_dct = zip_test_time_and_spending_alpha(
                spending_function, total_alpha, K, max_observation_num
            )

            dg = Generator(
                state_dimension=state_dimension,
                max_observation_num=max_observation_num,
                state_transfer_func=state_transfer_func,
                decision_making_policy=decision_making_policy,
                initial_policy=initial_policy,
                initial_action=initial_action,
                initial_state=initial_state,
            )

            phi_basis = make_poly_basis(state_dimension, q)
            m = Monitor(
                state_dimension=state_dimension, phi_basis=phi_basis, gamma=gamma, B=B
            )

            counter: dict[int, int] = {t: 0 for t in test_time_list[1:]}
            for _ in tqdm(range(iterations), desc=f"q={q} K={K} γ={gamma} δ={delta}"):
                flag_initial_policy = True
                for t in range(1, max_observation_num + 1):
                    Q_arr = m.cal_q_arr(
                        dg.df_action_reward_state.iloc[-1, 2:],
                        int(dg.df_action_reward_state.iloc[-1, 1]),
                        flag_initial_policy,
                    )

                    dg.run(
                        delta=delta,
                        last_action=int(dg.df_action_reward_state.iloc[-1, 1]),
                        flag_initial_policy=flag_initial_policy,
                        epsilon=0.1,
                        Q_arr=Q_arr,
                    )

                    if t in test_time_list and t != 0:
                        k = test_time_list.index(t)
                        if k == 1:
                            flag_initial_policy = False

                        if m.estimate_test(
                            dg.df_action_reward_state.iloc[
                                test_time_list[k - 1] : test_time_list[k], :
                            ],
                            test_time_list,
                            time_alpha_pair_dct,
                            k,
                        ):
                            counter[t] += 1
                            break

                dg.reset()
                m.reset()

            counter_arr = np.asarray(list(counter.values()))
            rates = counter_arr.cumsum() / iterations
            logger.debug(f"counter={counter_arr.tolist()}, rates={rates.tolist()}")

            # -- Persist result --
            result_entry = {
                "q": q,
                "K": K,
                "gamma": gamma,
                spending_function.name: {
                    "spending_function": spending_function.name,
                    decision_making_policy.name: {
                        "decision_making_policy": decision_making_policy.name,
                        "time_alpha_pair_dct": time_alpha_pair_dct,
                        "result": {
                            delta: {
                                "delta": delta,
                                "counter": counter_arr.tolist(),
                                "rates": rates.tolist(),
                            }
                        },
                    },
                },
            }

            _merge_into_json(
                json_path,
                result_entry,
                saver,
                delta,
                delta_lst,
                decision_making_policy,
                decision_making_policy_lst,
                spending_function,
                spending_function_lst,
            )


def _merge_into_json(
    json_path,
    result_entry,
    saver,
    delta,
    delta_lst,
    decision_making_policy,
    decision_making_policy_lst,
    spending_function,
    spending_function_lst,
):
    """Update the JSON result file with a new parameter combination."""
    with open(json_path, "r+") as f:
        try:
            context = json.load(f)
        except json.decoder.JSONDecodeError:
            context = {}

        sf_name = spending_function.name
        dp_name = decision_making_policy.name

        if delta == delta_lst[0]:
            if decision_making_policy == decision_making_policy_lst[0]:
                if spending_function == spending_function_lst[0]:
                    if saver == 1:
                        logger.info("create a json file")
                        context[str(saver)] = result_entry
                    else:
                        context[str(saver)] = result_entry
                else:
                    context[str(saver)][sf_name] = result_entry[sf_name]
            else:
                context[str(saver)][sf_name][dp_name] = result_entry[sf_name][dp_name]
        else:
            context[str(saver)][sf_name][dp_name]["result"][delta] = result_entry[
                sf_name
            ][dp_name]["result"][delta]

        f.seek(0)
        f.truncate()
        json.dump(context, f)


# -- CLI ---------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Usage: ``python main.py [version]``

    If *version* is omitted, defaults to ``"causal_rl"``.
    """
    if argv is None:
        argv = sys.argv

    version = argv[1] if len(argv) > 1 else "causal_rl"
    config = load_config(version)

    logger.info(f"Running experiment: {config['version']}")
    np.random.seed(420)
    run_experiment(config)

    # Generate plots
    total_row_num = len(config["decision_making_policy"])
    plot_row(config, total_row_num)


if __name__ == "__main__":
    main()
