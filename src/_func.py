# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2020/10/21  16:21
# File      : _func.py
# Project   : causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Backward-compatibility re-exports from ``functions.*``.

New code should import directly from the subpackage:

    from functions.spending import pocock_bound, power_bound
    from functions.simulation import two_dimension_simulation
    from functions.policy import alternating_policy

The legacy names (``alpha1``, ``OBF``, etc.) remain available here
so that ``main.py`` and ``montecarlo.py`` continue to work unchanged.
"""

from __future__ import annotations

from functions.policy import (
    alternating_policy,  # noqa: F401
    egreedy_policy,  # noqa: F401
    fix_q0_policy,  # noqa: F401
    fix_q1_policy,  # noqa: F401
    odf_egreedy_policy,  # noqa: F401
    random_policy,  # noqa: F401
)
from functions.simulation import (
    obf_simulation,  # noqa: F401
    ttest_simulation,  # noqa: F401
    two_dimension_simulation,  # noqa: F401
)
from functions.spending import exp_bound as alpha4  # noqa: F401
from functions.spending import log_bound as alpha3  # noqa: F401
from functions.spending import obf_bound as OBF  # noqa: F401

# -- re-export with legacy names (noqa: F401) --------------------------------
from functions.spending import pocock_bound as alpha1  # noqa: F401
from functions.spending import power_bound as alpha2  # noqa: F401

# -- utility functions (not moved) -------------------------------------------


def zip_test_time_and_spending_alpha(
    spending_function,
    total_alpha: float,
    estimate_times: int,
    max_observation_num: int,
    **kwargs: float,
) -> tuple[list[int], dict[int, float]]:
    """Build test time grid and corresponding alpha allocations."""
    test_time_list = [max_observation_num // 2]
    gap = int(max_observation_num / 2 / (estimate_times - 1))
    for _ in range(estimate_times - 2):
        test_time_list.append(test_time_list[-1] + gap)
    test_time_list.append(max_observation_num)
    alpha_list = [
        spending_function(total_alpha, t, TK=max_observation_num, **kwargs)
        for t in test_time_list
    ]
    time_alpha_pair_dct = dict(zip(test_time_list, alpha_list))
    return [0] + test_time_list, time_alpha_pair_dct


def zip_ofb_test_time_and_spending(
    spending_function,
    total_alpha: float,
    estimate_times: int,
    max_observation_num: int,
    **kwargs: float,
) -> tuple[list[int], dict[int, float]]:
    """Build test time grid for OBF design (starts at t=0)."""
    test_time_list = [0]
    gap = int(max_observation_num / estimate_times)
    for _ in range(estimate_times - 1):
        test_time_list.append(test_time_list[-1] + gap)
    test_time_list.append(max_observation_num)
    alpha_list = [
        spending_function(total_alpha, t, TK=max_observation_num)
        for t in test_time_list[1:]
    ]
    time_alpha_pair_dct = dict(zip(test_time_list[1:], alpha_list))
    return test_time_list, time_alpha_pair_dct
