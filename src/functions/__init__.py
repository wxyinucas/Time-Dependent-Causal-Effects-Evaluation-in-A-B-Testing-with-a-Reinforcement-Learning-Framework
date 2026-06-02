# -*- coding: utf-8 -*-
"""Functions subpackage (utility callables).

Modules
-------
spending    : Alpha spending functions for sequential testing.
simulation  : State-transition functions for MDP data generation.
policy      : Decision-making policies for action selection.
"""

from functions.policy import (
    alternating_policy,
    egreedy_policy,
    fix_q0_policy,
    fix_q1_policy,
    odf_egreedy_policy,
    random_policy,
)
from functions.simulation import (
    obf_simulation,
    ttest_simulation,
    two_dimension_simulation,
)
from functions.spending import (
    exp_bound,
    log_bound,
    obf_bound,
    pocock_bound,
    power_bound,
)

__all__ = [
    # spending
    "pocock_bound",
    "power_bound",
    "log_bound",
    "exp_bound",
    "obf_bound",
    # simulation
    "two_dimension_simulation",
    "obf_simulation",
    "ttest_simulation",
    # policy
    "alternating_policy",
    "fix_q1_policy",
    "fix_q0_policy",
    "random_policy",
    "egreedy_policy",
    "odf_egreedy_policy",
]
