# -*- coding: utf-8 -*-
r"""State transition functions for the MDP simulation.

Each function implements a different data-generating process
:math:`P(S_{t+1}, Y_t \mid S_t, A_t)`.

References
----------
Shi, Wang, Luo, Zhu, Ye & Song (2021), JASA.
"""

from __future__ import annotations

import numpy as np


def two_dimension_simulation(
    last_s: np.ndarray, last_a: int, delta: float
) -> tuple[float, np.ndarray]:
    r"""Two-dimensional state transition (main simulation in the paper).

    .. math::
        S_{t+1,1} &= (2A_t - 1)\,S_{t,1}/2 + S_{t,2}/4
                     + 0.1\,A_t + \varepsilon_1 \\
        S_{t+1,2} &= (2A_t - 1)\,S_{t,2}/2 + S_{t,1}/4
                     + 0.1\,A_t + \varepsilon_2 \\
        Y_t        &= 1 + (S_{t,1} + S_{t,2})/2
                     + \delta\,A_t + \varepsilon_3

    where :math:`\varepsilon_i \sim \mathcal{N}(0, 0.5^2)`
    for :math:`i=1,2` and :math:`\varepsilon_3 \sim \mathcal{N}(0, 0.3^2)`.
    """
    s1 = (
        (2 * last_a - 1) * last_s[0] / 2
        + last_s[1] / 4
        + 0.1 * last_a
        + np.random.normal(0, 0.5)
    )
    s2 = (
        (2 * last_a - 1) * last_s[1] / 2
        + last_s[0] / 4
        + 0.1 * last_a
        + np.random.normal(0, 0.5)
    )
    s = np.array([s1, s2])
    y = 1 + (s1 + s2) / 2 + delta * last_a + np.random.normal(0, 0.3)
    return y, s


def obf_simulation(
    last_s: np.ndarray, last_a: int, delta: float
) -> tuple[float, np.ndarray]:
    r"""State transition used for OBF comparison experiments.

    Differs from ``two_dimension_simulation`` in that :math:`Y_t` does not
    contain a direct treatment effect term :math:`\delta A_t`.
    """
    s1 = (
        (2 * last_a - 1) * last_s[0] / 2
        + last_s[1] / 4
        + delta * last_a
        + np.random.normal(0, 0.5)
    )
    s2 = (
        (2 * last_a - 1) * last_s[1] / 2
        - last_s[0] / 4
        + delta * last_a
        + np.random.normal(0, 0.5)
    )
    s = np.array([s1, s2])
    y = 1 + (s1 + s2) / 2 + np.random.normal(0, 0.3)
    return y, s


def ttest_simulation(
    last_s: np.ndarray, last_a: int, delta: float
) -> tuple[float, np.ndarray]:
    """State transition used for two-sample t-test comparison.

    Identical to ``two_dimension_simulation`` but kept as a separate
    function for clarity in experiment configs.
    """
    s1 = (
        (2 * last_a - 1) * last_s[0] / 2
        + last_s[1] / 4
        + delta * last_a
        + np.random.normal(0, 0.5)
    )
    s2 = (
        (2 * last_a - 1) * last_s[1] / 2
        + last_s[0] / 4
        + delta * last_a
        + np.random.normal(0, 0.5)
    )
    s = np.array([s1, s2])
    y = 1 + (s1 + s2) / 2 + delta * last_a + np.random.normal(0, 0.3)
    return y, s
