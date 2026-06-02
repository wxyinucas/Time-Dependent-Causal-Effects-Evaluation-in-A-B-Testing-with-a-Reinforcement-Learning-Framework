# -*- coding: utf-8 -*-
r"""Alpha spending functions for sequential testing.

A spending function :math:`\alpha(t; T, \alpha^*)` allocates the total
Type I error :math:`\alpha^*` across interim analysis times
:math:`t \in \{t_1, \ldots, t_K\}` where :math:`t_K = T`.

References
----------
Shi, Wang, Luo, Zhu, Ye & Song (2021), JASA.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def pocock_bound(total_alpha: float, t: float, TK: float) -> float:
    r"""Pocock-type spending function (:math:`\alpha_1` in the paper).

    .. math::
        \alpha_1(t; T, \alpha^*) =
        2 - 2\,\Phi\!\left(\Phi^{-1}(1 - \alpha^*/2)\,\sqrt{T / t}\right)
    """
    return 2 - 2 * norm.cdf(norm.ppf(1 - total_alpha / 2) * np.sqrt(TK / t))


def power_bound(total_alpha: float, t: float, TK: float, theta: float = 3) -> float:
    r"""Power-type spending function (:math:`\alpha_2` in the paper).

    .. math::
        \alpha_2(t; T, \alpha^*) = \alpha^*\,(t / T)^\theta
    """
    return total_alpha * (t / TK) ** theta


def log_bound(total_alpha: float, t: float, TK: float) -> float:
    r"""Logarithmic spending function (:math:`\alpha_3`).

    .. math::
        \alpha_3(t; T, \alpha^*) =
        \alpha^*\,\log(1 + (e - 1)\,t / T)
    """
    return total_alpha * np.log(1 + (np.e - 1) * t / TK)


def exp_bound(total_alpha: float, t: float, TK: float, gamma: float = 3) -> float:
    r"""Exponential spending function (:math:`\alpha_4`).

    .. math::
        \alpha_4(t; T, \alpha^*) =
        \alpha^*\,\frac{1 - \exp(-\gamma\,t / T)}{1 - \exp(-\gamma)}
    """
    return total_alpha * (1 - np.exp(-gamma * t / TK)) / (1 - np.exp(-gamma))


def obf_bound(total_alpha: float, t: float, TK: float) -> float:
    r"""O'Brien-Fleming boundary spending function.

    .. math::
        \alpha_{\text{OBF}}(t; T, \alpha^*) =
        2 - 2\,\Phi\!\left(\Phi^{-1}(1 - \alpha^*/2)\,\sqrt{T / t}\right)
    """
    return 2 - 2 * norm.cdf(norm.ppf(1 - total_alpha / 2) * np.sqrt(TK / t))


# -- human-readable names for JSON config reference --------------------------
pocock_bound.name = "alpha1"
power_bound.name = "alpha2"
log_bound.name = "alpha3"
exp_bound.name = "alpha4"
obf_bound.name = "O'Brien-Fleming Bound"
