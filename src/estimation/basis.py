# -*- coding: utf-8 -*-
"""Polynomial basis functions for value-function approximation."""

from __future__ import annotations

from typing import Callable

import numpy as np


def make_poly_basis(
    input_dimension: int, poly_degree: int
) -> list[Callable[[np.ndarray], float]]:
    r"""Construct polynomial basis functions :math:`\{\phi_j(s)\}`.

    Returns a list of callables:

    .. math::

        \{1,\; s_1, \ldots, s_p,\; s_1^2, \ldots, s_p^2,\;
         \ldots,\; s_1^q, \ldots, s_p^q\}

    where :math:`p =` *input_dimension* and :math:`q =` *poly_degree*.

    Parameters
    ----------
    input_dimension : int
        Dimension :math:`p` of the state vector.
    poly_degree : int
        Maximum polynomial degree :math:`q`.

    Returns
    -------
    list[Callable[[np.ndarray], float]]
        List of basis functions, each mapping a state vector to a scalar.
    """
    basis: list[Callable[[np.ndarray], float]] = [lambda x: 1.0]

    for degree in range(1, poly_degree + 1):
        for dim in range(input_dimension):
            # Capture degree and dim in a closure (replaces the old eval-based approach)
            def make_func(
                d: int = degree, i: int = dim
            ) -> Callable[[np.ndarray], float]:
                return lambda x, _d=d, _i=i: float(x[_i] ** _d)

            basis.append(make_func())

    return basis
