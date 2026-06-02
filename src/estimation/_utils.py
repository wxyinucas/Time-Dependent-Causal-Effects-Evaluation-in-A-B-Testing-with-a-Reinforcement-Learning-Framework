# -*- coding: utf-8 -*-
"""Utility functions for LSTD estimation."""

from __future__ import annotations

from typing import Callable

import numpy as np


def make_xi_func(phi_basis: list[Callable[[np.ndarray], float]]):
    r"""Return a function that constructs the xi(s, a) feature vector.

    .. math::

        \xi_t = \begin{cases}
            (\phi(S_t)^\top, \mathbf{0}^\top)^\top & a_t = 0 \\
            (\mathbf{0}^\top, \phi(S_t)^\top)^\top & a_t = 1
        \end{cases}
    """

    def cal_xi(s: np.ndarray, a: int | None, *, no_zeros: bool = False) -> np.ndarray:
        phi = np.array([f(s) for f in phi_basis])

        if no_zeros:
            return phi

        zeros = np.zeros_like(phi)
        if a == 0:
            res = np.hstack((phi, zeros))
        elif a == 1:
            res = np.hstack((zeros, phi))
        else:
            raise ValueError(f"Action must be 0 or 1, got {a}")
        return res[:, np.newaxis]

    return cal_xi


def matrix_sqrt(mat: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Compute the symmetric matrix square root via eigendecomposition.

    Parameters
    ----------
    mat : np.ndarray
        Square, symmetric positive semi-definite matrix.
    eps : float
        Tolerance for treating small negative eigenvalues as zero.

    Returns
    -------
    np.ndarray
        Matrix :math:`M^{1/2}` such that :math:`M^{1/2} M^{1/2} = M`.

    Raises
    ------
    ValueError
        If an eigenvalue is less than `-eps`.
    """
    values, vectors = np.linalg.eigh(mat)

    for idx, val in enumerate(values):
        if -eps < val < 0:
            values[idx] = 0.0
        elif val < -eps:
            raise ValueError("Matrix is not positive semi-definite.")

    Lambda_sqrt = np.diag(np.sqrt(values))
    return vectors @ Lambda_sqrt @ vectors.T
