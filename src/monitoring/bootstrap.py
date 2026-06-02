# -*- coding: utf-8 -*-
r"""Bootstrap-based group sequential monitoring (CausalRL Monitor).

Implements Algorithm 1 from:

    Shi, Wang, Luo, Zhu, Ye & Song (2021).
    *Dynamic Causal Effects Evaluation in A/B Testing with
    a Reinforcement Learning Framework.* JASA.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
from scipy.linalg import block_diag

from _logger import logger
from estimation._utils import make_xi_func, matrix_sqrt


def _update(
    z: float, index: np.ndarray, S: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter bootstrap samples that exceed the threshold z."""
    mask = Z <= z
    return index[mask].copy(), S[:, mask].copy(), Z[mask].copy()


class BootstrapMonitor:
    r"""CausalRL group sequential monitor with bootstrap calibration.

    Parameters
    ----------
    state_dimension : int
        Dimension of the state vector.
    phi_basis : list[Callable]
        Polynomial basis functions :math:`\{\phi_j\}`.
    gamma : float
        Discount factor :math:`\gamma`.
    B : int
        Number of bootstrap resamples.
    """

    def __init__(
        self,
        *,
        state_dimension: int,
        phi_basis: list[Callable[[np.ndarray], float]],
        gamma: float,
        B: int,
    ):
        L = len(phi_basis)

        # -- Estimate accumulators --
        cum_sigma_0: np.ndarray = np.zeros(2 * np.array([L, L]))
        cum_sigma_1: np.ndarray = np.zeros(2 * np.array([L, L]))
        cum_eta: np.ndarray = np.zeros(2 * L)
        cum_omega: np.ndarray = np.zeros(4 * np.array([L, L]))
        omega_star: np.ndarray = np.zeros(4 * np.array([L, L]))

        # U vector (expectation over normal draws)
        length = 5000
        x = np.random.normal(0, 1, state_dimension * length).reshape(
            -1, state_dimension
        )
        cal_xi_raw = make_xi_func(phi_basis)
        cum_u = np.zeros(len(phi_basis))
        for x_ in x:
            cum_u += cal_xi_raw(x_, None, no_zeros=True)
        u_part = cum_u / length
        u = np.hstack((-u_part, np.zeros(2 * L), u_part))[:, np.newaxis]

        self.cum: dict[str, np.ndarray] = {
            "sigma_0": cum_sigma_0,
            "sigma_1": cum_sigma_1,
            "eta": cum_eta,
            "omega": cum_omega,
            "omega_star": omega_star,
            "u": u,
        }
        self.cal_xi = cal_xi_raw
        self.phi_basis = phi_basis
        self.L = L
        self.gamma = gamma
        self.estimator: dict[str, object] = {}

        # -- Bootstrap test state --
        self.B = B
        self.S: np.ndarray = np.zeros((4 * L, B))
        self.Z: np.ndarray = np.zeros(B)
        self.index: np.ndarray = np.ones(B)
        self.test_info: dict[int, dict[str, float]] = {}

        # Snapshot for reset()
        self._backup: dict[str, object] = {
            "cum": copy.deepcopy(self.cum),
            "estimator": {},
            "S": self.S.copy(),
            "Z": self.Z.copy(),
            "index": self.index.copy(),
            "test_info": {},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cal_q_arr(
        self, s: np.ndarray, last_a: int, flag_initial_policy: bool
    ) -> np.ndarray:
        """Compute Q-value array for ε-greedy action selection."""
        Q_arr = np.array([0.0, 0.01])

        if flag_initial_policy:
            return Q_arr

        beta: np.ndarray = self.estimator["beta"]  # type: ignore[assignment]
        n_basis = self.L
        beta_00 = beta[:n_basis]
        beta_01 = beta[n_basis : 2 * n_basis]
        beta_10 = beta[2 * n_basis : 3 * n_basis]
        beta_11 = beta[3 * n_basis :]

        basis = np.array([f(s) for f in self.phi_basis])

        if last_a == 0:
            Q_arr[0] = basis @ beta_00
            Q_arr[1] = basis @ beta_01
        elif last_a == 1:
            Q_arr[0] = basis @ beta_10
            Q_arr[1] = basis @ beta_11

        return Q_arr

    def reset(self) -> None:
        """Reset internal state for a new Monte Carlo replication."""
        self.cum = copy.deepcopy(self._backup["cum"])  # type: ignore[arg-type]
        self.estimator = copy.copy(self._backup["estimator"])  # type: ignore[arg-type]
        self.S = self._backup["S"].copy()  # type: ignore[index]
        self.Z = self._backup["Z"].copy()  # type: ignore[index]
        self.index = self._backup["index"].copy()  # type: ignore[index]
        self.test_info = copy.copy(self._backup["test_info"])  # type: ignore[arg-type]

    def estimate_test(
        self,
        df_action_reward_state,
        test_time_list: list[int],
        time_alpha_pair_dct: dict[int, float],
        k: int,
    ) -> bool:
        """Perform LSTD estimation and bootstrap hypothesis test at stage *k*.

        Returns ``True`` if the null hypothesis is rejected.
        """
        cal_xi = self.cal_xi
        L = self.L
        cum = self.cum
        gamma = self.gamma

        cum_sigma_0 = cum["sigma_0"]
        cum_sigma_1 = cum["sigma_1"]
        cum_eta = cum["eta"]
        cum_omega = cum["omega"]
        omega_star = cum["omega_star"].copy()
        u = cum["u"]

        # Extract data for current stage
        x_s = df_action_reward_state.iloc[:, 3:].to_numpy()
        a_s = df_action_reward_state.iloc[:, 1].to_numpy()
        y_s = df_action_reward_state.iloc[:, 2].to_numpy()
        t = test_time_list[k]

        # -- Step 1: update Sigma and Eta --
        for idx in range(len(x_s) - 1):
            xi_t = cal_xi(x_s[idx], int(a_s[idx]))
            cum_sigma_0 += xi_t @ (xi_t - gamma * cal_xi(x_s[idx + 1], 0)).T
            cum_sigma_1 += xi_t @ (xi_t - gamma * cal_xi(x_s[idx + 1], 1)).T
            cum_eta += xi_t.reshape(-1) * y_s[idx]

        sigma_0 = cum_sigma_0 / t
        sigma_1 = cum_sigma_1 / t
        eta = cum_eta / t

        # -- Step 2: solve for β (LSTD) --
        beta0, _, _, _ = np.linalg.lstsq(sigma_0, eta, rcond=None)
        beta1, _, _, _ = np.linalg.lstsq(sigma_1, eta, rcond=None)
        beta = np.hstack((beta0, beta1))

        # -- Step 3: update Ω* (covariance of estimating equation) --
        for idx in range(len(x_s) - 1):
            xi_t = cal_xi(x_s[idx], int(a_s[idx]))
            phi_t = cal_xi(x_s[idx], None, no_zeros=True)
            phi_t_next = cal_xi(x_s[idx + 1], None, no_zeros=True)

            beta_00 = beta[:L]
            beta_01 = beta[L : 2 * L]
            beta_10 = beta[2 * L : 3 * L]
            beta_11 = beta[3 * L :]

            factor1 = (
                y_s[idx]
                + gamma * phi_t_next @ beta_00
                - phi_t @ (beta_00 if a_s[idx] == 0 else beta_01)
            )
            factor2 = (
                y_s[idx]
                + gamma * phi_t_next @ beta_11
                - phi_t @ (beta_10 if a_s[idx] == 0 else beta_11)
            )
            zeta = np.vstack((xi_t * factor1, xi_t * factor2))
            omega_star += zeta @ zeta.T

        # -- Step 4: compute information --
        sigma = block_diag(sigma_0, sigma_1)
        sigma_inv, _, _, _ = np.linalg.lstsq(sigma, np.eye(4 * L), rcond=None)
        tmp_omega = cum_omega + omega_star
        variance = (u.T @ sigma_inv @ tmp_omega @ sigma_inv.T @ u)[0, 0]
        information = t / variance

        cum_omega += omega_star
        cum.update(
            {
                "sigma_0": cum_sigma_0,
                "sigma_1": cum_sigma_1,
                "eta": cum_eta,
                "omega": cum_omega,
            }
        )

        self.estimator.update(
            {
                "beta": beta,
                "information": information,
                "u": u,
                "omega_star": omega_star,
                "t": t,
                "sigma_inv": sigma_inv,
            }
        )

        return self._test(time_alpha_pair_dct[test_time_list[k]], k)

    # ------------------------------------------------------------------
    # Bootstrap test
    # ------------------------------------------------------------------

    def _test(self, alpha_up_to_now: float, k: int) -> bool:
        B = self.B
        Z = self.Z
        S = self.S

        omega_star: np.ndarray = self.estimator["omega_star"]  # type: ignore[assignment]
        information: float = self.estimator["information"]  # type: ignore[assignment]
        u: np.ndarray = self.estimator["u"]  # type: ignore[assignment]
        sigma_inv: np.ndarray = self.estimator["sigma_inv"]  # type: ignore[assignment]
        beta: np.ndarray = self.estimator["beta"]  # type: ignore[assignment]
        t: int = self.estimator["t"]  # type: ignore[assignment]

        std = 1.0 / np.sqrt(information)
        identity = np.eye(4 * self.L)
        zeros = np.zeros(4 * self.L)
        omega_sqrt = matrix_sqrt(omega_star)
        factor = 1.0 / (np.sqrt(t) * std) * u.T @ sigma_inv

        # Update bootstrap samples
        for idx in range(len(self.index)):
            S[:, int(idx)] += omega_sqrt @ np.random.multivariate_normal(
                zeros, identity
            )
            Z[int(idx)] = factor @ S[:, int(idx)]

        I_c = B - len(self.index)
        percentile = (1 - alpha_up_to_now) / (1 - I_c / B)
        if percentile > 1:
            percentile = 1.0
            logger.critical("percentile exceeded 1!")

        z = float(np.percentile(Z, percentile * 100))
        self.index, self.S, self.Z = _update(z, self.index, S, Z)

        hat_z = float(np.sqrt(t) / std * u.T @ beta)
        rej = hat_z > z

        self.test_info[k] = {"cv": z, "z": hat_z}
        logger.debug(f"Stage {k}: hat_z={hat_z:.4f}, z={z:.4f}, rej={rej}")
        return rej
