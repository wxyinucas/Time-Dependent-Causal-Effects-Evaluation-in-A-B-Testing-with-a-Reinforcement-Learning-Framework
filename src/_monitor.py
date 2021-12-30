# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2020/10/21  16:36
# File      : _monitor.py
# Project   : causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Intro
"""
from _logger import logger
import numpy as np
from numpy import sqrt
from functools import partial
from scipy.linalg import block_diag
from scipy.stats import norm
from statsmodels.stats.weightstats import ttest_ind
import copy


def make_poly_basis(input_dimension: int, poly_degree: int) -> list:
    res = [lambda x: 1]
    for degree in range(1, poly_degree + 1):
        for _input in range(input_dimension):
            string = f'lambda x: x[{_input}]**{degree}'
            logger.debug(string)
            res.append(eval(string))
    return res


def _cal_xi(s, a, no_zeros: bool = False, *, phi_basis):
    """used in estimate"""
    xi = np.array([f(s) for f in phi_basis])

    if no_zeros:
        return xi
    else:
        zeros = np.zeros_like(xi)
        if a == 0:
            res = np.hstack((xi, zeros))
        elif a == 1:
            res = np.hstack((zeros, xi))
        else:
            raise ValueError(f'The value of a is {a}, not in 0, 1.')
        return res[:, np.newaxis]


def _update(z, index, S, Z):
    """used in grouped sequential test"""
    # done: very likely wrong!!!
    mask = Z <= z
    S_next = S[:, mask].copy()
    Z_next = Z[mask].copy()
    index_next = index[mask].copy()
    return index_next, S_next, Z_next


def matrix_sqrt(mat, eps=1e-3):
    values, vectors = np.linalg.eigh(mat)
    for idx, val in enumerate(values):
        if -eps < val < 0:
            values[idx] = 0
        elif val < -eps:
            raise ValueError('Wrong omega_star matrix!')
    # for idx, val in enumerate(values):
    #     if val < 0:
    #         values[idx]=0

    Lambda = np.diag(values)
    Lambda_sqrt = np.sqrt(Lambda)
    res = vectors @ Lambda_sqrt @ vectors.T
    return res


class Monitor:
    def __init__(self, *, state_dimension, phi_basis, gamma, B):
        L = len(phi_basis)
        # ================================================== #
        #
        #                     Estimate
        #
        # ================================================== #
        cum_sigma_0 = np.zeros(2 * np.array([L, L]))
        cum_sigma_1 = np.zeros(2 * np.array([L, L]))
        cum_eta = np.zeros(2 * L)

        cum_omega = np.zeros(4 * np.array([L, L]))
        omega_star = np.zeros(4 * np.array([L, L]))

        length = 5000  # 用于数值模拟G的分布
        x = np.random.normal(0, 1, state_dimension * length).reshape([-1, state_dimension])

        # calculate U
        cum_u = 0
        cal_xi = partial(_cal_xi, phi_basis=phi_basis)
        for x_ in x:
            phi_t = cal_xi(x_, None, no_zeros=True)
            cum_u += phi_t
        u_part = cum_u / length
        u = np.hstack((-1 * u_part, np.zeros(2 * L), u_part))[:, np.newaxis]

        self.cum = {'sigma_0': cum_sigma_0,
                    'sigma_1': cum_sigma_1,
                    'eta': cum_eta,
                    'omega': cum_omega,
                    'omega_star': omega_star,
                    'u': u}
        self.cal_xi = cal_xi
        self.phi_basis = phi_basis  # e-greedy
        self.L = L
        self.gamma = gamma
        self.estimator = {}

        # ================================================== #
        #
        #         Boost Strap Group Sequential Test
        #
        # ================================================== #
        self.B = B

        self.S = np.zeros((4 * L, B))
        self.Z = np.zeros(B)
        self.index = np.ones(B)
        self.test_info = {}

        # ================================================== #
        #
        #         backup for reset
        #
        # ================================================== #
        self.backup = {'cum': copy.deepcopy(self.cum),
                       'estimator': {},
                       'S': self.S.copy(),
                       'Z': self.Z.copy(),
                       'index': self.index.copy(),
                       'test_info': {}
                       }

    def cal_q_arr(self, s, last_a, flag_initial_policy):
        Q_arr = np.array([0, 0.01])

        if flag_initial_policy:
            return Q_arr
        else:
            beta = self.estimator['beta']
            n_basis = self.L
            beta_00 = beta[:n_basis]
            beta_01 = beta[n_basis: 2 * n_basis]
            beta_10 = beta[2 * n_basis: 3 * n_basis]
            beta_11 = beta[3 * n_basis:]

            basis = np.array([f(s) for f in self.phi_basis])

            if last_a == 0:
                Q_arr[0] = basis @ beta_00
                Q_arr[1] = basis @ beta_01
            elif last_a == 1:
                Q_arr[0] = basis @ beta_10
                Q_arr[1] = basis @ beta_11

            return Q_arr

    def reset(self):
        self.cum = copy.deepcopy(self.backup['cum'])
        self.estimator = self.backup['estimator'].copy()
        self.S = self.backup['S'].copy()
        self.Z = self.backup['Z'].copy()
        self.index = self.backup['index'].copy()
        self.test_info = self.backup['test_info'].copy()

    def estimate_test(self, df_action_reward_state, test_time_list, time_alpha_pair_dct, k):
        # load parameters
        cal_xi = self.cal_xi
        L = self.L
        cum = self.cum
        gamma = self.gamma

        cum_sigma_0 = cum['sigma_0']
        cum_sigma_1 = cum['sigma_1']
        cum_eta = cum['eta']

        cum_omega = cum['omega']
        omega_star = cum['omega_star'].copy()
        u = cum['u']

        # process data
        x_s, a_s, y_s = list(map(lambda df: df.to_numpy(),
                                 [df_action_reward_state.iloc[:, 3:],
                                  df_action_reward_state.iloc[:, 1],
                                  df_action_reward_state.iloc[:, 2]]
                                 )
                             )
        t = test_time_list[k]

        # calculate Sigma and Eta
        for idx, x in enumerate(x_s[:-1]):
            xi_t = cal_xi(x_s[idx], a_s[idx])
            cum_sigma_0 += xi_t @ (xi_t - gamma * cal_xi(x_s[idx + 1], 0)).T
            cum_sigma_1 += xi_t @ (xi_t - gamma * cal_xi(x_s[idx + 1], 1)).T
            cum_eta += xi_t.reshape(-1) * y_s[idx]
        sigma_0 = cum_sigma_0 / t
        sigma_1 = cum_sigma_1 / t
        eta = cum_eta / t

        # solve beta
        beta0, _residuals, _rank, _s = np.linalg.lstsq(sigma_0, eta, rcond=None)
        beta1, _residuals, _rank, _s = np.linalg.lstsq(sigma_1, eta, rcond=None)

        beta = np.hstack((beta0, beta1))

        # calculate omega_star
        for idx, x in enumerate(x_s[:-1]):
            xi_t = cal_xi(x_s[idx], a_s[idx])
            phi_t = cal_xi(x_s[idx], None, no_zeros=True)
            phi_t_next = cal_xi(x_s[idx + 1], None, no_zeros=True)

            beta_00 = beta[:L]
            beta_01 = beta[L: 2 * L]
            beta_10 = beta[2 * L: 3 * L]
            beta_11 = beta[3 * L:]

            # calculate zeta (the one added to omega)
            factor1 = y_s[idx] + gamma * phi_t_next @ beta_00 - phi_t @ (beta_00 if a_s[idx] == 0 else beta_01)
            factor2 = y_s[idx] + gamma * phi_t_next @ beta_11 - phi_t @ (beta_10 if a_s[idx] == 0 else beta_11)
            zeta = np.vstack((xi_t * factor1, xi_t * factor2))

            omega_star += zeta @ zeta.T

        # calculate std
        sigma = block_diag(sigma_0, sigma_1)
        sigma_inv, _, _, _ = np.linalg.lstsq(sigma, np.eye(4 * L), rcond=None)

        tmp_omega = (cum_omega + omega_star)
        variance = u.T @ sigma_inv @ tmp_omega @ sigma_inv.T @ u
        variance = variance[0][0]
        information = t / variance

        # calculate omega
        cum_omega += omega_star

        # update information
        cum.update({'sigma_0': cum_sigma_0,
                    'sigma_1': cum_sigma_1,
                    'eta': cum_eta,
                    'omega': cum_omega})

        # zip the results
        self.estimator.update({'beta': beta,
                               'information': information,
                               'u': u,
                               'omega_star': omega_star,
                               't': t,  # old_version test_time_list
                               'sigma_inv': sigma_inv})
        # 用bootstrap计算
        return self._test(time_alpha_pair_dct[test_time_list[k]], k)

    def _test(self, alpha_up_to_now, k):
        # localize
        B = self.B
        Z = self.Z
        S = self.S

        # unzip
        omega_star = self.estimator['omega_star']
        information = self.estimator['information']
        u = self.estimator['u']
        sigma_inv = self.estimator['sigma_inv']
        beta = self.estimator['beta']
        t = self.estimator['t']

        std = 1 / np.sqrt(information)

        # update S_arr and Z_arr
        identity = np.eye(4 * self.L)
        zeros = np.zeros(4 * self.L)
        omega_sqrt = matrix_sqrt(omega_star)

        factor = 1 / (np.sqrt(t) * std) * u.T @ sigma_inv

        for idx, _ in enumerate(self.index):
            S[:, idx] += omega_sqrt @ np.random.multivariate_normal(zeros, identity)
            # S[:, idx] += np.random.multivariate_normal(zeros, omega_star)
            Z[idx] = factor @ S[:, idx]

        I_c = B - len(self.index)

        percentile = (1 - alpha_up_to_now) / (1 - I_c / B)  # 这才是upper percentile todo:check
        # percentile = (alpha_up_to_now - I_c / B) / (1 - I_c / B)
        if percentile > 1:
            percentile = 1
            logger.fatal('to huge!')
        z = np.percentile(Z, percentile * 100)  # cv as z in article

        self.index, self.S, self.Z = _update(z, self.index, S, Z)

        # test
        hat_z = np.sqrt(t) / std * u.T @ beta
        hat_z = hat_z[0]
        if hat_z > z:
            rej = True
        else:
            rej = False

        # update and result
        detail = {'cv': z, 'z': hat_z}
        logger.debug(f'The {k}-th round')
        logger.debug(f'hat_z: {hat_z}; z: {z}.')
        self.test_info[k] = detail

        return rej


class TestMonitor:
    def __init__(self, **kwargs):
        pass

    def estimate_test(self, *args):
        pointer = np.random.binomial(1, 0.01)
        return True if pointer else False

    def cal_q_arr(self, *args):
        return False

    def reset(self):
        pass


class OBFMonitor:
    def __init__(self, **kwargs):
        self.threshold = kwargs['threshold'] * kwargs['K']
        self.cum = {
            'mu_pos': 0,
            'mu_neg': 0,
            't_pos': 0,
            't_neg': 0,
            'y': np.array([])
        }

    def estimate_test(self, df_action_reward_state, test_time_list, time_alpha_pair_dct, k):
        threshold = self.threshold
        cum = self.cum

        # process data
        x_s, a_s, y_s = list(map(lambda df: df.to_numpy(),
                                 [df_action_reward_state.iloc[:, 3:], df_action_reward_state.iloc[:, 1],
                                  df_action_reward_state.iloc[:, 2]]
                                 ))

        mask = a_s > 0
        y_pos = y_s[mask]
        y_neg = y_s[~mask]
        n_pos = len(y_pos)
        n_neg = len(y_neg)

        y = np.append(cum['y'], y_s)
        tmp = y - y.mean()

        t_pos = cum['t_pos'] + n_pos
        t_neg = cum['t_neg'] + n_neg

        mu_pos = (cum['mu_pos'] * cum['t_pos'] + y_pos.sum()) / t_pos
        mu_neg = (cum['mu_neg'] * cum['t_neg'] + y_neg.sum()) / t_neg

        Dx = 1 / (t_pos + t_neg - 1) * (tmp * tmp).sum()
        D_sub = (1 / t_pos + 1 / t_neg) * Dx
        z = k * (mu_pos - mu_neg) ** 2 / D_sub

        self.cum.update({
            'mu_pos': mu_pos,
            'mu_neg': mu_neg,
            't_pos': t_pos,
            't_neg': t_neg,
            'y': y
        })

        rej = True if z > threshold else False
        return rej

    def reset(self):
        self.cum = {
            'mu_pos': 0,
            'mu_neg': 0,
            't_pos': 0,  # a = 1时，正样本的总数
            't_neg': 0,  # a = 0时，负样本的总数
            'y': np.array([])
        }


class TTestMonitor:

    def __init__(self, state_dimension, *args, **kwargs):
        self.state_dimension = state_dimension
        self.y_pos = np.array([])
        self.y_neg = np.array([])

    def estimate_test(self, df_action_reward_state, test_time_list, time_alpha_pair_dct, k):
        x_s, a_s, y_s = list(map(lambda df: df.to_numpy(),
                                 [df_action_reward_state.iloc[:, 3:],
                                  df_action_reward_state.iloc[:, 1],
                                  df_action_reward_state.iloc[:, 2]]
                                 )
                             )
        t = test_time_list[k]

        mask = a_s > 0

        y_pos = y_s[mask]
        y_neg = y_s[~mask]

        self.y_pos = np.concatenate([self.y_pos, y_pos])
        self.y_neg = np.concatenate([self.y_neg, y_neg])

        sta, p, _ = ttest_ind(self.y_pos, self.y_neg, alternative='larger')

        if p < time_alpha_pair_dct[t]:
            rej = True
        else:
            rej = False

        return rej

    def cal_q_arr(self, *args, **kwargs):
        return np.array([1, 1])

    def reset(self):
        self.y_pos = np.array([])
        self.y_neg = np.array([])


class MonteCarloMonitor:
    def __init__(self, weights, delta, **kwargs):
        self.weights = weights
        self.cum = {
            'ys1_sum': 0,
            'ys0_sum': 0,
            'counter': 0,
        }
        self.delta = delta

        self.res = ''

    def estimate_test(self, *, df_action_reward_state_0, df_action_reward_state_1, **kwargs):
        counter = self.cum['counter']
        ys1_sum = self.cum['ys1_sum']
        ys0_sum = self.cum['ys0_sum']

        # process data
        _, _, y_s0 = list(map(lambda df: df.to_numpy(),
                              [df_action_reward_state_0.iloc[:, 3:], df_action_reward_state_0.iloc[:, 1],
                               df_action_reward_state_0.iloc[:, 2]]
                              ))

        _, _, y_s1 = list(map(lambda df: df.to_numpy(),
                              [df_action_reward_state_1.iloc[:, 3:], df_action_reward_state_1.iloc[:, 1],
                               df_action_reward_state_1.iloc[:, 2]]
                              ))

        ys0_sum += y_s0 @ self.weights
        ys1_sum += y_s1 @ self.weights
        counter += len(self.weights)

        mean_1 = ys1_sum / counter
        mean_0 = ys0_sum / counter

        self.cum.update({
            'ys1_sum': ys1_sum,
            'ys0_sum': ys0_sum,
            'counter': counter
        })

        self.res = f'mean y of q1:{mean_1:.4f}, ' \
                   f'mean y of q0: {mean_0:.4f}, diff:{mean_1 - mean_0:.4f}'

        return self._test(mean_0, mean_1)

    def reset(self):
        self.cum = {
            'y_s0': np.array([]),
            'y_s1': np.array([])
        }

    def _test(self, mean0, mean1, eps=1e-3):
        diff = mean1 - mean0

        return True if diff < eps else False

    def show_res(self):
        print(self.res)


if __name__ == '__main__':
    basis = make_poly_basis(2, 3)
    print([f([1, 2]) for f in basis])

    print('stop!')
