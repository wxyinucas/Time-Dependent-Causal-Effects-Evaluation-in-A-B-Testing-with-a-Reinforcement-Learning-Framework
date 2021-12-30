# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2020/10/21  16:21
# File      : _func.py
# Project   : causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Intro
"""
from scipy.stats import norm
import numpy as np


# ============================================ #
#              spending functions
# ============================================ #
def alpha3(total_alpha, t, TK):
    res = total_alpha * np.log(1 + (np.e - 1) * t / TK)
    return res


def alpha1(total_alpha, t, TK):
    res = 2 - 2 * norm.cdf(norm.ppf(1 - total_alpha / 2) * np.sqrt(TK / t))
    return res


def alpha2(total_alpha, t, TK, theta=3):
    res = total_alpha * (t / TK) ** theta
    return res


def alpha4(total_alpha, t, TK, gamma=3):
    res = total_alpha * (1 - np.exp(-gamma * t / TK)) / (1 - np.exp(-gamma))
    return res


alpha1.name, alpha2.name, alpha3.name, alpha4.name = 'alpha1', 'alpha2', 'alpha3', 'alpha4'


def OBF(total_alpha, t, TK):
    res = 2 - 2 * norm.cdf(norm.ppf(1 - total_alpha / 2) * np.sqrt(TK / t))
    return res


OBF.name = 'O\'Brien-Fleming Bound'


def zip_test_time_and_spending_alpha(spending_function, total_alpha, estimate_times, max_observation_num, **kwargs):
    test_time_list = [max_observation_num // 2]
    gap = int(max_observation_num / 2 / (estimate_times - 1))
    for i in range(estimate_times - 2):
        test_time_list.append(test_time_list[-1] + gap)
    test_time_list.append(max_observation_num)
    alpha_list = [spending_function(total_alpha, t, TK=max_observation_num, **kwargs) for t in test_time_list]

    time_alpha_pair_dct = {key: value for key, value in zip(test_time_list, alpha_list)}
    return [0] + test_time_list, time_alpha_pair_dct


def zip_ofb_test_time_and_spending(spending_function, total_alpha, estimate_times, max_observation_num, **kwargs):
    test_time_list = [0]
    gap = int(max_observation_num / estimate_times)
    for i in range(estimate_times - 1):
        test_time_list.append(test_time_list[-1] + gap)
    test_time_list.append(max_observation_num)
    alpha_list = [spending_function(total_alpha, t, TK=max_observation_num) for t in test_time_list[1:]]

    time_alpha_pair_dct = {key: value for key, value in zip(test_time_list[1:], alpha_list)}
    return test_time_list, time_alpha_pair_dct


# ============================================ #
#             state transfer
# ============================================ #
def two_dimension_simulation(last_s, last_a, delta):
    # s1 = (2 * last_a - 1) * last_s[0] / 2 + last_s[1] / 4 + delta * last_a + np.random.normal(0, 0.5)
    # s2 = (2 * last_a - 1) * last_s[1] / 2 + last_s[0] / 4 + delta * last_a + np.random.normal(0, 0.5)
    # s = np.array([s1, s2])
    # y = 1 + (s1 + s2) / 2 + 5 * delta * last_a + np.random.normal(0, 0.3)
    s1 = (2 * last_a - 1) * last_s[0] / 2 + last_s[1] / 4 + 0.1 * last_a + np.random.normal(0, 0.5)
    s2 = (2 * last_a - 1) * last_s[1] / 2 + last_s[0] / 4 + 0.1 * last_a + np.random.normal(0, 0.5)
    s = np.array([s1, s2])
    y = 1 + (s1 + s2) / 2 + delta * last_a + np.random.normal(0, 0.3)
    return y, s


def obf_simulation(last_s, last_a, delta):
    s1 = (2 * last_a - 1) * last_s[0] / 2 + last_s[1] / 4 + delta * last_a + np.random.normal(0, 0.5)
    s2 = (2 * last_a - 1) * last_s[1] / 2 - last_s[0] / 4 + delta * last_a + np.random.normal(0, 0.5)
    s = np.array([s1, s2])
    y = 1 + (s1 + s2) / 2 + np.random.normal(0, 0.3)
    return y, s


def ttest_simulation(last_s, last_a, delta):
    s1 = (2 * last_a - 1) * last_s[0] / 2 + last_s[1] / 4 + delta * last_a + np.random.normal(0, 0.5)
    s2 = (2 * last_a - 1) * last_s[1] / 2 + last_s[0] / 4 + delta * last_a + np.random.normal(0, 0.5)
    s = np.array([s1, s2])
    y = 1 + (s1 + s2) / 2 + delta * last_a + np.random.normal(0, 0.3)
    return y, s


# ============================================ #
#             policy
# ============================================ #
def alternating_policy(**kwargs):
    last_action = kwargs['last_a']
    assert last_action in (0, 1), f'wrong action {last_action}'
    return 1 - last_action


def fix_q1_policy(**kwargs):
    return 1


def fix_q0_policy(**kwargs):
    return 0


def random_policy(**kwargs):
    return np.random.randint(0, 2)


def egreedy_policy(**kwargs):
    last_action, Q_arr, epsilon = kwargs['last_action'], kwargs['Q_arr'], kwargs['epsilon']

    L = len(Q_arr)
    if last_action == 0:
        Q_arr[L // 2:] = [0] * (L // 2)
    else:
        Q_arr[:L // 2] = [0] * (L // 2)

    judge = np.random.uniform(0, 1)
    if judge > epsilon:
        argmax_arr = np.flatnonzero(Q_arr == Q_arr.max())
        return np.random.choice(argmax_arr)
    else:
        return np.random.choice(np.flatnonzero(Q_arr))


def odf_egreedy_policy(**kwargs):
    last_action, epsilon = kwargs['last_action'], kwargs['epsilon']
    if last_action == 0:
        return 0 if np.random.uniform(0, 1, 1) < 3 * epsilon else 1
    else:
        return 0 if np.random.uniform(0, 1, 1) < 5 * epsilon else 1


alternating_policy.name, fix_q1_policy.name, fix_q0_policy.name, random_policy.name, egreedy_policy.name = \
    'Alternating_time_interval design', 'fix_q1_policy', 'fix_q0_policy', 'Markov design', 'Adaptive design'

odf_egreedy_policy.name = 'Adaptive design'

if __name__ == '__main__':
    # for _ in range(100):
    #     print(random_policy())
    # print(alternating_policy(last_action=1))
    zip_ofb_test_time_and_spending(OBF, 0.05, 5, 100)
