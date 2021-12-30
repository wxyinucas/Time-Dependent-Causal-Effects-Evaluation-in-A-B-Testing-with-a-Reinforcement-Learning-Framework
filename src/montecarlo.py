# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2021/5/16  10:19
# File      : montecarlo.py
# Project   : work_causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Intro
"""
from _logger import logger
from _data_generator import DataGenerator, TestDataGenerator
from _monitor import make_poly_basis, MonteCarloMonitor
from _analyzer import plot_row
from _func import zip_test_time_and_spending_alpha, fix_q0_policy, fix_q1_policy
from tqdm import tqdm
from itertools import product
from conf import conf_dct, BASE_PATH
import numpy as np
import json


def monte_carlo(data_generator):
    saver = 0
    for q, K, gamma in product(q_lst, K_lst, gamma_lst):
        saver += 1

        for spending_function, decision_making_policy, delta in product(spending_function_lst,
                                                                        decision_making_policy_lst,
                                                                        delta_lst):
            test_time_list, time_alpha_pair_dct = zip_test_time_and_spending_alpha(spending_function,
                                                                                   total_alpha,
                                                                                   K,
                                                                                   max_observation_num)

            tra_len = int(np.log(1e-8) / np.log(gamma))
            weights = np.ones(tra_len + 1) * gamma
            weights = np.cumprod(weights)

            # data_generator
            dg0 = data_generator(state_dimension=state_dimension,
                                 max_observation_num=tra_len + 1,
                                 state_transfer_func=state_transfer_func,
                                 decision_making_policy=fix_q0_policy,
                                 initial_policy=fix_q0_policy,
                                 initial_action=initial_action,
                                 initial_state=initial_state)

            dg1 = data_generator(state_dimension=state_dimension,
                                 max_observation_num=tra_len + 1,
                                 state_transfer_func=state_transfer_func,
                                 decision_making_policy=fix_q1_policy,
                                 initial_policy=fix_q1_policy,
                                 initial_action=initial_action,
                                 initial_state=initial_state)

            m = MonteCarloMonitor(weights, delta)

            for _ in range(iterations):

                flag_initial_policy = True
                for t in range(1, tra_len + 1):
                    Q_arr = np.array([0.1, 0.1])

                    dg0.run(delta=delta,
                            last_action=dg0.df_action_reward_state.iloc[-1, 1],
                            flag_initial_policy=flag_initial_policy,
                            epsilon=0.1,
                            Q_arr=Q_arr)

                    dg1.run(delta=delta,
                            last_action=dg1.df_action_reward_state.iloc[-1, 1],
                            flag_initial_policy=flag_initial_policy,
                            epsilon=0.1,
                            Q_arr=Q_arr)
                m.estimate_test(df_action_reward_state_0=dg0.df_action_reward_state,
                                df_action_reward_state_1=dg1.df_action_reward_state)
                dg0.reset()
                dg1.reset()

            if m.estimate_test(df_action_reward_state_0=dg0.df_action_reward_state,
                               df_action_reward_state_1=dg1.df_action_reward_state):
                print(f'gamma:{gamma} delta: {delta}    null')
            else:
                print(f'gamma:{gamma} delta: {delta}    alter')
            m.show_res()


if __name__ == '__main__':
    logger.info(f'The conf dict is {conf_dct["version"]}')

    spending_function_lst = conf_dct['spending_function']
    max_observation_num = conf_dct['max_observation_num']
    total_alpha = conf_dct['total_alpha']
    iterations = conf_dct['iterations']

    state_transfer_func = conf_dct['state_transfer_func']
    decision_making_policy_lst = conf_dct['decision_making_policy']
    initial_policy = conf_dct['initial_policy']
    initial_action = conf_dct['initial_action']
    state_dimension = conf_dct['state_dimension']
    initial_state = conf_dct['initial_state']

    q_lst = conf_dct['q']
    K_lst = conf_dct['K']
    gamma_lst = conf_dct['gamma']
    delta_lst = conf_dct['delta']
    B = conf_dct['B']

    conf_dct['json_path'].touch()

    # run!
    np.random.seed(420)
    monte_carlo(conf_dct['generator'])
