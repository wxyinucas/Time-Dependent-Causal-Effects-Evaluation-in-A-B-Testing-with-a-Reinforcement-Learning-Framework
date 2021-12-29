# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2020/10/21  15:42
# File      : main.py
# Project   : causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Intro
main 改，暂时用的

"""
from _logger import logger
from _data_generator import DataGenerator, TestDataGenerator
from _monitor import Monitor, make_poly_basis, TestMonitor, OBFMonitor
from _analyzer import plot_row
from _func import zip_test_time_and_spending_alpha, zip_ofb_test_time_and_spending
from tqdm import tqdm
from itertools import product
from conf import conf_dct, BASE_PATH
import numpy as np
import json


# def obf():
#     saver = 0
#     for q, K, gamma in product(q_lst, K_lst, gamma_lst):
#         saver += 1
#
#         for spending_function, decision_making_policy, delta in product(spending_function_lst,
#                                                                         decision_making_policy_lst,
#                                                                         delta_lst):
#             test_time_list, time_alpha_pair_dct = zip_ofb_test_time_and_spending(spending_function,
#                                                                                  total_alpha,
#                                                                                  K,
#                                                                                  max_observation_num)
#
#             # data_generator
#             dg = DataGenerator(state_dimension=state_dimension,
#                                max_observation_num=max_observation_num,
#                                state_transfer_func=state_transfer_func,
#                                decision_making_policy=decision_making_policy,
#                                initial_policy=initial_policy,
#                                initial_action=initial_action,
#                                initial_state=initial_state)
#
#             m = OBFMonitor(threshold=4.149, K=K)  # from paper[14]
#
#             counter = {i: 0 for i in test_time_list[1:]}  # test_time_list 中位置和k一一对应，故排除第0个}
#             # 相同设定下，跑iterations次
#             for _ in tqdm(range(iterations)):
#
#                 # 相同设定下，跑一次
#                 flag_initial_policy = True
#                 for t in range(1, max_observation_num + 1):
#                     dg.run(delta=conf_dct['delta'][0],
#                            last_action=dg.df_action_reward_state.iloc[-1, 0],
#                            flag_initial_policy=flag_initial_policy,
#                            epsilon=0.1)  # checked last_a
#                     if t in test_time_list and t != 0:
#                         k = test_time_list.index(t)  # 只看正数
#                         if k == 1:
#                             flag_initial_policy = False
#
#                         if m.estimate_test(dg.df_action_reward_state.iloc[test_time_list[k - 1]:test_time_list[k], :],
#                                            test_time_list, time_alpha_pair_dct, k):
#                             counter[t] += 1
#                             break
#                 else:
#                     # counter[t] += 1 # 仅仅用于debug
#                     pass
#
#                 logger.debug(f"{_} times")
#                 dg.reset()
#                 m.reset()
#
#             # show results
#             counter = np.asarray(list(counter.values()))
#             rates = counter.cumsum() / iterations
#             logger.debug(f'counter is {counter}')
#             logger.debug(f'rates is {rates}')
#
#             # save results
#             with open(conf_dct['json_path'], 'r+') as json_file:
#                 result = {'q': q,
#                           'K': K,
#                           'gamma': gamma,
#                           spending_function.name: {
#                               'spending_function': spending_function.name,
#                               decision_making_policy.name: {
#                                   'decision_making_policy': decision_making_policy.name,
#                                   'time_alpha_pair_dct': time_alpha_pair_dct,
#                                   'result': {
#                                       delta: {'delta': delta,
#                                               'counter': counter.tolist(),
#                                               'rates': rates.tolist()
#                                               }
#                                   }
#                               }
#                           }
#                           }
#
#                 try:
#                     context = json.load(json_file)
#                 except json.decoder.JSONDecodeError:
#                     pass
#
#                 if delta == delta_lst[0]:
#                     if decision_making_policy == decision_making_policy_lst[0]:
#                         if spending_function == spending_function_lst[0]:
#                             if saver == 1:  # 用于删除旧文件
#                                 logger.info('create a json file')
#                                 context = {str(saver): result}
#                             else:
#                                 context.update({str(saver): result})
#                         else:
#                             context[str(saver)][spending_function.name] = result[spending_function.name]
#                     else:
#                         context[str(saver)][spending_function.name][decision_making_policy.name] = \
#                             result[spending_function.name][decision_making_policy.name]
#                 else:
#                     context[str(saver)][spending_function.name][decision_making_policy.name]['result'][delta] = \
#                         result[spending_function.name][decision_making_policy.name]['result'][delta]
#                     pass
#
#                 json_file.seek(0)
#                 json_file.truncate()
#
#                 json.dump(context, json_file)


# if __name__ == '__main__':
#     logger.info(f'The conf dict is {conf_dct["version"]}')
#
#     spending_function_lst = conf_dct['spending_function']
#     max_observation_num = conf_dct['max_observation_num']
#     total_alpha = conf_dct['total_alpha']
#     iterations = conf_dct['iterations']
#
#     state_transfer_func = conf_dct['state_transfer_func']
#     decision_making_policy_lst = conf_dct['decision_making_policy']
#     initial_policy = conf_dct['initial_policy']
#     initial_action = conf_dct['initial_action']
#     state_dimension = conf_dct['state_dimension']
#     initial_state = conf_dct['initial_state']
#
#     q_lst = conf_dct['q']
#     K_lst = conf_dct['K']
#     gamma_lst = conf_dct['gamma']
#     delta_lst = conf_dct['delta']
#     B = conf_dct['B']
#
#     conf_dct['json_path'].touch()
#
#     # run!
#     np.random.seed(420)
#     # test()
#     # obf_test()
#     plot_row(conf_dct, total_row_num=len(conf_dct['decision_making_policy']))
