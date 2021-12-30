# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2020/10/21  15:22
# File      : oct_21.py
# Project   : causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Intro
"""
from _func import *
from _base_conf import BASE_PATH
from _data_generator import DataGenerator, TestDataGenerator
from _monitor import Monitor, OBFMonitor, TestMonitor, TTestMonitor

conf_dicts = {
    'develop': {
        'version': 'develop',
        'spending_function': [alpha1, alpha2],
        'max_observation_num': 100,
        'total_alpha': 0.05,
        'iterations': 200,
        # data generator

        'state_transfer_func': two_dimension_simulation,
        # 'decision_making_policy': [alternating_policy, egreedy_policy, random_policy],
        'decision_making_policy': [alternating_policy],
        'initial_policy': alternating_policy,
        'initial_action': 0,
        'state_dimension': 2,  # dimension of states
        'initial_state': np.array([0, 0]),
        'q': [4],  # degree of poly basis
        'K': [5],  # max interim analyses
        'gamma': [0.5],  # [0.5, 0.6, 0.7]
        'delta': [0, 0.10, 0.20],
        'B': 100,

        # machine
        'generator': TestDataGenerator,
        'monitor': TestMonitor,
    },

    'causal_rl': {
        'version': 'causal_rl',
        'spending_function': [alpha1, alpha2],
        'max_observation_num': 600,
        'total_alpha': 0.05,
        'iterations': 500,
        # data generator

        'state_transfer_func': two_dimension_simulation,
        # 'decision_making_policy': [alternating_policy],
        'decision_making_policy': [alternating_policy, egreedy_policy, random_policy],
        'initial_policy': alternating_policy,
        'initial_action': 0,
        'state_dimension': 2,  # dimension of states
        'initial_state': np.array([0, 0]),
        'q': [4],  # degree of poly basis
        'K': [5],  # max interim analyses
        'gamma': [0.6],  # [0.5, 0.6, 0.7]
        # 'delta': [0, 0.05, 0.1, 0.15, 0.20],
        'delta': [0, 0.1, 0.2, 0.3, 0.4],
        'B': 1000,

        # machine
        'generator': DataGenerator,
        'monitor': Monitor,
    },

    't_test': {
        'version': 't_test',
        'spending_function': [alpha1, alpha2],
        'max_observation_num': 600,
        'total_alpha': 0.05,
        'iterations': 500,
        # data generator

        'state_transfer_func': ttest_simulation,
        # 'decision_making_policy': [alternating_policy],
        'decision_making_policy': [alternating_policy, egreedy_policy, random_policy],
        'initial_policy': alternating_policy,
        'initial_action': 0,
        'state_dimension': 2,  # dimension of states
        'initial_state': np.array([0, 0]),
        'q': [4],  # degree of poly basis
        'K': [5],  # max interim analyses
        'gamma': [0.5],  # [0.5, 0.6, 0.7]
        # 'delta': [0, 0.4],
        'delta': [0, 0.1, 0.2, 0.3, 0.4],
        'B': 1000,

        # machine
        'generator': DataGenerator,
        'monitor': TTestMonitor,
    },

    'obf': {
        'version': 'compare',
        'spending_function': [OBF],
        'max_observation_num': 600,
        'total_alpha': 0.05,
        'iterations': 500,
        # data generator

        'state_transfer_func': two_dimension_simulation,
        # 'decision_making_policy': [alternating_policy, egreedy_policy, random_policy],
        'decision_making_policy': [alternating_policy, odf_egreedy_policy, random_policy],
        'initial_policy': alternating_policy,
        'initial_action': 0,
        'state_dimension': 2,  # dimension of states
        'initial_state': np.array([0, 0]),
        'q': [4],  # degree of poly basis
        'K': [5],  # max interim analyses
        'gamma': [0.5],  # [0.5, 0.6, 0.7]
        'delta': [0, 0.05, 0.1, 0.15, 0.20],
        'B': 1000,

        # machine
        'generator': DataGenerator,
        'monitor': OBFMonitor,
    },

    'monte_carlo': {
        'version': 'monte_carlo',
        'spending_function': [alpha1],
        'max_observation_num': 600,
        'total_alpha': 0.05,
        'iterations': 10000,
        # data generator

        'state_transfer_func': ttest_simulation,
        # 'state_transfer_func': two_dimension_simulation,
        'decision_making_policy': [alternating_policy],
        # 'decision_making_policy': [alternating_policy, egreedy_policy, random_policy],
        'initial_policy': alternating_policy,
        'initial_action': 0,
        'state_dimension': 2,  # dimension of states
        'initial_state': np.array([0, 0]),
        'q': [4],  # degree of poly basis
        'K': [5],  # max interim analyses
        'gamma':  [0.1, 0.3, 0.5, 0.7],
        'delta': [0, 0.1, 0.2, 0.3, 0.4],
        'B': 1000,

        # machine
        'generator': DataGenerator,
        'monitor': Monitor,
    },
}

# 选择设定
# conf_dct = conf_dicts['develop']
conf_dct = conf_dicts['causal_rl']
# conf_dct = conf_dicts['obf']
# conf_dct = conf_dicts['t_test']
# conf_dct = conf_dicts['monte_carlo']
conf_dct.update({'json_path': BASE_PATH.joinpath(f'data/{conf_dct["version"]}.json')})
