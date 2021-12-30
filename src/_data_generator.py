# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2020/10/21  16:27
# File      : _data_generator.py
# Project   : causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Intro
"""
import numpy as np
import pandas as pd


class DataGenerator:
    def __init__(self, *,
                 state_dimension: int,
                 max_observation_num: int,
                 state_transfer_func,
                 decision_making_policy,
                 initial_policy,
                 initial_action,
                 initial_state, **kwargs):
        # generate result dataframe
        self.df_action_reward_state = pd.DataFrame(np.zeros((max_observation_num, 3 + state_dimension)),
                                                   columns=['t', 'a', 'y'] + [f's{i}' for i in range(state_dimension)])
        self.df_action_reward_state['t'] = np.array(range(1, max_observation_num + 1))
        self.t_pointer = 0  # Notice the initial time of time pointer

        # bind functions
        self.state_transfer_func = state_transfer_func
        self.decision_making_policy = decision_making_policy
        self.initial_policy = initial_policy

        # bind initial values
        self._init_a = initial_action
        self._init_s = initial_state
        assert state_dimension == len(
            initial_state), f'wrong dimension {state_dimension} and initial state {initial_state}'

    def run(self, **kwargs):
        """return a time t."""
        prev_time = self.t_pointer
        curr_time = prev_time + 1
        self.t_pointer = curr_time

        prev_time_row = self.df_action_reward_state['t'] == prev_time
        curr_time_row = self.df_action_reward_state['t'] == curr_time

        if prev_time == 0:  # todo 随机化
            s = self._init_s
            y = 0
            a = self._init_a
        else:
            last_s = self.df_action_reward_state[prev_time_row].iloc[0, 3:]
            last_a = self.df_action_reward_state[prev_time_row].iloc[0, 1]
            # y, s = _state_transfer_func(last_s, last_a, kwargs['delta']) # wrong ! last state and this action

            a = self.decision_making_policy(last_a=last_a, **kwargs) if not kwargs['flag_initial_policy'] \
                else self.initial_policy(last_a=last_a, **kwargs)
            # y, s = self.state_transfer_func(last_s, last_a, kwargs['delta'])  # t-test
            y, s = self.state_transfer_func(last_s, a, kwargs['delta'])  # proposed

        self.df_action_reward_state.iloc[curr_time_row, 3:] = s
        self.df_action_reward_state.iloc[curr_time_row, 2] = y
        self.df_action_reward_state.iloc[curr_time_row, 1] = a
        self.t_pointer = curr_time
        return curr_time

    def reset(self):
        self.df_action_reward_state.iloc[:, 1:] = 0
        self.t_pointer = 0


class TestDataGenerator:
    def __init__(self, **kwargs):
        self.df_action_reward_state = pd.DataFrame(np.zeros((2000, 7)))
        self.t_pointer = 0

    def run(self, **kwargs):
        prev_time = self.t_pointer
        curr_time = prev_time + 1
        self.t_pointer = curr_time

        self.df_action_reward_state.iloc[curr_time, 1] = curr_time
        self.df_action_reward_state.iloc[curr_time, 2] = np.random.normal(0, 1, 1)
        return curr_time

    def reset(self):
        self.df_action_reward_state = pd.DataFrame(np.zeros((2000, 7)))
        self.t_pointer = 0


if __name__ == '__main__':
    dg = DataGenerator(state_dimension=3, max_observation_num=20)
    for i in range(10):
        dg.run()
    print(dg.df_action_reward_state)
