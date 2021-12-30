# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2020/10/21  16:49
# File      : _analyzer.py
# Project   : causalRL
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

Intro
"""
import json

from _logger import logger
from _base_conf import BASE_PATH
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def _make_title():
    pass


def _plot_alternative(ax, ob_times, data, **kwargs):
    data.pop('0')
    for delta, alter in data.items():
        sns.scatterplot(ax=ax, x=ob_times, y=alter['rates'])
        sns.lineplot(ax=ax, x=ob_times, y=alter['rates'], label=rf'$\delta = {delta}$')

    plt.xlabel('Observation Times')
    plt.ylabel('Reject Probability')
    plt.yticks(np.arange(0, 1.1, 0.1))  # draw a normal pic
    # plt.yticks(np.arange(0, 0.1, 0.01))  # draw an obf pic

    plt.legend(loc='best', fontsize='x-small')


def _plot_null(ax, ob_times, true_alphas, null_rates, **kwargs):
    sns.scatterplot(ax=ax, x=ob_times, y=true_alphas)
    ax.plot(ob_times, true_alphas, linestyle='-.', label=f'{kwargs["alpha_name"]}')

    sns.scatterplot(ax=ax, x=ob_times, y=null_rates, color=sns.xkcd_rgb['blue violet'])
    ax.plot(ob_times, null_rates, color=sns.xkcd_rgb['blue violet'], label=r'$\delta = 0$')
    ax.set_xlabel('Observation Times')
    ax.legend(loc='best', fontsize='x-small')


def _plot_row(row_num, total_row_number: int, data: dict, time_alpha_pair_dct, **kwargs):
    """process title and legend"""
    ax_null = plt.subplot(total_row_number, 2, row_num * 2 + 2)
    ax_alter = plt.subplot(total_row_number, 2, row_num * 2 + 1)

    # process data
    ob_times = list(time_alpha_pair_dct.keys())
    ture_alphas = list(time_alpha_pair_dct.values())

    # draw a pic of null
    _plot_null(ax_null, ob_times, ture_alphas, null_rates=data['0']['rates'], **kwargs)

    # draw a pic of alternative
    _plot_alternative(ax_alter, ob_times, data)

    plt.ylabel(kwargs['policy_name'] + '\nReject Probability')


def plot_row(conf_dct: dict, total_row_num: int):
    with open(conf_dct['json_path'], 'r') as f:
        data_dct = json.load(f)

    data_path = BASE_PATH.joinpath('data')

    row_number = -1
    for saver, experiment in data_dct.items():

        q = experiment['q']
        K = experiment['K']
        gamma = experiment['gamma']
        for alpha_name, alpha_result in experiment.items():
            if ('alpha' not in alpha_name) and ('Bound' not in alpha_name):
                continue

            fig_size = (8, 10)
            # fig_size = (6, 3)
            plt.figure(figsize=fig_size)
            for policy_name, policy_result in alpha_result.items():
                if 'design' not in policy_name:
                    continue
                time_alpha_pair_dct = policy_result['time_alpha_pair_dct']
                result = policy_result['result']

                row_number = (row_number + 1) % total_row_num

                _plot_row(row_number, total_row_num, result, time_alpha_pair_dct,
                          gamma=gamma,
                          alpha_name=alpha_name,
                          policy_name=policy_name)

                if row_number == total_row_num - 1:
                    plt.savefig(f'{data_path.__str__()}/{conf_dct["version"]}_gamma_{gamma}_{alpha_name}_J_{q}.png')
                    plt.show()


def __test_subplot():
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_xlabel('1')

    ax2 = plt.subplot(2, 2, 3)
    ax2.set_xlabel('c')

    ax3 = plt.subplot(2, 2, 2)
    ax3.set_xlabel('fuck')

    plt.show()


if __name__ == '__main__':
    __test_subplot()
    # plot_3_2()
