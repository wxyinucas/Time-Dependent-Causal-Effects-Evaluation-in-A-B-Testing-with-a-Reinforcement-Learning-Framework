# -*- coding: utf-8 -*-
r"""Plotting utilities for experiment results.

Produces rejection-probability plots (Figure 2 in the paper):
left column = alternative (δ > 0), right column = null (δ = 0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# -- project root ------------------------------------------------------------
BASE_PATH = Path(__file__).parents[2]  # src/analysis/plot.py → project root


def _plot_alternative(ax, ob_times, data, **kwargs):
    for delta, alter in data.items():
        if delta == "0":
            continue
        sns.scatterplot(ax=ax, x=ob_times, y=alter["rates"])
        sns.lineplot(ax=ax, x=ob_times, y=alter["rates"], label=rf"$\delta = {delta}$")

    plt.xlabel("Observation Times")
    plt.ylabel("Reject Probability")
    plt.yticks(np.arange(0, 1.1, 0.1))


def _plot_null(ax, ob_times, true_alphas, null_rates, **kwargs):
    sns.scatterplot(ax=ax, x=ob_times, y=true_alphas)
    ax.plot(ob_times, true_alphas, linestyle="-.", label=f"{kwargs['alpha_name']}")

    sns.scatterplot(ax=ax, x=ob_times, y=null_rates, color=sns.xkcd_rgb["blue violet"])
    ax.plot(
        ob_times,
        null_rates,
        color=sns.xkcd_rgb["blue violet"],
        label=r"$\delta = 0$",
    )
    ax.set_xlabel("Observation Times")
    ax.legend(loc="best", fontsize="x-small")


def _plot_row(
    row_num: int,
    total_row_number: int,
    data: dict[str, Any],
    time_alpha_pair_dct: dict[int, float],
    **kwargs: object,
):
    """Draw one row of subplots: null (right) + alternative (left)."""
    ax_alter = plt.subplot(total_row_number, 2, row_num * 2 + 1)
    ax_null = plt.subplot(total_row_number, 2, row_num * 2 + 2)

    ob_times = list(time_alpha_pair_dct.keys())
    true_alphas = list(time_alpha_pair_dct.values())

    _plot_alternative(ax_alter, ob_times, data)
    _plot_null(ax_null, ob_times, true_alphas, null_rates=data["0"]["rates"], **kwargs)

    plt.ylabel(kwargs["policy_name"] + "\nReject Probability")


def plot_row(conf_dct: dict[str, Any], total_row_num: int):
    """Generate and save rejection-probability plots.

    Parameters
    ----------
    conf_dct : dict
        Experiment configuration (must contain ``json_path`` and ``version``).
    total_row_num : int
        Number of decision policies (rows in the output figure).
    """
    with open(conf_dct["json_path"], "r") as f:
        data_dct = json.load(f)

    data_path = BASE_PATH / "data"

    row_number = -1
    for _saver, experiment in data_dct.items():
        q = experiment["q"]
        gamma = experiment["gamma"]

        for alpha_name, alpha_result in experiment.items():
            if ("alpha" not in alpha_name) and ("Bound" not in alpha_name):
                continue

            plt.figure(figsize=(8, 10))
            for policy_name, policy_result in alpha_result.items():
                if "design" not in policy_name:
                    continue
                time_alpha_pair_dct = policy_result["time_alpha_pair_dct"]
                result = policy_result["result"]

                row_number = (row_number + 1) % total_row_num
                _plot_row(
                    row_number,
                    total_row_num,
                    result,
                    time_alpha_pair_dct,
                    gamma=gamma,
                    alpha_name=alpha_name,
                    policy_name=policy_name,
                )

                if row_number == total_row_num - 1:
                    fname = (
                        f"{conf_dct['version']}_gamma_{gamma}_{alpha_name}_J_{q}.png"
                    )
                    plt.savefig(str(data_path / fname))
                    plt.show()
