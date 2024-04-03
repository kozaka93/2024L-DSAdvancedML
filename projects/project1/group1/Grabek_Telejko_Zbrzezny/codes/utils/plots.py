from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_figures_for_cv(
    l_iwls_vals_list: list[float],
    l_sgd_vals_list: list[float],
    l_adam_vals_list: list[float],
) -> None:
    """
    Function plots Minus log likelihood vs Iteration number for each model and split from CV.

    Arguments:
        l_iwls_vals_list: list of minus log-likelihood for every split for IWLS training
        l_sgd_vals_list: list of minus log-likelihood for every split for SGD training
        l_adam_vals_list: list of minus log-likelihood for every split for ADAM training
    """
    n_splits = len(l_iwls_vals_list)
    for i in range(n_splits):

        l_iwls_vals = l_iwls_vals_list[i]
        l_sgd_vals = l_sgd_vals_list[i]
        l_adam_vals = l_adam_vals_list[i]

        plt.figure(figsize=(12, 6))
        plt.plot(
            np.linspace(1, len(l_iwls_vals), len(l_iwls_vals)),
            l_iwls_vals,
            label="IWLS",
        )
        plt.plot(
            np.linspace(1, len(l_sgd_vals), len(l_sgd_vals)), l_sgd_vals, label="SGD"
        )
        plt.plot(
            np.linspace(1, len(l_adam_vals), len(l_adam_vals)),
            l_adam_vals,
            label="ADAM",
        )
        plt.legend()
        plt.rc("xtick", labelsize=10)
        plt.rc("ytick", labelsize=10)
        plt.rc("legend", fontsize=15)
        plt.xlabel("Iterations", fontsize=20)
        plt.ylabel("Minus log-likelihood", fontsize=20)
        plt.title(f"Comparison between models for split {i+1}", fontsize=20)
        plt.show()


def plot_acc_boxplots(
    acc_vals_splits_dict: dict[str, list[float]],
    inter_acc_vals_splits_dict: Optional[dict[str, list[float]]] = None,
) -> None:
    """
    Function plots balanced accuracy scores using boxplots for given results of given models.
    It might compare on the plot results for models trained with interactions and without them
    if interaction scores are given.

    Arguments:
        acc_vals_splits_dict: Dictionary containing balanced accuracy scores for every split for
        every model without interactions
        inter_acc_vals_splits_dict: Dictionary containing balanced accuracy scores for every split
        for every model with interactions

    """
    plt.figure(figsize=(10, 6))
    acc_final_list = []
    model_names_list = []

    for key, value in acc_vals_splits_dict.items():
        acc_final_list += value
        model_names_list += [key for i in range(len(value))]

    if inter_acc_vals_splits_dict is not None:
        inter = ["without"] * len(acc_final_list) + ["with"] * len(acc_final_list)
        interactions = pd.DataFrame({"interactions": inter})
        for key, value in inter_acc_vals_splits_dict.items():
            acc_final_list += value
            model_names_list += [key for i in range(len(value))]

    df = pd.DataFrame(
        {
            "acc": acc_final_list,
            "model": [model_name.upper() for model_name in model_names_list],
        }
    )

    if inter_acc_vals_splits_dict is None:
        sns.boxplot(data=df, x="model", y="acc").set(
            title=f"Models balanced accuracy for {len(acc_vals_splits_dict['iwls'])} train test splits",
            xlabel="Models",
            ylabel="Balanced accuracy",
        )
    else:
        df = pd.concat([df, interactions], axis=1)
        sns.boxplot(data=df, x="model", y="acc", hue="interactions").set(
            title=f"Models balanced accuracy for {len(acc_vals_splits_dict['iwls'])} train test splits",
            xlabel="Models",
            ylabel="Balanced accuracy",
        )
