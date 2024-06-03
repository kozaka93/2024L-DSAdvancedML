"""Module for experiment utils."""

import os
import pickle

import numpy as np

from src.train import cv


def perform_experiments(
    X, y, experiment_configs, experiment_path="experiment_results", split_indices=None
):
    scores = {}
    indices = {}
    for exp_config in experiment_configs:
        print(f"Experiment {exp_config.experiment_name} in progress...")
        exp_config.scores, exp_config.indices = cv(
            X=X,
            y=y,
            experiment_config=exp_config,
            k_folds=5,
            split_indices=split_indices,
        )
        pickle_name = (
            exp_config.experiment_name + "_" + str(int(np.mean(exp_config.scores)))
        )
        with open(os.path.join(experiment_path, pickle_name), "wb") as f:
            pickle.dump(exp_config, f)

        scores[exp_config.experiment_name] = int(np.mean(exp_config.scores))
        indices[exp_config.experiment_name] = exp_config.indices

    return scores, indices


def find_best_experiments(k=5, experiment_path="experiment_results"):
    exp_names = []
    scores = []
    for pickle_names in os.listdir(experiment_path):
        exp_name, score = pickle_names.rsplit("_", 1)

        exp_names.append(exp_name)
        scores.append(int(score))

    max_score_indices = np.argsort(scores)[-min(k, len(scores)) :]
    best_experiments = []

    for index in max_score_indices:
        path_to_best = exp_names[index] + "_" + str(scores[index])
        with open(os.path.join(experiment_path, path_to_best), "rb") as f:
            best_experiments.append(pickle.load(f))

    return best_experiments
