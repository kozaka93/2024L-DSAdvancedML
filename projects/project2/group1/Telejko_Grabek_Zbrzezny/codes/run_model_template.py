import os
import pickle

EXPERIMENT_PATH = "experiment_results"
EXPERIMENT_NAME = "exp_mlpc_rffis_905309_6850"

if __name__ == "__main__":
    with open(os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME), "rb") as file:
        experiment = pickle.load(file)  # follows the Experiment class

    # get the feature selector
    feature_selector = experiment.feature_selector(**experiment.feature_selector_config)
    # get and initialize the classifier
    model = experiment.classifier(**experiment.classifier_config)
    # continue...