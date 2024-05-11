import numpy as np
from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone


def run_grid_search_cv(
        feature_selector_grid_params,
        model_grid_params,
        X,
        y,
        scoring_function,
        cv=5
):
    results_dict = {}
    for feature_selector_grid in feature_selector_grid_params:
        feature_selector_class = feature_selector_grid["selector"]
        selector_params_sets = feature_selector_grid["params"]
        for selector_params_set in selector_params_sets:
            selector = feature_selector_class(**selector_params_set)
            selector_str = str(selector)
            if selector_str not in results_dict:
                results_dict[selector_str] = {}
            for model_grid in model_grid_params:
                model_class = model_grid["model"]
                model_params_sets = model_grid["params"]
                for model_params_set in model_params_sets:
                    model = model_class(**model_params_set)
                    model_str = str(model)
                    if model_str not in results_dict:
                        results_dict[selector_str][model_str] = []
                    print(f"Running on: {str(model)} and {str(selector)}")

                    skf = StratifiedKFold(cv)
                    for cv_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
                        model_clone = clone(model)
                        selector_clone = clone(selector)

                        # Decrease the number of features
                        selector_clone.fit(X[train_index], y[train_index])
                        X_train_smaller = selector_clone.transform(X[train_index])
                        X_valid_smaller = selector_clone.transform(X[test_index])

                        n_new_features = X_train_smaller.shape[1]

                        # Train the model
                        model_clone.fit(X_train_smaller, y[train_index])

                        # Predict the new labels
                        y_preds = model_clone.predict(X_valid_smaller)

                        # Calculate the score
                        score = scoring_function(y[test_index], y_preds, n_new_features)
                        results_dict[selector_str][model_str].append(score)
                    print(f"scores: {results_dict[selector_str][model_str]}")

    return results_dict


def apply_transform_to_res(results_dict, numpy_transform=np.mean):
    new_results_dict = {}
    for selector_name, selector_values in results_dict.items():
        new_results_dict[selector_name] = {}
        for model_name, model_values in selector_values.items():
            new_results_dict[selector_name][model_name] = numpy_transform(model_values)
    return new_results_dict
