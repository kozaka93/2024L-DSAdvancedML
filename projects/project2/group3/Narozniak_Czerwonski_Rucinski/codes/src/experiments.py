import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone


def run_grid_search_cv(
        feature_selector_grid_params,
        model_grid_params,
        X,
        y,
        scoring_function,
        cv=5,
        scaler=None,
        verbose=False,
):
    results_dict = {}
    X_df = X.copy()

    X = X.values
    y = y.values
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
                        results_dict[selector_str][model_str] = {"scores": [], "features": []}
                    print(f"Running on: {str(model)} and {str(selector)}")

                    skf = StratifiedKFold(cv)
                    for cv_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
                        # Create (clone for this set of features) a scaler if given
                        if scaler is not None:
                            scaler_clone = clone(scaler).set_output(transform="pandas")

                        model_clone = clone(model)
                        selector_clone = clone(selector)

                        # Scale the features first
                        if scaler is not None:
                            scaler_clone.fit(X[train_index])
                            X_train_scaled = scaler_clone.transform(X[train_index])
                            X_valid_scaled = scaler_clone.transform(X[test_index])
                        else:
                            X_train_scaled = X[train_index]
                            X_valid_scaled = X[test_index]

                        # Decrease the number of features
                        selector_clone.fit(X_train_scaled, y[train_index])
                        X_train_smaller = selector_clone.transform(X_train_scaled)
                        X_valid_smaller = selector_clone.transform(X_valid_scaled)

                        selected_features = X_df.columns[selector_clone.get_support()]

                        n_new_features = X_train_smaller.shape[1]

                        # Train the model
                        model_clone.fit(X_train_smaller, y[train_index])

                        # Predict the new labels
                        y_preds = model_clone.predict(X_valid_smaller)

                        # Calculate the score
                        #score = scoring_function(y[test_index], y_preds, n_new_features)
                        money_score = scoring_function(y[test_index], y_preds, n_new_features, model_clone, X_valid_smaller)
                        #results_dict[selector_str][model_str].append(score)
                        results_dict[selector_str][model_str]["features"].append(selected_features.tolist())
                        results_dict[selector_str][model_str]["scores"].append(money_score)

                    if verbose:
                        print(f"scores: {results_dict[selector_str][model_str]['scores']}")
                        print(f"features: {results_dict[selector_str][model_str]['features']}")

    return results_dict


def apply_transform_to_res(results_dict, numpy_transform=np.mean):
    new_results_dict = {}
    for selector_name, selector_values in results_dict.items():
        new_results_dict[selector_name] = {}
        for model_name, model_values in selector_values.items():
            new_results_dict[selector_name][model_name] = numpy_transform(model_values['scores'])
    return new_results_dict

def profit_scoring(y_true, y_pred, n_features, model, X_valid_smaller):
    y_probs = model.predict_proba(X_valid_smaller)[:, 1]

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    y_probs = y_probs.ravel()

    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'prob': y_probs})

    df = df.sort_values('prob', ascending=False)

    top_20_percent = df.head(int(len(df) * 0.2))
    correct_predictions = ((top_20_percent['pred'] == 1) & (top_20_percent['true'] == 1)).sum()
    profit = correct_predictions * 10 - n_features * 40

    return profit

def create_feature_score_plot(results_dict):
    flat_features = list(itertools.chain.from_iterable(itertools.chain.from_iterable(results_dict[key]['QuadraticDiscriminantAnalysis()']['features'] for key in results_dict)))

    feature_counts = Counter(map(str, flat_features))
    feature_scores = {feature: {"total": 0, "count": 0} for feature in feature_counts.keys()}

    for selector_str in results_dict:
        for model_str in results_dict[selector_str]:
            for features, score in zip(results_dict[selector_str][model_str]["features"], results_dict[selector_str][model_str]["scores"]):
                for feature in features:
                    feature_scores[str(feature)]["total"] += score
                    feature_scores[str(feature)]["count"] += 1

    feature_avg_scores = {feature: scores["total"] / scores["count"] for feature, scores in feature_scores.items()}

    feature_avg_scores_sorted = dict(sorted(feature_avg_scores.items(), key=lambda item: item[1]))

    sorted_counts = [feature_counts[feature] for feature in feature_avg_scores_sorted.keys()]

    counts = np.array(sorted_counts)
    normalized_counts = (counts - counts.min()) / (counts.max() - counts.min())

    cmap = plt.get_cmap('Reds')

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.bar(feature_avg_scores_sorted.keys(), feature_avg_scores_sorted.values(), color=cmap(normalized_counts))
    plt.xlabel('Features')
    plt.ylabel('Average Score')
    plt.title('Average score for each feature')
    plt.xticks(rotation='vertical')
    plt.tight_layout()

    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, label='Feature Count')  # Add a colorbar
    plt.show()