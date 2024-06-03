import numpy as np
from sklearn.pipeline import Pipeline


def evaluate(model, X: np.array, y: np.array) -> float:
    """
    Evaluate a model based on the reward function. Proportion is used for scaling the cost on the validation set.
    Args:
        model: A trained model / pipeline
        X: A numpy array of features
        y: A numpy array of labels
    """
    y_prob = model.predict_proba(X)[:, 1]
    sorted_indices = np.argsort(y_prob)[::-1]
    proportion = len(y) / 5000
    num = int(proportion * 1000)
    selected_indices = sorted_indices[:num]
    reward = 10 * sum(y[selected_indices])
    if isinstance(model, Pipeline):
        num_features = model.named_steps["feature_selection"].get_support(indices=True)
    else:
        num_features = X.shape[1]
    cost = 200 * num_features
    print("Number of customers who took the offer: ", sum(y[selected_indices]))
    print(
        "Number of variables used: ",
        num_features,
    )

    return 1/proportion * reward - cost
