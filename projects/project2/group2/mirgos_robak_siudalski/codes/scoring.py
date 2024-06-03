import numpy as np

def scoring_function(estimator: object, X: np.ndarray, y: np.ndarray) -> int:

    pred_proba = estimator.predict_proba(X)[:, 1]

    top_20_percent_indices = np.argsort(pred_proba)[-int(0.2 * len(pred_proba)):]
    
    # Selecting the corresponding true labels and predicted labels for top 20%
    top_20_true = y[top_20_percent_indices]
    top_20_preds = (pred_proba[top_20_percent_indices] >= 0.5).astype(int)

    # Calculating the number of correct predictions
    correct_preds = (top_20_true == top_20_preds).sum()
    customer_scaled = (correct_preds / len(top_20_preds)) * 1000
    customer_gain = 10 * customer_scaled
    variable_cost = 200 * X.shape[1]
    score = customer_gain - variable_cost
    return score