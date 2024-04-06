import numpy as np


def generate_interactions(X):
    num_samples, num_features = X.shape

    interaction_data = []
    for i in range(num_features - 1):
        for j in range(i + 1, num_features):
            interaction_terms = X[:, i] * X[:, j]
            interaction_data.append(interaction_terms)

    interaction_data = np.column_stack(interaction_data)

    return np.concatenate((X, interaction_data), axis=1)


def test():
    a = 2
    b = 4

    X = np.arange(a * b).reshape((a, b))
    XX = generate_interactions(X)

    print(X)
    print(XX)
