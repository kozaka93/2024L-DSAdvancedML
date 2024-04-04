import torch


def create_interactions(x):
    _, num_features = x.shape
    interactions = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            interactions.append(x[:, i] * x[:, j])
    interactions = torch.stack(interactions, dim=1)
    return torch.cat([x, interactions], dim=1)
