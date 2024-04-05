import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
from interactions import generate_interactions
from adam import ADAM
from iwls import IWLS
from sgd import SGD
from sklearn.datasets import make_classification
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


# Custom Optimizer Experiment Function
def run_custom_optimizer_experiment(
    optimizer_class, X_train, y_train, X_test, y_test, tolerance, max_epochs
):
    log_reg = LogisticRegression()
    optimizer = optimizer_class()
    weight_changes = log_reg.fit(X_train, y_train, optimizer, max_epochs, tolerance)
    predictions = log_reg.predict(X_test)
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    return balanced_acc, weight_changes


# PyTorch Experiment Function
def run_pytorch_experiment(
    X_train, y_train, X_test, y_test, optimizer_name, tolerance, max_epochs
):
    # Convert data to torch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).squeeze()
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)

    # Define a simple logistic regression model
    model = nn.Sequential(nn.Linear(X_train.shape[1], 1), nn.Sigmoid())
    criterion = nn.BCELoss()

    # Choose optimizer
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        raise ValueError("Unsupported optimizer. Choose 'adam' or 'sgd'.")

    # Training loop with loss tracking
    losses = []
    prev_loss = None
    converged_epoch = max_epochs
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_torch).squeeze()
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        losses.append(current_loss)

        # Check for convergence
        if prev_loss is not None and abs(prev_loss - current_loss) < tolerance:
            converged_epoch = epoch
            break
        prev_loss = current_loss

    # Predict and calculate accuracy
    with torch.no_grad():
        predictions = model(X_test_torch).squeeze().numpy()
    predictions = (predictions >= 0.5).astype(int)
    balanced_acc = balanced_accuracy_score(y_test, predictions)

    return balanced_acc, losses


# Simulation parameters
max_epochs = 500
tolerance = 1e-6


# Prepare the dataset
X, y = make_classification(n_samples=1000, n_features=7, n_classes=2)
X = generate_interactions(X)
y = y.reshape(-1, 1).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Run experiments
custom_adam_acc, adam_weight_changes = run_custom_optimizer_experiment(
    ADAM, X_train_scaled, y_train, X_test_scaled, y_test, tolerance, max_epochs
)
custom_iwls_acc, iwls_weight_changes = run_custom_optimizer_experiment(
    IWLS, X_train_scaled, y_train, X_test_scaled, y_test, tolerance, max_epochs
)
custom_sgd_acc, sgd_weight_changes = run_custom_optimizer_experiment(
    SGD, X_train_scaled, y_train, X_test_scaled, y_test, tolerance, max_epochs
)

pytorch_adam_acc, pytorch_adam_losses = run_pytorch_experiment(
    X_train_scaled, y_train, X_test_scaled, y_test, "adam", tolerance, max_epochs
)
pytorch_sgd_acc, pytorch_sgd_losses = run_pytorch_experiment(
    X_train_scaled, y_train, X_test_scaled, y_test, "sgd", tolerance, max_epochs
)

# Print balanced accuracies
print(f"Custom ADAM Accuracy: {custom_adam_acc}")
print(f"Custom IWLS Accuracy: {custom_iwls_acc}")
print(f"Custom SGD Accuracy: {custom_sgd_acc}")
print(f"PyTorch ADAM Accuracy: {pytorch_adam_acc}")
print(f"PyTorch SGD Accuracy: {pytorch_sgd_acc}")

# Plot convergence
plt.figure(figsize=(12, 6))

plt.plot(adam_weight_changes, label="Custom ADAM")
plt.plot(iwls_weight_changes, label="Custom IWLS")
plt.plot(sgd_weight_changes, label="Custom SGD")
plt.plot(pytorch_adam_losses, label="PyTorch ADAM")
plt.plot(pytorch_sgd_losses, label="PyTorch SGD")

plt.xlabel("Epoch")
plt.ylabel("Weight Change / Loss")
plt.title("Optimizer Convergence Comparison")
plt.legend()
plt.show()
