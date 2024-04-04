import torch
from torch.utils.data import DataLoader, TensorDataset
from model import LogisticRegression
from optimizers.sgd import SGD
from optimizers.adam import ADAM
from sklearn.metrics import balanced_accuracy_score


# Dummy dataset
num_samples, num_features = 1000, 10
y = torch.randint(0, 2, (num_samples,))
x = torch.normal(mean=0, std=3, size=(num_samples, num_features))
x[y == 1] = torch.normal(mean=4, std=1, size=(torch.sum(y == 1), num_features))

y_test = torch.randint(0, 2, (num_samples,))
x_test = torch.normal(mean=0, std=3, size=(num_samples, num_features))
x_test[y_test == 1] = torch.normal(
    mean=4, std=1, size=(torch.sum(y_test == 1), num_features)
)


# Create a DataLoader
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
dataset_test = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

# Initialize the model
model = LogisticRegression(num_features=x.shape[1])
# optimizer2 = SGD(model, lr=0.001)
optimizer2 = ADAM(model, lr=0.001)

for i in range(100):
    losses = []
    for x, y in train_loader:
        y_hat = model.forward(x)
        # print(y_hat, y)
        loss = model.loss(y_hat, y)
        # print(loss)
        optimizer2.backprop(x, y, y_hat)
        # optimizer.backprop(x, y, y_hat)
        losses.append(loss)
        # break

    # print(torch.mean(torch.tensor(losses)))
    # print(model.beta1)

acc = []
for x, y in test_loader:
    y_hat = model.forward(x)
    acc.append(balanced_accuracy_score(y, (y_hat > 0.5).long()))

print("ACC", torch.mean(torch.tensor(acc)))
