import numpy as np

x_train_original = np.loadtxt('../data/x_train.txt')
y_train_original = np.loadtxt('../data/y_train.txt')

# Split the data into real training set and validation set

from sklearn.model_selection import train_test_split

print("Splitting the data into training and validation set")

np.random.seed(2137) # john paul is praying for our project

x_train, x_val, y_train, y_val = train_test_split(x_train_original, y_train_original, test_size=0.2)

np.save('../data/x_train.npy', x_train)
np.save('../data/y_train.npy', y_train)
np.save('../data/x_val.npy', x_val)
np.save('../data/y_val.npy', y_val)

print("Data split successfully!")

print("Training set size: ", x_train.shape[0])
print("Validation set size: ", x_val.shape[0])
print("Balance of classes in training set: ", np.mean(y_train))
print("Balance of classes in validation set: ", np.mean(y_val))

print("Balance of classes in original dataset: ", np.mean(y_train_original))


