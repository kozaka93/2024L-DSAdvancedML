# From https://github.com/sampath9dasari/NaiveBayesClassifier-KDE/blob/master/lib/classifier.py

import numpy as np


class KDENaiveBayesClassifier:
    """
    A Naive Bayes Classifier using Kernel Density Estimation with Parzen windows.

    The classifier implements two kernels for parzen window  - Radial and Hypercube

    It also implements Single bandwidth model and class-specific Multi bandwidth model

    The kernel and model type are passed as arguments to class object initialization.

    Along with the number of bandwidths necessary, in case of Multi bandwidth model.
    """

    def __init__(self, bandwidth=1, kernel='radial', multi_bw=False):
        """
        Initialize the classifier with proper parameters.

        :param bandwidth: An integer giving the number of bandwidths necessary
        :param kernel: A string specifying the kernel to be used for the model
        :param multi_bw: A boolean variable specifying if the Multi bandwidth
                        model is to be used.
                        By default Single bandwidth model is selected.
        """
        self.priors = dict()
        self.dim = 1
        self.multi_bw = multi_bw
        self.bandwidth = bandwidth
        if kernel == "radial":
            self.kernel = self.radial
        elif kernel == "hypercube":
            self.kernel = self.hypercube

    def hypercube(self, k):
        """
        Hypercube kernel for Density Estimation.
        """
        return np.all(k < 0.5, axis=1)

    def radial(self, k):
        """
        Radial Kernel for Density estimation.
        """
        const_part = (2 * np.pi) ** (-self.dim / 2)
        return const_part * np.exp(-0.5 * np.add.reduce(k ** 2, axis=1))

    def parzen_estimation(self, h, x, x_train):
        """
        Density estimation for a single sample against the training set with
        parzen window using the specified bandwidth, kernel.

        :param h: An integer value giving the bandwidth to be used for the class.
        :param x: A single input sample, whose density needs to be estimated.
        :param x_train: Array of input data to calculate KDE value against.
        :return: A single float value giving the density of the function at the given point.
        """
        N = x_train.shape[0]
        dim = self.dim
        k = np.abs(x - x_train) * 1.0 / h
        summation = np.add.reduce(self.kernel(k))
        return summation / (N * (h ** dim))

    def KDE(self, h, x_test, x_train):
        """
        Kernel Density Estimation based on the parameters set.

        :param h: An integer value giving the bandwidth to be used for the class.
        :param x_test: Array of input data to make predictions.
        :param x_train: Array of input data to calculate KDE value against.
        :return: A list of floats giving the density estimation values for each
                 row in x_test, x_test[i] calculated against the training set, previously set
        """
        P_x = np.zeros(len(x_test))
        N = x_train.shape[0]
        dim = self.dim
        for i in range(len(x_test)):
            P_x[i] = self.parzen_estimation(h, x_test[i], x_train)

        return P_x

    def fit(self, X, Y):
        """
        Fits the model to the training set.
        Since KDE is a lazy learner we just need to save the necessary information.

        :param X: Array of input data
        :param Y: Array of output labels
        :return: None
        """
        self.x_train = X
        self.y_train = Y
        self.dim = X.shape[1]
        labels = set(Y)
        for c in labels:
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def predict_proba(self, x_test):
        """
        Predict the probabilities of each class for the given input data.

        :param x_test: Array of input data to make predictions.
        :return: A list of probabilities for each class for each row in x_test.
        """
        N, D = x_test.shape
        priors = self.priors
        K = len(priors)
        P = np.zeros((N, K))
        x_train = self.x_train
        y_train = self.y_train
        if self.multi_bw:
            bw = self.bandwidth
        else:
            bw = np.repeat(self.bandwidth, K)
        for c, p in priors.items():
            P[:, int(c)] = self.KDE(bw[int(c)], x_test, x_train[y_train == c]) * p

        return P

    def predict(self, x_test):
        """
        Predict the labels of testing set, using KDE.

        :param x_test: Array of input data to make predictions.
        :return: Predicted labels of the data.
        """
        P = self.predict_proba(x_test)
        pred_y = np.argmax(P, axis=1)
        self.pred_y = pred_y

        return pred_y

    def accuracy(self, y_test):
        """
        Calculates the accuracy between the predicted label and actual labels.

        :param y_test: Array of actual output labels of Testing set.
        :return: A float value giving the accuracy.
        """
        pred_y = self.pred_y
        return np.array([pred_y == y_test]).mean()

    def score(self, x_test, y_test):
        """
        Function that runs both Predict and Accuracy and returns the accuracy
        score of the model.

        :param x_test: Array of input data to make predictions.
        :param y_test: Array of actual output labels of Testing set.
        :return: A float value giving the accuracy of the model.
        """
        self.predict(x_test)
        return self.accuracy(y_test)
