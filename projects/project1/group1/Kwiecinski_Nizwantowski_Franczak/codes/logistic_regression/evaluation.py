import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from . import LogisticRegressor
from typing import Union, List, Literal
import pandas as pd

import matplotlib.pyplot as plt

def confusion_matrix(y_true: Union[np.ndarray, pd.DataFrame], y_pred: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    function returns confusion matrix
    
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    TP = np.sum(y_true & y_pred)
    TN = np.sum(~y_true & ~y_pred)
    FP = np.sum(~y_true & y_pred)
    FN = np.sum(y_true & ~y_pred)
    
    return np.array([[TP, FP], [FN, TN]])

def balanced_accuracy(y_true: Union[np.ndarray, pd.DataFrame], y_pred: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    function returns balanced accuracy of classification
    """

    conf = confusion_matrix(y_true, y_pred)
    TPR = conf[0, 0] / (conf[0, 0] + conf[1, 0])
    TNR = conf[1, 1] / (conf[1, 1] + conf[0, 1])

    return 0.5 * (TPR + TNR)


def compare_methods(X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame], methods: List[str], k = 5, test_size = 0.2):
    """
    function compares performance of several build-in classification models
    
    """

    available_methods = ["decision tree", "random forest", "LDA", "QDA", "logistic regression"]

    assert all([method in available_methods for method in methods]), "Invalid method, available methods are lda, qda, decision tree, random forest you have choosen " + str(methods)

    results = {}

    for iteration in range(k):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=iteration)

        for method in methods:
            if method == "LDA":
                model = LinearDiscriminantAnalysis()
            elif method == "QDA":
                model = QuadraticDiscriminantAnalysis()
            elif method == "decision tree":
                model = DecisionTreeClassifier()
            elif method == "random forest":
                model = RandomForestClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[method] = results.get(method, []) + [balanced_accuracy(y_test, y_pred)]

    return results

def plot_boundaries(X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame], method: Literal["decision tree", "random forest", "LDA", "QDA", "logistic regression"], alg: Literal[None, "iwls", "adam", "sgd"] = None, interactions = False, test_size = 0.2, save_plot = False):
    if method == "LDA":
        model = LinearDiscriminantAnalysis()

    elif method == "QDA":
        model = QuadraticDiscriminantAnalysis()

    elif method == "decision tree":
        model = DecisionTreeClassifier()

    elif method == "random forest":
        model = RandomForestClassifier()

    elif method == "logistic regression":
        if alg is None:
            raise ValueError("Please specify the algorithm for the logistic regression")
        
        model = LogisticRegressor(descent_algorithm=alg, include_interactions=interactions)
    else:
        raise ValueError("Invalid method " + method )

    X = X.values if isinstance(X, pd.DataFrame) else X
    y = y.values if isinstance(y, pd.DataFrame) else y

    print("Splitting the data into training and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=43)

    print(f"Fitting model with {X.shape[1]} features")
    print(f"Model: {method}" + ('' if alg is None else  ' with algorithm: ' + alg))

    model.fit(X_train, y_train)
    print("Model fitted!")
    print("Balanced accuracy on the test set: ", balanced_accuracy(y_test, model.predict(X_test)))

    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    if not isinstance(Z, np.ndarray):
        Z = Z.to_numpy()
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_test[y_test==0, 0], X_test[y_test == 0, 1], s=20, edgecolor='k')
    plt.scatter(X_test[y_test==1, 0], X_test[y_test == 1, 1], s=20, edgecolor='k')

    plt.suptitle("Classification with decision boundaries\n", fontsize=16)
    plt.title(f"for {method}{'' if alg is None else  ' with algorithm: ' + alg}{' with interactions' if interactions else  ''}; " +
            f"balanced accuracy: {balanced_accuracy(y_test, model.predict(X_test)):.2f}", fontsize=10)
    

    plt.legend(["y = 0", "y = 1"])
    if save_plot:
        plt.savefig(f"plots/artificial_data/{method}{'' if alg is None else f'_{alg}'}{'' if not interactions else '_inter'}.png")
    plt.show()
        

