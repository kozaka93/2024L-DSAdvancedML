import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def createFeatureInteractions(X : np.ndarray) -> np.ndarray:  
    """
    Create feature interactions for a given dataset X
    """
    n = X.shape[1]
    for i in range(n):
        for j in range(i+1, n):
            new_col = X[:, i] * X[:, j]
            X = np.column_stack((X, new_col))
    return X

def fitComparisonModels(X_train: np.ndarray, y_train : np.ndarray, X_test : np.ndarray) -> tuple:
    """
    fit comparison models to check accuracy vs our LogReg
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    lda_predictions = lda.predict(X_test)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    qda_predictions = qda.predict(X_test)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_predictions = dt.predict(X_test)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)

    return lda_predictions, qda_predictions, dt_predictions, rf_predictions