import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
import autogluon
from autogluon.tabular import TabularPredictor


def alter_dataset(X_train, y_train, final_selected_features, percent = 0.25):
    """
    Train an SVC model and alter the dataset by relabeling a percentage of false negatives.

    Parameters:
    X_train (pd.DataFrame): The feature matrix for the training dataset.
    y_train (pd.DataFrame): The target variable for the training dataset.
    final_selected_features (list): A list of feature names to be used for training the model.
    percent (float, optional): The percentage of false negatives to relabel. Default is 0.25.

    Returns:
    sklearn.svm.SVC: The trained SVC model on the altered dataset.
    """
    model1 = SVC(degree=2, kernel='poly', probability=True, random_state=42)
    X_train_selected = X_train[final_selected_features]

    model1.fit(X_train_selected, y_train)
    y_pred  = model1.predict(X_train_selected)

    false_negatives = (np.array(y_train[0]) == 1) & (y_pred == 0)
    false_indices = np.where(false_negatives)[0]

    # Randomly select percent of false negatives
    num_false_negatives = len(false_indices)
    num_to_select = int(percent * num_false_negatives)
    print(f'num to select: {num_to_select}')
    selected_indices = np.random.choice(false_indices, num_to_select, replace=False)
    y_train_new = y_train.copy()
    y_train_new.iloc[selected_indices, 0] = 0
    model2 = SVC(degree=2, kernel='poly', probability=True, random_state=42).fit(X_train_selected, y_train_new)

    return model2


def custom_scorer(num):
    """
    Create a custom scorer function for evaluating model predictions.

    This function generates a custom scorer based on a specified number, which will be used
    to calculate a score for model predictions. The score is calculated based on the top 20%
    of predicted probabilities and true labels.

    Parameters:
    num (int): A constant value used in the custom score calculation.

    Returns:
    sklearn.metrics._scorer._PredictScorer: A custom scorer function that can be used in model evaluation.
    """
    def count_score(y_true, y_pred_proba):
        """
        Calculate the custom score for model predictions.

        Parameters:
        y_true (pd.Series): The true labels of the instances.
        y_pred_proba (np.ndarray): The predicted probabilities for the positive class.

        Returns:
        float: The calculated custom score.
        """
        top_indices = np.argsort(y_pred_proba)[-int(0.2*len(y_pred_proba)):]
        top_labels = y_true.iloc[top_indices]
        num_label = top_labels[0].sum()
        price = 10*num_label/(len(y_true)/5000) - 200*num
        return price
    return make_scorer(count_score, needs_proba=True)


def custom_score(y, y_pred_proba, n_features):
    """
    Calculate a custom score based on predicted probabilities and the number of features.

    Parameters:
    y (pd.Series): The true labels of the instances.
    y_pred_proba (np.ndarray): The predicted probabilities for the positive class.
    n_features (int): The number of features used in the model.

    Returns:
    float: The calculated custom score.
    """
    top_indices = np.argsort(y_pred_proba)[-int(0.2*len(y)):]
    top_labels = y.iloc[top_indices]
    num_label_1 = top_labels[0].sum()
    score = 10*num_label_1/(len(y)/5000) - 200 * n_features
    return score


def evaluate(X_train, X_valid, y_train, y_valid, model):
    """
    Evaluate a model using custom scoring on training and validation datasets.

    Parameters:
    X_train (pd.DataFrame): The feature matrix for the training dataset.
    X_valid (pd.DataFrame): The feature matrix for the validation dataset.
    y_train (pd.Series): The true labels for the training dataset.
    y_valid (pd.Series): The true labels for the validation dataset.
    model (sklearn.base.BaseEstimator): The machine learning model to be evaluated.

    Returns:
    tuple: A tuple containing the custom scores for the training and validation datasets 
           (score_train, score_valid).
    """
    proba_train = model.predict_proba(X_train)[:, 1]
    score_train = custom_score(y_train, proba_train, len(X_train.columns))
    print(f"Train score: {score_train}")

    proba_valid = model.predict_proba(X_valid)[:, 1]
    score_valid = custom_score(y_valid, proba_valid, len(X_valid.columns))
    print(f"Validation score: {score_valid}")
    return score_train, score_valid


def search(model, features, X, y):
    """
    Perform cross-validation to evaluate a model using a custom scorer.

    Parameters:
    model (sklearn.base.BaseEstimator): The machine learning model to be evaluated.
    features (list): A list of feature names to be used for training the model.
    X (pd.DataFrame): The feature matrix containing the data.
    y (pd.Series): The target variable.

    Returns:
    list: A list of custom scores from all cross-validation trials.
    """
    print(f"model: {model}, features: {features}")
    num = len(features)
    all_scores = []
    for i in range(5):
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        scores = cross_val_score(model, X[features], y, cv=kf, scoring=custom_scorer(num))
        all_scores.extend(scores)
        print(f"Mean score: {scores.mean()}")
    print(f"Mean score for all trials: {np.mean(all_scores)}")
    return all_scores


# --------------------------- functions used with Autogluon ---------------------------
def n_largest_indices(arr, n):
    """
    Returns the indices of the n largest elements in a numpy array.
    
    Parameters:
        arr (numpy.ndarray): The input array.
        n (int): The number of largest elements to find.
    
    Returns:
        numpy.ndarray: The indices of the n largest elements.
    """
    flat_arr = arr.flatten()
    sorted_indices = np.argsort(flat_arr)
    largest_indices = sorted_indices[-n:]
    return largest_indices


def custom_score_for_autogluon(y_true:np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate a custom score based on the top 20% predicted probabilities and true labels.
    Adjusted to as an eval_metric in Autogluon.

    Parameters:
    y_true (np.ndarray): The true labels of the instances.
    y_pred_proba (np.ndarray): The predicted probabilities for the positive class.

    Returns:
    float: The calculated custom score.
    """    
    idxs = n_largest_indices(y_pred_proba, len(y_pred_proba)//5)
    y_pred = y_pred_proba>0.5
    y_pred = y_pred.astype(float)
    y_true = np.array(y_true)
    result = (sum([1 == y_true[i] == y_pred[i] for i in idxs])*10*1000/len(idxs) - 3* 200)#/10000
    return result


def calculate_metric_automl(features, X, y):
    """
    Calculate custom metrics for an AutoML model using stratified cross-validation.

    This function trains an AutoML model using the specified features and target variable,
    evaluates it using a custom scoring function with stratified k-fold cross-validation,
    and returns the custom metrics along with the path to the best model.

    Parameters:
    features (list): A list of feature names to be used for training the model.
    X (pd.DataFrame): The feature matrix containing the data.
    y (pd.Series): The target variable.

    Returns:
    tuple: A tuple containing:
           - metrics (list): A list of custom scores for each fold in the cross-validation.
           - path_to_best_model (str): The path to the best model saved by the AutoML predictor.
    """
    metrics = []
    metric_scorer = autogluon.core.metrics.make_scorer(name='custom_score_for_autogluon',
                                                       score_func=custom_score_for_autogluon,
                                                       optimum=1,
                                                       greater_is_better=True,
                                                       needs_proba=True)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.loc[train_index,:], X.loc[test_index, :]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        data_train = X_train.loc[:,features]
        data_train['class'] = y_train

        data_test = X_test.loc[:,features]
        data_test['class'] = y_test
 
        predictor = TabularPredictor(label='class', eval_metric=metric_scorer,
                                     ).fit(data_train, presets = ['good_quality', 'optimize_for_deployment'])#,presets = ['good_quality']) 
        print(predictor.leaderboard(data_test, silent=True))
        predictions = predictor.predict_proba(data_test)
        
        metrics.append(custom_score_for_autogluon(np.array(y_test).T[0], np.array(predictions[1])))
        path_to_best_model = predictor.path +'/'+predictor.get_model_best()
    return metrics, path_to_best_model
