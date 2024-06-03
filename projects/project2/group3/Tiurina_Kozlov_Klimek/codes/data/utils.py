from typing import Optional, List, Callable, Dict

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso, SGDClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from skrebate import ReliefF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

from aml_projects.Project2.data.kdenb import KDENaiveBayesClassifier


def cleanup_dataset_remove_features_vif(
        df: pd.DataFrame,
        target: Optional[str | int] = None,
        threshold: float = 10.0
) -> List[str | int]:
    """
    Remove features with VIF greater than threshold.
    :param df: DataFrame - input data
    :param target: str - target column name
    :param threshold: float - VIF threshold
    :return: List[str | int] - list of removed features
    """

    features = list(df.columns)
    if target is not None:
        features.remove(target)

    all_removed_features = []
    while True:
        vif = pd.DataFrame()
        vif["features"] = features
        vifs = []

        tqdm._instances.clear()
        bar = tqdm(total=len(features), desc="Calculating VIF")
        for i in range(len(features)):
            vifs.append(variance_inflation_factor(df[features].values, i))
            bar.update(1)

        bar.close()
        vif["VIF"] = vifs

        max_vif = vif['VIF'].max()
        if max_vif > threshold:
            print(f"Removing feature {vif.loc[vif['VIF'].idxmax()]['features']} with VIF {max_vif}")
            vif_max_feature = vif.loc[vif['VIF'].idxmax()]['features']
            features.remove(vif_max_feature)
            all_removed_features.append(vif.loc[vif['VIF'].idxmax()]['features'])
        else:
            break

    return all_removed_features


def cleanup_dataset_apply_standard_scaler(
        df: pd.DataFrame,
        target: Optional[str | int] = None
) -> pd.DataFrame:
    """
    Apply StandardScaler to the features

    :param df: DataFrame - input data
    :param target: str - target column name

    :return: DataFrame - cleaned data
    """
    df = df.copy()

    features = list(df.columns)
    if target is not None:
        features.remove(target)
    scaler = StandardScaler()

    if target is not None:
        df[features] = scaler.fit_transform(df[features])

    return df


class MultiCollinearityEliminator:

    def __init__(
            self,
            df: pd.DataFrame,
            threshold: float = 0.9
    ):
        """
        MultiCollinearityEliminator class is created to eliminate multicollinearity from the dataframe.
        From: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on

        :param df: DataFrame - input data
        :param threshold: float - threshold value to consider the correlation between the features
        """

        self.df = df
        self.threshold = threshold

    def __create_correlation_matrix(self, current_filtered_features: List[str | int]) -> pd.DataFrame:
        """
        Create the correlation matrix of the dataframe.
        If include_target is True, the target column will be included in the correlation matrix.

        :param current_filtered_features: List[str | int] - list of features to be excluded from the correlation matrix
        :return: DataFrame - correlation matrix
        """
        # Checking we should include the target in the correlation matrix
        df_temp = self.df.drop(columns=current_filtered_features)
        # Setting method to Pearson to prevent issues in case the default method for df.corr() gets changed
        # Setting min_period to 30 for the sample size to be statistically significant (normal) according to
        # central limit theorem
        correlation_matrix = df_temp.corr(method='pearson', min_periods=30).abs()

        return correlation_matrix

    def __create_correlated_features_list(
            self,
            current_filtered_features: List[str | int]
    ) -> List[str | int]:
        """
        Create the list of correlated features based on the threshold value.

        :return: list - list of correlated features
        """

        # Obtaining the correlation matrix of the dataframe (without the target)
        correlation_matrix = self.__create_correlation_matrix(current_filtered_features)
        correlated_columns = []

        # Iterating through the columns of the correlation matrix dataframe
        for column in correlation_matrix.columns:
            # Iterating through the values (row wise) of the correlation matrix dataframe
            for idx, row in correlation_matrix.iterrows():
                if (row[column] > self.threshold) and (row[column] < 1):
                    # Adding the features that are not already in the list of correlated features
                    if idx not in correlated_columns:
                        correlated_columns.append(idx)
                    if column not in correlated_columns:
                        correlated_columns.append(column)

        return correlated_columns

    def auto_eliminate_multicollinearity(self) -> List[str | int]:
        """
        Automatically eliminate multicollinearity from the dataframe.

        :return: List[str | int] - list of removed features
        """

        # Obtaining the list of correlated features
        correlated_columns = self.__create_correlated_features_list([])

        all_correlated_columns = correlated_columns.copy()
        while correlated_columns:
            # Obtaining the dataframe after deleting the feature (from the list of correlated features)
            # that is least correlated with the target
            correlated_columns = self.__create_correlated_features_list(all_correlated_columns)
            all_correlated_columns.extend(correlated_columns)

        return all_correlated_columns


def cleanup_dataset_remove_features_correlation(
        df: pd.DataFrame,
        target: Optional[str | int] = None,
        threshold: float = 0.9
) -> List[str | int]:
    """
    Remove features with correlation greater than threshold.

    :param df: DataFrame - input data
    :param target: str - target column name
    :param threshold: float - correlation threshold
    :return: List[str | int] - list of removed features
    """
    df = df.copy()
    if target is not None:
        df = df.drop(columns=[target])

    eliminator = MultiCollinearityEliminator(df, threshold)
    return eliminator.auto_eliminate_multicollinearity()


SelectFeaturesMethod = Callable[[pd.DataFrame, pd.Series, int], pd.Series]


def select_features_rfecv_forest(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Recursive Feature Elimination with Random Forest

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    rfe = RFECV(
        estimator=model,
        step=0.1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=1,
        n_jobs=-1
    )

    rfe = rfe.fit(X, y)
    feature_mask = rfe.support_
    return X.columns[feature_mask]


def select_features_rfecv_support_vector_machine(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Recursive Feature Elimination with Support Vector Machine

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = SVC(kernel='rbf', random_state=random_state)
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    rfe = RFECV(
        estimator=model,
        step=0.1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=1,
        n_jobs=-1
    )

    rfe = rfe.fit(X, y)
    feature_mask = rfe.support_
    return X.columns[feature_mask]


def select_features_forest(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Forest

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = RandomForestClassifier(
        n_jobs=-1,
        random_state=random_state,
        n_estimators=500
    )
    model = model.fit(X, y)
    sfm = SelectFromModel(model, prefit=True, max_features=20)
    feature_mask = sfm.get_support()
    return X.columns[feature_mask]


def select_features_rfecv_sgd(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Perceptron

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = SGDClassifier(loss="perceptron", random_state=random_state, n_jobs=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    rfe = RFECV(
        estimator=model,
        step=0.1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=1,
        n_jobs=-1
    )

    rfe = rfe.fit(X, y)
    feature_mask = rfe.get_support()
    return X.columns[feature_mask]


def select_features_support_vector_machine(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Support Vector Machine

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = LinearSVC(random_state=random_state, dual=False)
    model = model.fit(X, y)
    sfm = SelectFromModel(model, prefit=True, max_features=20, importance_getter=lambda x: x.dual_coef_)
    feature_mask = sfm.get_support()
    return X.columns[feature_mask]


def select_features_lasso(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using Lasso

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    alpha = GridSearchCV(
        Lasso(random_state=random_state),
        param_grid={'alpha': np.arange(0.00001, 10, 500)},
        cv=kf,
        n_jobs=-1
    )
    alpha = alpha.fit(X, y)

    model = Lasso(alpha=alpha.best_params_['alpha'], random_state=random_state)
    model = model.fit(X, y)

    sfm = SelectFromModel(model, prefit=True, max_features=20)
    feature_mask = sfm.get_support()
    return X.columns[feature_mask]


def select_features_xgb(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using XGBoost

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    model = GradientBoostingClassifier(random_state=random_state, n_estimators=500, n_iter_no_change=10)
    model = model.fit(X, y)
    sfm = SelectFromModel(model, prefit=True, max_features=20)
    feature_mask = sfm.get_support()
    return X.columns[feature_mask]


def select_features_mrmr(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using mRMR

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    X_string_index = X.copy()
    X_string_index.columns = [str(i) for i in range(X.shape[1])]
    mrmr = mrmr_classif(X_string_index, y, n_jobs=-1, K=20, show_progress=False, relevance="ks")

    return X.columns[[int(i) for i in mrmr]]


def select_features_surf(X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.Series:
    """
    Select features using MultiSURFstar

    :param X: DataFrame - input data
    :param y: DataFrame - target data
    :param random_state: int - random state
    :return: Series - selected features
    """
    mss = ReliefF(n_features_to_select=20, n_jobs=-1)
    mss.fit(X.values, y)
    top_features = mss.top_features_
    return X.columns[top_features[:20]]


feature_selectors: Dict[str, SelectFeaturesMethod] = {
    'mrmr': select_features_mrmr,
    'surf': select_features_surf,
    'forest': select_features_forest,
    'xgb': select_features_xgb,

    # Linear models perform poorly on the dataset
    # 'lasso': select_features_lasso,
    # 'support_vector_machine': select_features_support_vector_machine,

    # Recursive Feature Elimination is unusable due to the large number of features in the dataset
    # 'rfecv_support_vector_machine': select_features_rfecv_support_vector_machine,
    # 'rfecv_sgd': select_features_rfecv_sgd,
    # 'rfecv_forest': select_features_rfecv_forest,
}


ApplyModelMethod = Callable[[pd.DataFrame, pd.Series, pd.DataFrame, Optional[int]], pd.Series]
# ApplyModelMethod does not return predictions for the test data
# It returns probability the customer will buy the product - probability of class 1 for each customer


def apply_model_logistic_regression(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Logistic Regression model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - probability of class 1 for each customer
    """
    model = LogisticRegression(max_iter=10000, random_state=random_state, n_jobs=-1)
    model = model.fit(X, y)

    return pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)


def apply_model_random_forest(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Random Forest model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - probability of class 1 for each customer
    """
    model: RandomForestClassifier = RandomForestClassifier(n_jobs=-1, random_state=random_state, n_estimators=500)
    model.fit(X, y)

    return pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)


def apply_model_gradient_boosting_classifier(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Gradient Boosting Classifier model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - probability of class 1 for each customer
    """
    model = GradientBoostingClassifier(random_state=random_state, n_estimators=500, n_iter_no_change=10, max_depth=5)
    model.fit(X, y)

    return pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)


def apply_model_support_vector_machine(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Support Vector Machine model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - probability of class 1 for each customer
    """
    # Prefer 1 over 0
    model = SVC(
        random_state=random_state,
        kernel='rbf',
        probability=True,
        # class_weight={0: 1, 1: 10}
    )
    model.fit(X, y)

    return pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)


def apply_model_mlp(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Perceptron model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - probability of class 1 for each customer
    """
    model = MLPClassifier(random_state=random_state, hidden_layer_sizes=(10, 50, 50, 10), max_iter=1000)
    model.fit(X, y)

    return pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)


def apply_model_sgd(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply SGD model

    :param X: DataFrame - input data
    :param y: Series - target data
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - probability of class 1 for each customer
    """
    model = SGDClassifier(loss="modified_huber", random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    return pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)


def apply_model_knn(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply K-Nearest Neighbors model

    :param X: DataFrame - input data
    :param y: Series - target
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - probability of class 1 for each customer
    """
    model = KNeighborsClassifier(n_jobs=-1, n_neighbors=250)
    model.fit(X, y)

    return pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)


def apply_model_qda(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Quadratic Discriminant Analysis model

    :param X: DataFrame - input data
    :param y: Series - target
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - probability of class 1 for each customer
    """
    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)

    return pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)


def apply_model_nb(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply Naive Bayes model

    :param X: DataFrame - input data
    :param y: Series - target
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - probability of class 1 for each customer
    """
    model = GaussianNB()
    model.fit(X, y)

    return pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)


def apple_model_kdenb(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        random_state: int = 0
) -> pd.Series:
    """
    Apply K-Nearest Neighbors model

    :param X: DataFrame - input data
    :param y: Series - target
    :param X_test: DataFrame - test data
    :param random_state: int - random state
    :return: Series - probability of class 1 for each customer
    """
    model = KDENaiveBayesClassifier(
        multi_bw=True,
        bandwidth=[0.1, 0.5],
        kernel='radial'
    )
    model.fit(X.values, y.values.ravel())

    return pd.Series(model.predict_proba(X_test.values)[:, 1], index=X_test.index)


model_appliers: Dict[str, ApplyModelMethod] = {
    'random_forest': apply_model_random_forest,
    'gradient_boosting_classifier': apply_model_gradient_boosting_classifier,
    'mlp': apply_model_mlp,
    'sgd': apply_model_sgd,
    'support_vector_machine': apply_model_support_vector_machine,
    'logistic_regression': apply_model_logistic_regression,
    'knn': apply_model_knn,
    'qda': apply_model_qda,
    'nb': apply_model_nb,
    'kdenb': apple_model_kdenb,
}


def select_customers(
        predicted_probabilities: pd.Series,
        threshold_num: int = 1000
) -> pd.Series:
    """
    Select customers based on the predicted probabilities

    :param predicted_probabilities: Series - predicted probabilities
    :param threshold_num: int - number of customers to select
    :return: Series - selected customers, a 0 or 1 for each customer
    """
    probability_threshold = predicted_probabilities.quantile(1 - threshold_num / len(predicted_probabilities))
    return pd.Series(predicted_probabilities > probability_threshold, index=predicted_probabilities.index)


def compute_score(
        predicted: pd.Series,
        actual: pd.Series,
        feature_num: int,
        should_penalize_feature_num: bool = True,
        threshold_num: int = 1000
) -> int:
    """
    Compute score based on the number of correctly predicted customers and the number of variables used.
    :param predicted: pd.Series - Predicted values
    :param actual: pd.Series - Actual values
    :param feature_num: int - Number of variables used
    :param should_penalize_feature_num: bool - Should penalize the number of variables used
    :param threshold_num: int - Number of customers to select
    :return: int - Score
    """
    correct_instances_num = len(np.intersect1d(np.where(predicted.values == 1), np.where(actual.values == 1)))
    score = 10 * correct_instances_num + (
        # Since 200 is for 1000 customers, we need to scale 200 to threshold_num
        (-200 * feature_num * threshold_num / 1000)
        if should_penalize_feature_num
        else 0
    )
    return max(0, score)


def max_score(
        threshold_num: int = 1000
) -> int:
    """
    Compute the maximum score that can be achieved
    :param threshold_num: int - Number of customers to select
    :return: int - Maximum score
    """
    return 10 * threshold_num


GenerateFeatureInteractionsMethod = Callable[[pd.DataFrame], pd.DataFrame]


def generate_feature_interactions_noop(X: pd.DataFrame) -> pd.DataFrame:
    """
    No operation function to generate feature interactions
    :param X: DataFrame - input data
    :return: DataFrame - input data
    """
    return X


def generate_feature_interactions_quadratic(X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate quadratic feature interactions
    :param X: DataFrame - input data
    :return: DataFrame - input data with quadratic feature interactions
    """
    return pd.DataFrame(
        np.hstack([X.values, X.values ** 2]),
        columns=list(map(str, X.columns)) + [f"{col}^2" for col in X.columns],
        index=X.index
    )


def train_and_evaluate_model(
        model: ApplyModelMethod,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        selected_features: List[str | int],
        generate_feature_interactions: GenerateFeatureInteractionsMethod = generate_feature_interactions_noop,
        should_penalize_feature_num: bool = True,
        random_state: int = 0,
) -> int:
    """
    Train and evaluate the model
    1. Select features using the selected feature selection method
    2. Train the model
    3. Select customers
    4. Compute the score

    :param model: ApplyModelMethod - model to apply
    :param X_train: DataFrame - training data
    :param y_train: Series - training target data
    :param X_test: DataFrame - test data
    :param y_test: Series - test target data
    :param selected_features: List[str | int] - selected features
    :param generate_feature_interactions: GenerateFeatureInteractionsMethod - function to generate feature interactions
    :param should_penalize_feature_num: bool - should penalize the number of variables used
    :param random_state: int - random state
    :return: int - score
    """
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    X_train = generate_feature_interactions(X_train)
    X_test = generate_feature_interactions(X_test)

    predicted_probabilities = model(X_train, y_train, X_test, random_state)
    threshold_num = np.sum(y_test.values)  # Number of customers to select is the number of customers who bought the product
    selected_customers = select_customers(predicted_probabilities, threshold_num)

    return compute_score(
        selected_customers,
        y_test,
        len(selected_features),
        should_penalize_feature_num,
        threshold_num
    )
