import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV


def features_correlations(X):
    """
     Compute linear and non-linear correlations between features in the dataset.

     Parameters:
     X : pandas.DataFrame
         Input dataset containing features.

     Returns:
     merged_corr_df : pandas.DataFrame
         DataFrame containing the correlations between features, sorted by absolute Pearson and Spearman correlations.
         Columns include 'Var 1', 'Var 2', 'Corr Pearson', 'Corr Spearman', 'Abs Corr Pearson', and 'Abs Corr Spearman'.
     linear_correlations : pandas.DataFrame
         DataFrame of linear correlations between features using Pearson method.
     non_linear_correlations : pandas.DataFrame
         DataFrame of non-linear correlations between features using Spearman method.
     """
    linear_correlations = X.corr(method='pearson')
    non_linear_correlations = X.corr(method='spearman')

    linear_corr_df = pd.DataFrame(linear_correlations.unstack().sort_values(ascending=False), columns=['Corr Pearson'])
    non_linear_corr_df = pd.DataFrame(non_linear_correlations.unstack().sort_values(ascending=False),
                                      columns=['Corr Spearman'])

    merged_corr_df = pd.merge(linear_corr_df, non_linear_corr_df, left_index=True, right_index=True)
    merged_corr_df['Abs Corr Pearson'] = abs(merged_corr_df['Corr Pearson'])
    merged_corr_df['Abs Corr Spearman'] = abs(merged_corr_df['Corr Spearman'])

    merged_corr_df = merged_corr_df[
        merged_corr_df.index.get_level_values(0) < merged_corr_df.index.get_level_values(1)
        ]

    merged_corr_df = merged_corr_df.sort_values(by=['Abs Corr Pearson', 'Abs Corr Spearman'], ascending=False)
    merged_corr_df.reset_index(inplace=True)
    merged_corr_df.rename(columns={'level_0': 'Var 1', 'level_1': 'Var 2'}, inplace=True)
    return merged_corr_df, linear_correlations, non_linear_correlations

def target_correlations(X, y):
    """
    Compute linear and non-linear correlations between features and target variable.

    Parameters:
    X : pandas.DataFrame
       Input dataset containing features.
    y : pandas.Series
       Target variable.

    Returns:
    corr_df : pandas.DataFrame
       DataFrame containing the correlations between features and the target variable, sorted by absolute Pearson
       and Spearman correlations. Columns include 'Corr Pearson', 'Corr Spearman', 'Abs Corr Pearson',
       and 'Abs Corr Spearman'.
    """
    pearson_corr = X.corrwith(y, method='pearson')
    spearman_corr = X.corrwith(y, method='spearman')

    pearson_corr_df = pd.DataFrame(pearson_corr, columns=['Corr Pearson'])
    spearman_corr_df = pd.DataFrame(spearman_corr, columns=['Corr Spearman'])

    corr_df = pd.merge(pearson_corr_df, spearman_corr_df, left_index=True, right_index=True)
    corr_df['Abs Corr Pearson'] = abs(corr_df['Corr Pearson'])
    corr_df['Abs Corr Spearman'] = abs(corr_df['Corr Spearman'])
    corr_df = corr_df.sort_values(by=['Abs Corr Pearson', 'Abs Corr Spearman'], ascending=False)

    return corr_df


def select_features_with_rfecv(X, y, model, cv=5, file_name='feature_selection.csv'):
    """
    Select features using Recursive Feature Elimination with Cross-Validation (RFECV).

    Parameters:
    X : pandas.DataFrame
        Input dataset containing features.
    y : pandas.Series
        Target variable.
    model : object
        Estimator object implementing 'fit' and 'predict' methods.
    cv : int, default=5
        Number of cross-validation folds.
    file_name : str, default='feature_selection.csv'
        Name of the file to save feature selection results.

    Returns:
    selected_features : array-like
        List of selected feature names.
    feature_importances : array-like or None
        Feature importances if available, otherwise None.
    """
    rfecv_selector = RFECV(estimator=model, step=1, cv=cv)
    rfecv_selector.fit(X, y)
    selected_features = X.columns[rfecv_selector.support_]

    if hasattr(rfecv_selector.estimator_, 'feature_importances_'):
        feature_importances = rfecv_selector.estimator_.feature_importances_
    elif hasattr(rfecv_selector.estimator_, 'coef_'):
        feature_importances = rfecv_selector.estimator_.coef_
    else:
        feature_importances = None

    df = pd.DataFrame({
        'Method': [model.__class__.__name__],
        'Selected features': [selected_features],
        'Importances': [feature_importances]
    })
    df.to_csv(file_name, mode='a', header=False, index=False)

    print(f"Selected features: {selected_features}")
    n_scores = len(rfecv_selector.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(1, n_scores + 1),
        rfecv_selector.cv_results_["mean_test_score"],
        yerr=rfecv_selector.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()
    return selected_features, feature_importances
