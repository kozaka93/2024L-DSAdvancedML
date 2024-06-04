import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector


def calculate_score(n_features: int, 
                    y_proba: np.array, 
                    y_val: np.array,
                    n_to_choose: int=200):
    '''
    Function calculating the score for the model.

    Input:
    model: chosen model
    X_val: array of predictors
    y_val: array of real target values

    Output:
    reward: final score of the model

    Function is making a prediction of probabilities, 
    then sorting the observations by the highest probability of belonging to the class 1. 
    Next, the score is calculated based on the task's description and scaled for the correct size of data.
    '''
    top_probas = np.argpartition(y_proba[:,1], -n_to_choose)[-n_to_choose:]
    correctly_chosen = np.sum([1 for x in top_probas if y_val[x]==1])
    coef = 1000/n_to_choose
    reward = correctly_chosen*10*coef-200*n_features

    return reward

def calculate_score_2(n_features, y_pred, y_val):
    correctly_chosen = np.sum(((y_pred==1)==(y_val==1))*1)
    reward = correctly_chosen*10-200*n_features

    return reward    
    

def prepare_scores_data(model, 
                        n_features_list: list, 
                        X_train: np.array,
                        y_train: np.array,
                        X_val: np.array, 
                        y_val: np.array, 
                        rank: np.array):
    '''
    Function returning list of scores for number of features specified with n_features_list parameter.

    Input:
    model: ...,
    n_features_list: list of numbers specyfing how many features we want to take for prediction
    X_train: array of training predictors
    y_train: array of training target values
    X_val: array of validation predictors
    y_val: array of validation target values
    rank: array specyfing how important the feature corresponding to the index is

    Output:
    scores_list: list of scores corresponding to number of features specified in n_features_list

    Function makes predictions for every number of features specified in n_features_list. It chooses the highest ranked features.

    Example:
    n_features_list: [1, 3, 5]
    rank: [5, 7, 1, 2, 4, 3, 6]

    Which means that feature 2 is the most important, next is feature 3, etc.
    The prediction will at first be made using only one feature - 2.
    Next the prediction will be made using 3 features - 2, 3 and 5.
    At the end, the predicion will be made using 5 features - 2, 3, 5, 4 and 0.

    '''
    scores_list = []

    for n_features in n_features_list:
        X_train_tmp = X_train.iloc[:,np.argwhere(rank <= n_features).flatten()]
        X_val_tmp = X_val.iloc[:,np.argwhere(rank <= n_features).flatten()]
        model.fit(X_train_tmp, y_train)
        y_pred = model.predict_proba(X_val_tmp)

        score = calculate_score(n_features, y_pred, y_val)

        scores_list.append(score)
    
    return scores_list


def plot_scores(n_features_list: list,
                scores_list: list,
                plot_title: str) -> None:
    """
    Plots the scores against the number of features.

    Parameters:
    n_features_list (list or array-like): A list or array containing the number of features.
    scores_list (list or array-like): A list or array containing the scores corresponding to each number of features.
    plot_title (str): The title of the plot.

    Returns:
    None: This function does not return any value. It displays a plot of the scores versus the number of features.
    """
    plt.plot(n_features_list, scores_list)
    plt.xlim(np.min(n_features_list)-1, np.max(n_features_list)+1)
    plt.xlabel("Number of features")
    plt.xticks(range(np.min(n_features_list)-1, np.max(n_features_list)+1, 5))
    plt.ylabel("Score ($)")
    plt.title(plot_title)
    plt.grid()
    plt.show()

def plot_scores_2(n_features_list, means, std, color, plot_title):
    plt.plot(n_features_list, means, color=color, linestyle="-")
    plt.fill_between(n_features_list, means - std, means + std, color=color, alpha=0.2)
    plt.xlabel("Number of features")
    plt.ylabel("score ($)")
    plt.title(plot_title)
    plt.show()

def remove_collinear_features(df: pd.DataFrame,
                              threshold: float):
    """
    Remove collinear features from a DataFrame based on a correlation threshold.

    This function calculates the absolute pairwise Pearson correlation matrix of the 
    DataFrame's features and identifies features to drop based on the specified 
    correlation threshold. Features that have a correlation greater than or equal 
    to the threshold with any other feature are removed.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features to evaluate for collinearity.
    threshold (float): The correlation threshold for identifying collinear features. 
                       Features with correlation values greater than or equal to this 
                       threshold will be dropped.

    Returns:
    pd.DataFrame: A DataFrame with collinear features removed based on the specified threshold.

    Notes:
    ------
    - If multiple features are collinear with each other, the function retains the 
      feature that appears first and removes subsequent collinear features.
    """
    corr_matrix = df.corr().abs()
    upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column]>=threshold)]
    print(f"Removed columns: {to_drop}")

    return df.drop(columns=to_drop)


def get_ranks_from_scores(scores: np.array,
                          best: str='max'):
    """
    Computes ranks for the given scores. By default, higher scores get lower rank numbers 
    (i.e., rank 1 for the highest score), but this behavior can be inverted to rank lower 
    scores higher by setting the `best` parameter to 'min'.
    
    Parameters
    ----------
    scores : np.array
        A numpy array of scores for which ranks need to be computed.
    best : str, optional
        A string indicating whether the highest or lowest score is considered the best.
        It should be either 'max' or 'min'. 
        - 'max': Higher scores are considered better.
        - 'min': Lower scores are considered better.
        Default is 'max'.
    
    Returns
    -------
    np.array
        An array of ranks corresponding to the input scores.
    
    Examples
    --------
    >>> import numpy as np
    >>> scores = np.array([50, 20, 30, 10, 40])
    >>> get_ranks_from_scores(scores)
    array([1, 5, 4, 6, 2])
    
    >>> get_ranks_from_scores(scores, best='min')
    array([5, 2, 3, 1, 4])
    """
    if best=='max':
        args_sorted = scores.argsort()[::-1]
    elif best=='min':
        args_sorted = scores.argsort()
    ranks = np.arange(len(scores))[args_sorted.argsort()]+1

    return ranks

def sfs(model_sel, model_pred, support, X_train, X_val, y_train, y_val, verbose=True):

    X_train_sfs = X_train.iloc[:,support]
    X_val_sfs = X_val.iloc[:,support]
    n_features = X_train_sfs.shape[1]-1
    rewards = []
    scores = []
    feature_nums = []
    feature_names = []
    while True:
        sfs = SequentialFeatureSelector(model_sel, n_features_to_select=n_features, cv=2, direction='backward')
        sfs.fit(X_train_sfs, y_train)
        sfs_sup = sfs.get_support()
        X_train_sfs = X_train_sfs.iloc[:, sfs_sup]
        X_val_sfs = X_val_sfs.iloc[:,sfs_sup]
        model_pred.fit(X_train_sfs, y_train)
        y_pred = model_pred.predict_proba(X_val_sfs)
        reward = calculate_score(n_features, y_pred, y_val)
        
        if verbose:
            print(f'N of features: {n_features}')
            print(f'Feature names: {list(X_train_sfs.columns)}')
            print(f'Reward: {reward}')
        
        rewards.append(reward)
        feature_nums.append(n_features)
        feature_names.append(list(X_train_sfs.columns))
        
        n_features-=1
        if n_features==0:
            break


    result_dict = {}

    result_dict['rewards'] = rewards

    return rewards