import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from .model_managing import recreate_clf



def create_meta_sets(X_train, y_train, X_test, model_library, model_indeces):
    """
    Generates meta-sets containing prediction of component models of ensembles and probability
    of belonging to class 1, if model has such feature. The set is build on X_train and X_test.

    Parameters:
        X_train (data frame): Train set
        y_train (list, pd.series, np.array): train target labels
        X_test (data frame): Test set
        model_library (data frame): saved models library
        model_indeces (list): list of indeces of models in model_library data frame to include in ensemble

    Returns: train meta-set and test meta-set for stacking ensemble
    """
    df_train, df_test = pd.DataFrame(), pd.DataFrame()
    for i in range(len(model_indeces)):
        clf, features = recreate_clf(X_train, y_train, model_library, model_indeces[i])
        df_train[f'pred_{i}'] = clf.predict(X_train[features])
        df_test[f'pred_{i}'] = clf.predict(X_test[features])
        try: 
            df_train[f'proba_{i}'] = clf.predict_proba(X_train[features])[:,0]
            df_test[f'proba_{i}'] = clf.predict_proba(X_test[features])[:,0]
        except Exception:
            continue       
    return df_train, df_test


def stacking(X_train, y_train, X_test, model_library, model_indeces):
    """
    Performs stacking ensembling.
    Trains ensemble based on XGBoost meta-learner and generated meta-sets 
    and predicts labels on test set.

     Parameters:
        X_train (data frame): Train set
        y_train (list, pd.series, np.array): train target labels
        X_test (data frame): Test set
        model_library (data frame): saved models library
        model_indeces (list): list of indeces models in model_library data frame to include in ensemble
    
    Returns: prediction on X_test
    """
    df_train, df_test = create_meta_sets(X_train, y_train, X_test, model_library, model_indeces)
    meta_learner = GridSearchCV(estimator=XGBClassifier(n_estimators=50),
                                param_grid={
                                'learning_rate': [0.01, 0.1, 0.3],
                                'max_depth': [3, 4, 5],
                                'gamma': [0, 0.1, 0.2],
                                'subsample': [0.6, 0.8, 1.0],
                                'colsample_bytree': [1.0],
                                'objective': ['binary:logistic']
                            }, scoring='accuracy',
                            cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=0),
                            n_jobs=-1
                            ).fit(df_train, y_train)
    return meta_learner.predict(df_test)