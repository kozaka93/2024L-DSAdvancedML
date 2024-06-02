from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

def get_classfier_param_grid(code, best_params={}):
    """
    Generates ML Pipeline consisting of Standard scaler and one of 11 classifiers
    based on provided code. It also generates a grid of hyperparameters to tune for each model.

    Parameters:
        code (strng): One of 11 values that corresponds to type of classifier:
            'rf' - Random Forest, 'mlp' - Multilayer Perceptron, 'svm' - Support Vector Machnes,
            'gp' - Gaussan Process, 'qda' - Quadratic discriminant analysis, 'xgb' - XGBoost,
            'nb' - Naive Bayes, 'knn' - K-Nearest Neighbors, 'ada'- AdaBoost, 
            'lr' - Logistic Regression, 'ert' - Extremely Randomized Trees}
        best_params (dict): mapping of hyperparameters for model, if one want to specify them 
                        without tuning.

    Returns: ML Pipeline, hyperparameters grid dictionary
    """

    if code == 'rf':
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=50, random_state=0, **best_params)),
            ]), {
            "rf__max_depth": [2, 4, 6],
            "rf__min_samples_split": [2, 4, 6],
            "rf__min_samples_leaf": [2, 5, 10,],
            "rf__max_features": ["sqrt", "log2"],
            },

    elif code == 'mlp':
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(early_stopping=True, tol=0.005, n_iter_no_change=8, **best_params)),
            ]), {
        'mlp__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'mlp__activation': ['tanh', 'relu'],
        'mlp__solver': ['sgd', 'adam'],
        'mlp__alpha': [0.0001, 0.05],
        'mlp__learning_rate': ['constant','adaptive']}
    
    
    elif code == 'svm':
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(random_state=0, **best_params, probability=True)),
            ]),{
            "svm__C": [0.1, 1, 10],
            "svm__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "svm__gamma": ["scale", "auto"],
        },


    elif code == 'gp':
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("gpc", GaussianProcessClassifier(kernel=RBF(1.0))),
            ]
        ), {}
    

    elif code == 'qda':
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("qda", QuadraticDiscriminantAnalysis()),
            ]
        ), {}
    

    elif code == 'xgb':
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("xgb", XGBClassifier(random_state=0, use_label_encoder=False, **best_params)),
            ]),{
            "xgb__n_estimators": [100, 200, 300],
            "xgb__max_depth": [3, 4, 5],
            "xgb__learning_rate": [0.01, 0.1, 0.3],
            "xgb__subsample": [0.8, 1.0],
        }
    
    
    elif code == 'nb':
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("nb", GaussianNB())
            ]), {
            "nb__var_smoothing": [1e-9, 1e-8, 1e-7]
        }
    
    elif code == 'knn':
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier())
            ]), {
            "knn__n_neighbors": [3, 5, 7, 9],
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2]
        }
    
    elif code == 'ada':
        return Pipeline([
                ("scaler", StandardScaler()),
                ("ada", AdaBoostClassifier())
            ]), {
            "ada__n_estimators": [50, 100, 200],
            "ada__learning_rate": [0.1, 0.5, 1.0],
            "ada__algorithm": ["SAMME", "SAMME.R"]
        }
    elif code == 'lr':
        return Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression())
            ]), {
            "lr__C": [0.001, 0.01, 0.1, 1.0],
            "lr__penalty": ["l1", "l2"],
            "lr__solver": ["liblinear", "saga"]
        }

    elif code == 'ert':
        return Pipeline([
                ("scaler", StandardScaler()),
                ("ert", ExtraTreesClassifier(n_estimators=50))
            ]), {
            "ert__max_depth": [None, 10, 20],
            "ert__min_samples_split": [2, 5, 10],
            "ert__min_samples_leaf": [1, 2, 4],
            "ert__max_features": ["sqrt", "log2"]
        }
    else:
        raise Exception('Code not recognizible')
    

def fit_grid_search(X, y, clf_code):
    """
    Creates hyperparameter grid search pipeline based on cross validation and fits it
    to provided data: X and y.

    Parameters:
        X (data frame): data to train
        y (list, np.array, pd.series): target variable
        clf_code: classfier code identifying the type

    Returns: fitted model
    """
    clf, param_grid = get_classfier_param_grid(clf_code)
    return GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=0),
        n_jobs=-1,
    ).fit(X=X, y=y)



def recreate_clf(X, y, model_library, index):
    """
    Recreates the classifier obtained by fit_grid_search function based on model library 
    data frame and index of this specific model. Then, model is fitted to provided data: X and y.
    
    Parameters:
        X (data frame): data to train
        y (list, np.array, pd.series): target variable
        model_library (data frame): saved models library
        index (int): index of desired model in model_library data frame

    Returns: fitted model
    """
    code, params, features = model_library.loc[index][['Model', 'hyperparameters', 'Features']]
    model, _ = get_classfier_param_grid(code, eval(params))
    return model.fit(X[eval(features)], y), eval(features)



def rename_hyperparameters(params, code):
    """
    Renames hyperaparameters: from nomenclature used in GridSearhCV, where
    parameters need to have prefix, to only names needed to reproduce the model.

    Parameters:
        params (dict): dictionary of parameters
        code (string): code of classifier type.

    Returns: modified dicitonary of parameters.
    """
    new_params = {}
    for key, val in params.items():
        new_key = key.replace(f'{code}__', '') 
        new_params[new_key] = val
    return new_params