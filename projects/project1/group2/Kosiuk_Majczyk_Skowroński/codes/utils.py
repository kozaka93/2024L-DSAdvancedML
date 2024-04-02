import json
import openml
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import time
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score as bal_acc
import copy


def validate_datasets(datasets: dict):
    valid_datasets = []
    # Load datasets from json file
    
    num_datasets = len(datasets)
    failures = len(datasets)

    # print(datasets)

    for name, dataset in datasets.items():
        print(f"Dataset: {name}")
        
        if ((type(dataset['X'])==pd.DataFrame) and (type(dataset['y'])==pd.Series)):
            X, y = dataset["X"], dataset["y"]
        elif (dataset["X"] == None or dataset["y"] == None):
            data = openml.datasets.get_dataset(dataset["id"])
            X, y, _, _ = data.get_data(target=data.default_target_attribute)
            dataset["X"] = X
            dataset["y"] = y
        else:
            X, y = dataset["X"], dataset["y"]
        
        # print(dataset)
        
       
        
        if (len(y.unique()) == 2):
            print("Binary target - OK")
        elif  (len(y.unique()) > 2):
            print("Not binary target - WRONG")
            print("-"*100)
            print()
            continue

        
        if (len(X.columns) <= 10 and dataset["size"] == "small"):
            print(f"Low dimensionality ({len(X.columns)}) when assumed small (<=10) - OK")
        elif (len(X.columns) > 10 and dataset["size"] == "small"):
            print(f"High dimensionality ({len(X.columns)}) when assumed small (<=10) - WRONG")
            print("-"*100)
            print()
            continue
        
        if (len(X.columns) > 10 and dataset["size"] == "large"):
            print(f"High dimensionality ({len(X.columns)}) when assumed large (>10) - OK")
        elif (len(X.columns) <= 10 and dataset["size"] == "large"):
            print(f"Low dimensionality ({len(X.columns)}) when assumed large (>10) - WRONG")
            print("-"*100)
            print()
            continue
            
        if (X.isna().sum().sum() == 0):
            print("No missing values - OK")
        else:
            print("Missing values - WRONG")
            print("-"*100)
            print()
            continue

        # get number of non-numeric columns
        non_numeric = X.select_dtypes(exclude=np.number).columns
        if (len(non_numeric) == 0):
            print("No non-numeric columns - OK")
        else:
            print(f"Non-numeric columns: {non_numeric} - WRONG")
            print("-"*100)
            print()
            continue

        # check that there are more rows than columns
        if (X.shape[0] > X.shape[1]):
            if (X.shape[0] > X.shape[1]):
                print(f"More rows ({X.shape[0]}) than columns ({X.shape[1]}) - OK")
            else:
                print(f"More  columns ({X.shape[1]}) than rows ({X.shape[0]}) - WRONG")
                print("-"*100)
                print()
                continue
        
        print(f"Dataset {name} is OK")
        print("-"*100)
        print()
        failures -= 1
    if failures == 0:
        print("All datasets are valid")
    else:
        print(f"{failures} datasets are invalid")

def dataset_preprocessing(X, y, collinear_threshold=0.9):
    # Drop columns with only one unique value
    X = X.loc[:, X.apply(pd.Series.nunique) != 1]
    print(f"Columns with only one unique value were dropped: {set(X.columns) - set(X.loc[:, X.apply(pd.Series.nunique) != 1].columns)}")

    # Fill missing values with mean
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Drop collinear columns
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > collinear_threshold)]
    print(f"Columns to drop: {to_drop} because of collinearity")
    X = X.drop(columns=to_drop)

    # encode target variable as 0 and 1
    y = y.map({y.unique()[0]: 0, y.unique()[1]: 1})

    # change dtype to int
    y = y.astype(int)

    return X, y

def preprocess_datasets(datasets, collinear_threshold=0.9):
    for name, dataset in datasets.items():
        print(f"Preprocessing dataset {name}")
        if ((type(dataset['X'])==pd.DataFrame) and (type(dataset['y'])==pd.Series)):
            X, y = dataset["X"], dataset["y"]
        elif (dataset["X"] == None or dataset["y"] == None):
            data = openml.datasets.get_dataset(dataset["id"])
            X, y, _, _ = data.get_data(target=data.default_target_attribute)
        else:
            X, y = dataset["X"], dataset["y"]
        
        
        dataset['X'], dataset['y'] = dataset_preprocessing(X, y, collinear_threshold)
        print()
    return datasets


def performance_test(X, y, models, model_name, batch_size = 8, test_size=0.2):
    bal_acc_scores = []
    iteration_scores = []
    training_times = []
    iteration_betas = []
    n_iter = len(models)
    for i, model in tqdm(enumerate(models), total=len(models)):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i, shuffle=True
        )
        time_Start = time.time()
        if model_name in ["SGD", "ADAM", "IWLS"]:
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)
        time_End = time.time()
        training_times.append(time_End - time_Start)
        y_pred = model.predict(X_test)
        score = bal_acc(y_test, y_pred)
        bal_acc_scores.append(score)
        if model_name in ["SGD", "ADAM", "IWLS"]:
            iteration_scores.append(copy.deepcopy(model.cost_iterations))
            betas = [list(beta) for beta in model.beta_iterations]
            iteration_betas.append(copy.deepcopy(betas))
        # print(model.cost_iterations)
    return { 
        "mean_bal_acc": np.mean(bal_acc_scores),
        "std_dev_bal_acc": np.std(bal_acc_scores),
        "min": np.min(bal_acc_scores),
        "max": np.max(bal_acc_scores),
        "accuracies": bal_acc_scores,
        "model_name": model_name,
        "n_iter": n_iter,
        "test_size": test_size,
        "train_costs": iteration_scores,
        "training_times": training_times,
        "avg_training_time": np.mean(training_times),
        "std_dev_training_time": np.std(training_times),
        "min_training_time": np.min(training_times),
        "max_training_time": np.max(training_times),
        "iteration_betas": iteration_betas
    }