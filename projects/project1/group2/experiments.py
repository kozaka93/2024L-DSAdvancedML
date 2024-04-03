from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from load_data import load_data


# ----------------------------------------- Task 3.2 -----------------------------------------
def calculate_balanced_accuracies(classifiers, iterations: int = 10, files_dir: str = './data/') -> pd.DataFrame:
    balanced_accuracy_dict = {}

    for file_name in os.listdir(files_dir):
        dataset_name = os.path.splitext(file_name)[0]
        X, y = load_data(os.path.join(files_dir, file_name))
        for classifier_name, classifier in classifiers.items():
            balanced_accuracy_dataset = []
            for i in range(iterations):
                classifier_instance = classifier(patience=50)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                classifier_instance.fit(X_train, y_train)
                y_pred = classifier_instance.predict(X_test)
                balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                balanced_accuracy_dataset.append(balanced_accuracy)
            if dataset_name not in balanced_accuracy_dict:
                balanced_accuracy_dict[dataset_name] = {}
            balanced_accuracy_dict[dataset_name][classifier_name] = balanced_accuracy_dataset

    balanced_accuracy_df = pd.concat({key: pd.DataFrame(value).T for key, value in balanced_accuracy_dict.items()}, axis=0)
    balanced_accuracy_df.reset_index(inplace=True)
    balanced_accuracy_df.columns = ['dataset', 'method'] + [f'iteration_{i+1}' for i in range(iterations)]
    balanced_accuracy_df.to_csv(f'balanced_accuracy_overall.csv', index=False)
    return balanced_accuracy_df


def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    mean_df = df.mean().rename('mean balanced accuracy')
    std_df = df.std().rename('standard deviation')
    stats_df = pd.concat([mean_df, std_df], axis=1)
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'index': 'dataset'}, inplace=True)
    return stats_df


# ----------------------------------------- Task 3.3 -----------------------------------------
def calculate_log_likelihoods(X, y, file_name, classifiers):
    log_likelihoods = pd.DataFrame()
    for classifier_name, classifier in classifiers.items():
        classifier_instance = classifier(early_stopping=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        classifier_instance.fit(X_train, y_train)
        log_losses = classifier_instance.train_losses
        log_likelihoods[classifier_name] = [-x * len(X_train) for x in log_losses]

    log_likelihoods.to_csv(f'loglik_{os.path.splitext(file_name)[0]}.csv', index=False)
    return log_likelihoods


# ----------------------------------------- Task 3.4 & 3.5 -----------------------------------------
def compare_classifiers(X: pd.DataFrame, y: pd.Series,
                        classifiers, dataset_name, iterations: int = 10, classifier_instance=False) -> pd.DataFrame:
    classifiers_balanced_accuracies = {}
    for classifier_name, classifier in classifiers.items():
        balanced_accuracies = []
        if not classifier_instance:
            classifier = classifier()
        for _ in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        classifiers_balanced_accuracies[classifier_name] = balanced_accuracies

    classifiers_balanced_accuracies_df = pd.DataFrame(classifiers_balanced_accuracies)
    classifiers_balanced_accuracies_df.to_csv(f'compared_methods_{dataset_name}.csv', index=False)
    return classifiers_balanced_accuracies_df
