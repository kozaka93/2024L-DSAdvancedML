import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.4, style='darkgrid')
from lr import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from optimizers import SGD, Adam, IRLS
from sklearn.metrics import balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import time


def run_experiments(data_list, optimizer_classes, optimizer_params, include_interactions=False):
    results = []
    for i, data in enumerate(data_list):
        X, y = data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for split_index, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for optimizer_class in optimizer_classes:
                optimizer = optimizer_class(**optimizer_params.get(optimizer_class.__name__, {}))
                model = LogisticRegression(optimizer, early_stopping_rounds=5, epochs=500, include_interactions=include_interactions)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = balanced_accuracy_score(y_test, y_pred)
                results.append({'dataset': i+1, 'method': optimizer_class.__name__, 'balanced_accuracy': score, 'losses': model.losses, 'split': split_index})
    results_df = pd.DataFrame(results)
    return results_df

def plot_boxplots(results_df):
    for optimizer in results_df['method'].unique():
        optimizer_data = results_df[results_df['method'] == optimizer]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=optimizer_data, x='dataset', y='balanced_accuracy', color=sns.color_palette()[0])
        plt.title(f'Boxplot of Balanced Accuracy for {optimizer}')
        plt.xlabel('Dataset')
        plt.ylabel('Balanced Accuracy')
        plt.savefig(f'boxplot_{optimizer}.png')
        plt.show()

def plot_losses(data_list, optimizer_classes, optimizer_params):
    for optimizer_class in optimizer_classes:
        plt.figure(figsize=(10, 6))
        for i in range(len(data_list)):
            dataset = data_list[i]
            X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=101)
            optimizer = optimizer_class(**optimizer_params.get(optimizer_class.__name__, {}))
            model = LogisticRegression(optimizer, epochs=500)
            model.fit(X_train, y_train)
            plt.plot(model.losses, label='Dataset ' + str(i+1))
        plt.title(f'Model Losses with {type(optimizer).__name__} optimizer')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Log-likelihood')
        plt.savefig(f'losses_{type(optimizer).__name__}.png')
        plt.show()

def run_baseline(data_list, classifiers):
    results = []
    for i, data in enumerate(data_list):
        X, y = data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for split_index, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for classifier in classifiers:
                model = classifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = balanced_accuracy_score(y_test, y_pred)
                results.append({'dataset': i+1, 'method': type(model).__name__, 'balanced_accuracy': score, 'split': split_index})
    results_df = pd.DataFrame(results)
    return results_df

def plot_all(results_df):
    for dataset in results_df['dataset'].unique():
        df = results_df[results_df['dataset'] == dataset]
        plt.figure(figsize=(15, 6))
        sns.boxplot(data=df, x='method', y='balanced_accuracy', color=sns.color_palette()[0])
        plt.title(f'Boxplot of Balanced Accuracy for Dataset {dataset}')
        plt.xticks(ticks=range(len(df['method'].unique())), labels=['SGD', 'ADAM', 'IRLS', 'LDA', 'QDA', 'Decision Tree', 'Random Forest'])
        plt.xlabel('Method')
        plt.ylabel('Balanced Accuracy')
        plt.savefig(f'boxplot_dataset_{dataset}.png')
        plt.show()


def main():
    start_time = time.time()
    with open('data_list.pkl', 'rb') as f:
        data_list = pickle.load(f)

    optimizer_classes = [SGD, Adam, IRLS]
    optimizer_params = {
        'SGD': {'learning_rate': 0.0001},
        'Adam': {'learning_rate': 0.0001, 'beta1': 0.9, 'beta2': 0.999},
        'IRLS': {'tol': 0.0001}
    }


    print("task 3.2")
    results_df = run_experiments(data_list, optimizer_classes, optimizer_params)
    plot_boxplots(results_df)


    print("task 3.3")
    plot_losses(data_list, optimizer_classes, optimizer_params)
    df = results_df[results_df['split'] == 1]

    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        plt.figure(figsize=(15, 6))
        plt.title(f'Log-likelihood per Epoch - {method} optimizer')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        for index, row in method_df.iterrows():
            plt.plot(range(1, len(row['losses']) + 1), row['losses'], label=f'Dataset {row["dataset"]}')
        plt.legend()
        plt.savefig(f'losses_{method}_early_stopping.png')
        plt.show()


    print("task 3.4")
    classifiers = [LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, DecisionTreeClassifier, RandomForestClassifier]
    baseline_results_df = run_baseline(data_list, classifiers)
    final_results_df = pd.concat([results_df, baseline_results_df])
    plot_all(final_results_df)


    print("task 3.5")
    data_list_interactions = data_list
    results_interactions = run_experiments(data_list_interactions, optimizer_classes, optimizer_params, include_interactions=True)
    small_results_df = results_df[results_df['dataset'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])]
    small_results_df['interactions'] = False
    results_interactions['interactions'] = True
    final_results_df = pd.concat([small_results_df, results_interactions])

    for dataset in final_results_df['dataset'].unique():
        df = final_results_df[final_results_df['dataset'] == dataset]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='method', y='balanced_accuracy', hue='interactions')
        plt.title(f'Performance of models with and without interactions for Dataset {dataset}')
        plt.xlabel('Method')
        plt.ylabel('Balanced Accuracy')
        plt.legend(title='Interactions')
        plt.savefig(f'boxplot_interactions_dataset_{dataset}.png')
        plt.show()


    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_minutes = execution_time_seconds / 60

    print("Execution time:", execution_time_minutes, "minutes")

if __name__ == '__main__':
    main()