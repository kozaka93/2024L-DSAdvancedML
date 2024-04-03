import pandas as pd
import seaborn
from matplotlib import pyplot as plt

from library_implementations import (
    DecisionTreeLIB,
    LinearDiscriminantAnalysisLIB,
    LogisticRegressionLIB,
    QuadraticDiscriminantAnalysisLIB,
    RandomForestLIB,
)
from logistic_adam import LogisticRegressionADAM
from logistic_iwls import LogisticRegressionIWLS
from logistic_sgd import LogisticRegressionSGD

seaborn.set_theme()


def test_algorithms(
    X,
    y,
    interactions=False,
    repetitions=5,
    limit_iwls_iterations=None,
    extended=False,
    output=True,
    dataset_name="test",
) -> list:
    scores = []
    model_labels = ["logistic lib", "IWLS", "SGD", "Adam"]
    if extended:
        model_labels += ["LDA", "QDA", "decision tree", "random forest"]
    model_count = 4 if not extended else 8

    model = LogisticRegressionLIB()
    scores.append(model.test_efficiency(X, y, repetition_count=repetitions))

    model_iwls = LogisticRegressionIWLS(
        max_iter=limit_iwls_iterations if limit_iwls_iterations else 500
    )  # need to limit it because it is unstable for some datasets (maybe tol parameter increase will fix it)
    scores.append(model_iwls.test_efficiency(X, y, repetition_count=repetitions))

    model_sgd = LogisticRegressionSGD()
    scores.append(model_sgd.test_efficiency(X, y, repetition_count=repetitions))

    model_adam = LogisticRegressionADAM()
    scores.append(model_adam.test_efficiency(X, y, repetition_count=repetitions))

    if extended:
        model_lda = LinearDiscriminantAnalysisLIB()
        scores.append(model_lda.test_efficiency(X, y, repetition_count=repetitions))

        model_qda = QuadraticDiscriminantAnalysisLIB()
        scores.append(model_qda.test_efficiency(X, y, repetition_count=repetitions))

        model_decision_tree = DecisionTreeLIB()
        scores.append(
            model_decision_tree.test_efficiency(X, y, repetition_count=repetitions)
        )

        model_random_forest = RandomForestLIB()
        scores.append(
            model_random_forest.test_efficiency(X, y, repetition_count=repetitions)
        )

    scores_interactions = []
    if interactions:
        scores_interactions.append(
            model.test_efficiency(
                X, y, interactions=interactions, repetition_count=repetitions
            )
        )
        scores_interactions.append(
            model_iwls.test_efficiency(
                X, y, interactions=interactions, repetition_count=repetitions
            )
        )
        scores_interactions.append(
            model_sgd.test_efficiency(
                X, y, interactions=interactions, repetition_count=repetitions
            )
        )
        scores_interactions.append(
            model_adam.test_efficiency(
                X, y, interactions=interactions, repetition_count=repetitions
            )
        )

        if extended:
            scores_interactions.append(
                model_lda.test_efficiency(
                    X, y, interactions=interactions, repetition_count=repetitions
                )
            )
            scores_interactions.append(
                model_qda.test_efficiency(
                    X, y, interactions=interactions, repetition_count=repetitions
                )
            )
            scores_interactions.append(
                model_decision_tree.test_efficiency(
                    X, y, interactions=interactions, repetition_count=repetitions
                )
            )
            scores_interactions.append(
                model_random_forest.test_efficiency(
                    X, y, interactions=interactions, repetition_count=repetitions
                )
            )
        if output:
            print(f"Results without interactions: {scores}")
            print(f"Results with interactions: {scores_interactions}")

            grouped_data = list(zip(scores, scores_interactions))
            df = pd.DataFrame(
                grouped_data, columns=["without interactions", "with interactions"]
            )
            df_stacked = pd.concat([df] * model_count, ignore_index=True)
            df_stacked["Stack"] = list(range(1, model_count + 1)) * len(df)
            df_plot = pd.melt(
                df_stacked, id_vars="Stack", var_name="List", value_name="Value"
            )
            seaborn.barplot(
                x="Stack",
                y="Value",
                hue="List",
                data=df_plot,
                palette="viridis",
            )
            plt.ylabel("Accuracy")
            plt.xlabel("Method")
            plt.title("Prediction accuracy by method")
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    else:
        if output:
            print(f"Results: {scores}")
            seaborn.barplot(
                y=scores,
                x=model_labels,
                palette="viridis",
            )
            plt.ylabel("Accuracy")
            plt.xlabel("Method")
            plt.title("Prediction accuracy by method")
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

    models = [model, model_iwls, model_sgd, model_adam]
    if extended:
        models += [model_lda, model_qda, model_decision_tree, model_random_forest]
    if output:
        filename = f"convergence/{dataset_name}.csv"
        plot_convergence(models[1:4], filename)
    print(scores)
    return models, scores, scores_interactions


def plot_convergence(models, filename):
    fig, ax = plt.subplots(1, 3)
    fig.text(0.04, 0.5, "Log-likelihood", va="center", rotation="vertical")
    fig.text(0.5, 0.02, "Iteration", ha="center")
    plt.suptitle("Log-likelihood function value by iteration")
    for i in range(len(models)):
        seaborn.lineplot(
            y=models[i].log_likelihood,
            x=range(1, len(models[i].log_likelihood) + 1),
            ax=ax[i],
            palette="viridis",
        )

    df = pd.DataFrame.from_dict(
        {
            "IWLS": models[0].log_likelihood,
            "SGD": models[1].log_likelihood,
            "Adam": models[2].log_likelihood,
        },
        orient="index",
    ).T
    df.to_csv(filename, index_label="Iteration")


def test_all_datasets(
    data: dict,
    repetitions=5,
    extended=False,
    filename=None,
    show_lib=True,
    interactions=False,
):
    limited_iwls = ["large2", "large3", "large4", "large5"]
    results = pd.DataFrame()
    for item in data.items():
        label, data = item
        print(label)

        X, y = data
        _, scores, scores_interactions = test_algorithms(
            X,
            y,
            repetitions=repetitions,
            extended=extended,
            # output=False,
            limit_iwls_iterations=30 if label in limited_iwls else None,
            dataset_name=label,
            interactions=interactions,
        )
        results[label] = scores + scores_interactions

    methods = ["logistic lib", "IWLS", "SGD", "Adam"]
    if extended:
        methods += ["LDA", "QDA", "decision tree", "random forest"]
    if interactions:
        methods += ["logistic lib+INT", "IWLS+INT", "SGD+INT", "Adam+INT"]

    results["Method"] = methods
    df = pd.DataFrame(results)
    df_plot = pd.melt(df, id_vars="Method", var_name="Column", value_name="Value")

    if not show_lib:
        df_plot = df_plot[df_plot["Method"] != "logistic lib"]
        df_plot = df_plot[df_plot["Method"] != "logistic lib+INT"]

    plt.figure(figsize=(12, 6))
    seaborn.barplot(
        x="Column",
        y="Value",
        hue="Method",
        data=df_plot,
        hue_order=(
            ["IWLS", "IWLS+INT", "SGD", "SGD+INT", "Adam", "Adam+INT"]
            if interactions
            else None
        ),
        # palette="viridis",
    )
    plt.ylabel("Accuracy")
    plt.ylabel("Dataset")
    plt.title("Algorithms accuracy on all datasets")
    plt.legend(title="Method", bbox_to_anchor=(1, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    plt.show()
    return results
