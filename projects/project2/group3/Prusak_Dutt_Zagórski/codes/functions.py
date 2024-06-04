import numpy as np
import pandas as pd
import itertools
from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    RFECV,
    SelectFpr,
    SelectFdr,
    SelectFwe,
)
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score
import time
import xgboost as xgb
from sklearn.decomposition import PCA


RESULTS_COLUMNS = [
    "score",
    "numberOfTruePositives",
    "accuracy",
    "precision",
    "numberOfFeatures",
    "model",
    "model_parameters",
    "feature_selector",
    "selector_parameters",
    "scaler",
    "scaler_parameters",
    "feature_generator",
    "feature_generator_parameters",
]

# %%
POSITIVE_CLASS_LABEL = 1

Y_LIMIT_PERCENT = 0.8

# %% [markdown]
# # Functions

# %% [markdown]
# ## Generate parameters
#


# %%
def namestr(obj, namespace) -> str:
    """Get name of a variable as a string"""
    return [name for name in namespace if namespace[name] is obj]


# %%
def generateParameters(arrays, namespace):
    """Takes any number of arrays and returns an array of
    dictionaries with keys as array names and values, all possible combinations

    namespace in general should be globals() or locals()
    """

    allCombinations = list(itertools.product(*arrays))

    allCombinations = [list(elem) for elem in allCombinations]

    arrayNames = []

    for array in arrays:
        arrayNames.append(namestr(array, namespace)[0])

    return [dict(zip(arrayNames, value)) for value in allCombinations]


# %% [markdown]
# - example usage, create arrays with parameters (their names should be the same as desired parameter names), pass them in an array to generateParamters function

# %%
arr1 = [1, 2, 7]
arr2 = [3, 4, 8]
generateParameters([arr1, arr2], globals())

# %% [markdown]
# ## Model helpers

# %% [markdown]
# - definition of available models and feature selection methods
#
# - getters for models and feature selection methods


# %%
class ModelType(Enum):
    """Available classifiers"""

    LDA = 0

    QDA = 1

    DecisionTree = 2

    KNN = 3

    SVC = 4

    GradientBoosting = 5

    HistGradientBoosting = 6

    MLPClassifier = 7

    ADABoost = 8

    Voting = 9

    XGBoost = 10

    ExtraTrees = 11


# %%
def getModel(modelType, arguments):
    """Returns a classifier that can fit() and predict()"""

    match modelType:

        case ModelType.LDA:

            return LinearDiscriminantAnalysis(**arguments)

        case ModelType.QDA:

            return QuadraticDiscriminantAnalysis(**arguments)

        case ModelType.DecisionTree:

            return DecisionTreeClassifier(**arguments)

        case ModelType.KNN:

            return KNeighborsClassifier(**arguments)

        case ModelType.SVC:

            return SVC(**arguments)

        case ModelType.GradientBoosting:

            return GradientBoostingClassifier(**arguments)

        case ModelType.HistGradientBoosting:

            return HistGradientBoostingClassifier(**arguments)

        case ModelType.MLPClassifier:

            return MLPClassifier(**arguments)

        case ModelType.ADABoost:

            return AdaBoostClassifier(**arguments)

        case ModelType.Voting:

            return VotingClassifier(**arguments)

        case ModelType.XGBoost:

            clf = xgb.XGBClassifier()
            clf.set_params(**arguments)
            return clf

        case ModelType.ExtraTrees:

            return ExtraTreesClassifier(**arguments)


# %%
class FeatureSelectorType(Enum):
    """Available Feature Selectors that can fit_transform() a dataset"""

    NoFeatureSelection = 0

    KBest = 1

    RFE = 2

    RFECV = 3

    FPR = 4

    FDR = 5

    FWE = 6

    PCA = 7


# %%
class NoFeatureSelection:
    """Wrapper class for no feature selection, exposes
    fit_transform method that returns unchanged X
    """

    def fit_transform(self, X, y):

        return X


# %%
def getFeatureSelector(selectorType, arguments):
    """Returns a feature selector that can fit_transform()"""

    match selectorType:

        case FeatureSelectorType.NoFeatureSelection:

            return NoFeatureSelection()

        case FeatureSelectorType.KBest:

            return SelectKBest(**arguments)

        case FeatureSelectorType.RFE:

            return RFE(**arguments)

        case FeatureSelectorType.RFECV:

            return RFECV(**arguments)

        case FeatureSelectorType.FPR:

            return SelectFpr(**arguments)

        case FeatureSelectorType.FDR:

            return SelectFdr(**arguments)

        case FeatureSelectorType.FWE:

            return SelectFwe(**arguments)

        case FeatureSelectorType.PCA:

            return PCA(**arguments)


class NoScaling:

    def fit_transform(self, X, y):
        return X


class Scaler(Enum):

    NoScaling = 0

    Standard = 1

    Robust = 2


def getScaler(scalerType, arguments):

    match scalerType:

        case Scaler.NoScaling:

            return NoScaling()

        case Scaler.Standard:

            return StandardScaler(**arguments)

        case Scaler.Robust:

            return RobustScaler(**arguments)


class FeatureGenerator(Enum):

    NoFeatureGeneration = 0

    Polynomial = 1


class NoFeatureGeneration:
    def fit_transform(self, X):
        return X


def getFeatureGenerator(generatorType, arguments):
    match generatorType:
        case FeatureGenerator.NoFeatureGeneration:

            return NoFeatureGeneration()

        case FeatureGenerator.Polynomial:

            return PolynomialFeatures(**arguments)


# %% [markdown]
# ## Experiment process


# %%
def getTotalNoOfExperiments(models, featureSelectors):
    modelCount = 0
    selectorCount = 0
    for model in models:
        modelCount += len(model["parameters"])
    for featureSelector in featureSelectors:
        selectorCount += len(featureSelector["parameters"])
    return modelCount * selectorCount


def getTotalNoOfExperimentsWithScalersAndGenerators(
    models, featureSelectors, scalers, generators
):
    modelCount = 0
    selectorCount = 0
    scalerCount = 0
    generatorCount = 0

    for model in scalers:
        scalerCount += len(model["parameters"])
    for model in generators:
        generatorCount += len(model["parameters"])
    for model in models:
        modelCount += len(model["parameters"])
    for featureSelector in featureSelectors:
        selectorCount += len(featureSelector["parameters"])
    return modelCount * selectorCount * scalerCount * generatorCount


# %%
def getScore(y_true, y_pred, featuresUsed):
    """Get score based on y_true, y_pred and number of feature used

    Scoring function, based on which, the best model is selected.
    Score is calculated according to task description: +10 points for each
    correctly classified positive class, -200 points for each feature used
    """

    score = 0

    correct = 0

    for i, y in enumerate(y_true):
        if y == POSITIVE_CLASS_LABEL and y == y_pred[i]:
            correct += 1

    score = 10 * correct - 200 * featuresUsed
    return correct, score


def getScoreLimited(y_true, y_pred, featuresUsed, limit):
    """Get score based on y_true, y_pred and number of feature used

    Scoring function, based on which, the best model is selected.
    Score is calculated according to task description: +10 points for each
    correctly classified positive class, -200 points for each feature used
    """

    amounToTake = int(limit * len(y_true[y_true == 1]))

    df = pd.DataFrame(y_pred[:, 1], columns=["result"])
    sortedDf = df.sort_values(by="result", ascending=False).head(amounToTake)

    y_predicted = np.zeros_like(y_true)

    y_predicted[sortedDf.index] = 1

    score = 0

    correct = 0

    for i, y in enumerate(y_true):
        if y == POSITIVE_CLASS_LABEL and y == y_predicted[i]:
            correct += 1

    score = 10 * correct - 200 * featuresUsed
    return correct, score / (amounToTake * 10 - 400)


# %%
def performExperiment(X_train, y_train, X_test, y_test, model, limit, getLimitedScore):
    """Returns a score for given model and provided data"""

    model.fit(X_train, y_train)

    numberOfFeatures = len(X_train[0])

    correct = 0
    score = 0
    accuracy = 0
    precision = 0

    if getLimitedScore:
        y_pred = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))
        precision = precision_score(y_test, np.argmax(y_pred, axis=1))
        correct, score = getScoreLimited(y_test, y_pred, numberOfFeatures, limit)
    else:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        correct, score = getScore(y_test, y_pred, numberOfFeatures)
    # finalResult = getScore(y_test, y_pred, model["classification"].n_features_in_)

    #

    # Take only 1000 of the highest class 1 probabilities

    # TODO: Turn it into a dataframe and keep track of the indexes

    # finalResult = np.sort(result[:, 1])[::-1][:1000]

    return correct, score, accuracy, numberOfFeatures, precision


# %%
def conductExperimentsWithScalersAndGenerators(
    models,
    featureSelectors,
    X_orig,
    y_orig,
    scalers,
    featureGenerators,
    limit=0.8,
    getLimitedScore=False,
):
    """Collects score results for all provided models and feature selectors on given dataset

    Parameters:

    models: array in a format: [{"model":"model name","parameters":[{paramters object}]}]

    featureSelectors: array in a format: [{"model":"model name","parameters":[{paramters object}]}]


    """
    results = []
    totalNumberOfExperiments = getTotalNoOfExperimentsWithScalersAndGenerators(
        models, featureSelectors, scalers, featureGenerators
    )
    experimentCount = 0
    for featureSelector in featureSelectors:
        for featureSelectorParameters in featureSelector["parameters"]:

            try:
                startFeatureSelector = time.time()
                selector = getFeatureSelector(
                    featureSelector["model"], featureSelectorParameters
                )

                X_new = selector.fit_transform(X_orig, y_orig)

                endFeatureSelector = time.time()
                for scaler in scalers:
                    for scalerParameters in scaler["parameters"]:

                        scalerInstance = getScaler(scaler["model"], scalerParameters)
                        X_scaled = scalerInstance.fit_transform(X_new, y_orig)
                        for featureGenerator in featureGenerators:
                            for featureGeneratorParameters in featureGenerator[
                                "parameters"
                            ]:
                                featureGeneratorInstance = getFeatureGenerator(
                                    featureGenerator["model"],
                                    featureGeneratorParameters,
                                )
                                X_scaled = featureGeneratorInstance.fit_transform(
                                    X_scaled
                                )

                                if len(X_scaled[0]) > 1:
                                    (
                                        X_split_train,
                                        X_split_test,
                                        y_split_train,
                                        y_split_test,
                                    ) = train_test_split(
                                        X_scaled,
                                        y_orig,
                                        test_size=0.33,
                                        random_state=42,
                                    )
                                    # print(len(y_split_test[y_split_test==1]))
                                    for model in models:
                                        for modelParameters in model["parameters"]:

                                            # clf = Pipeline(
                                            #     [
                                            #         ("scaling", StandardScaler()),
                                            #         (
                                            #             "feature_selection",
                                            #             getFeatureSelector(
                                            #                 featureSelector["model"], featureSelectorParameters
                                            #             ),
                                            #         ),
                                            #         (
                                            #             "classification",
                                            #             getModel(model["model"], modelParameters),
                                            #         ),
                                            #     ]
                                            # )
                                            try:
                                                # X_new = selector.fit_transform(X_orig, y_orig)
                                                startModel = time.time()
                                                (
                                                    correct,
                                                    score,
                                                    accuracy,
                                                    numberOfFeatures,
                                                    precision,
                                                ) = performExperiment(
                                                    X_train=X_split_train,
                                                    y_train=y_split_train,
                                                    X_test=X_split_test,
                                                    y_test=y_split_test,
                                                    model=getModel(
                                                        model["model"], modelParameters
                                                    ),
                                                    limit=limit,
                                                    getLimitedScore=getLimitedScore,
                                                )

                                                results.append(
                                                    [
                                                        score,
                                                        correct,
                                                        accuracy,
                                                        precision,
                                                        numberOfFeatures,
                                                        model["model"].name,
                                                        modelParameters,
                                                        featureSelector["model"].name,
                                                        featureSelectorParameters,
                                                        scaler["model"].name,
                                                        scalerParameters,
                                                        featureGenerator["model"].name,
                                                        featureGeneratorParameters,
                                                    ]
                                                )
                                                endModel = time.time()
                                                experimentCount += 1
                                                print(
                                                    "Performed Experiment",
                                                    str(experimentCount)
                                                    + "/"
                                                    + str(totalNumberOfExperiments)
                                                    + "(approx)",
                                                    "took (s):",
                                                    "model:",
                                                    str(
                                                        round(endModel - startModel, 2)
                                                    ),
                                                    "selector",
                                                    str(
                                                        round(
                                                            endFeatureSelector
                                                            - startFeatureSelector,
                                                            2,
                                                        )
                                                    ),
                                                    "with:",
                                                    featureSelector["model"],
                                                    featureSelectorParameters,
                                                    model["model"],
                                                    modelParameters,
                                                    scaler["model"],
                                                    scalerParameters,
                                                    featureGenerator["model"],
                                                    featureGeneratorParameters,
                                                )
                                            except Exception as e:
                                                print(
                                                    "!!!Experiment failed for:",
                                                    featureSelector["model"],
                                                    featureSelectorParameters,
                                                    model["model"],
                                                    modelParameters,
                                                    scaler["model"],
                                                    scalerParameters,
                                                    featureGenerator["model"],
                                                    featureGeneratorParameters,
                                                    str(e),
                                                )

                                else:
                                    print(
                                        "!!!",
                                        featureSelector["model"],
                                        "produced 1 or fewer features with parameters:",
                                        featureSelectorParameters,
                                    )
            except Exception as e:
                print(
                    "!!!Experiment failed for:",
                    featureSelector["model"],
                    featureSelectorParameters,
                    model["model"],
                    modelParameters,
                    scaler["model"],
                    scalerParameters,
                    featureGenerator["model"],
                    featureGeneratorParameters,
                    str(e),
                )
    return results


def conductExperiments(
    models,
    featureSelectors,
    X_orig,
    y_orig,
    limit=0.8,
    getLimitedScore=False,
):
    """Collects score results for all provided models and feature selectors on given dataset

    Parameters:

    models: array in a format: [{"model":"model name","parameters":[{paramters object}]}]

    featureSelectors: array in a format: [{"model":"model name","parameters":[{paramters object}]}]


    """
    results = []
    totalNumberOfExperiments = getTotalNoOfExperiments(models, featureSelectors)
    experimentCount = 0

    # X_scaled = scaler.fit_transform(X_orig, y_orig)
    for featureSelector in featureSelectors:
        for featureSelectorParameters in featureSelector["parameters"]:
            try:
                startFeatureSelector = time.time()
                selector = getFeatureSelector(
                    featureSelector["model"], featureSelectorParameters
                )

                X_new = selector.fit_transform(X_orig, y_orig)

                endFeatureSelector = time.time()

                if len(X_new[0]) > 1:
                    X_split_train, X_split_test, y_split_train, y_split_test = (
                        train_test_split(X_new, y_orig, test_size=0.33, random_state=42)
                    )
                    # print(len(y_split_test[y_split_test==1]))
                    for model in models:
                        for modelParameters in model["parameters"]:

                            # clf = Pipeline(
                            #     [
                            #         ("scaling", StandardScaler()),
                            #         (
                            #             "feature_selection",
                            #             getFeatureSelector(
                            #                 featureSelector["model"], featureSelectorParameters
                            #             ),
                            #         ),
                            #         (
                            #             "classification",
                            #             getModel(model["model"], modelParameters),
                            #         ),
                            #     ]
                            # )
                            try:
                                # X_new = selector.fit_transform(X_orig, y_orig)
                                startModel = time.time()
                                (
                                    correct,
                                    score,
                                    accuracy,
                                    numberOfFeatures,
                                    precision,
                                ) = performExperiment(
                                    X_train=X_split_train,
                                    y_train=y_split_train,
                                    X_test=X_split_test,
                                    y_test=y_split_test,
                                    model=getModel(model["model"], modelParameters),
                                    limit=limit,
                                    getLimitedScore=getLimitedScore,
                                )

                                results.append(
                                    [
                                        score,
                                        correct,
                                        accuracy,
                                        precision,
                                        numberOfFeatures,
                                        model["model"].name,
                                        modelParameters,
                                        featureSelector["model"].name,
                                        featureSelectorParameters,
                                    ]
                                )
                                endModel = time.time()
                                experimentCount += 1
                                print(
                                    "Performed Experiment",
                                    str(experimentCount)
                                    + "/"
                                    + str(totalNumberOfExperiments)
                                    + "(approx)",
                                    "took (s):",
                                    "model:",
                                    str(round(endModel - startModel, 2)),
                                    "selector",
                                    str(
                                        round(
                                            endFeatureSelector - startFeatureSelector,
                                            2,
                                        )
                                    ),
                                    "with:",
                                    featureSelector["model"],
                                    featureSelectorParameters,
                                    model["model"],
                                    modelParameters,
                                )
                            except Exception as e:
                                print(
                                    "!!!Experiment failed for:",
                                    featureSelector["model"],
                                    featureSelectorParameters,
                                    model["model"],
                                    modelParameters,
                                    str(e),
                                )

                else:
                    print(
                        "!!!",
                        featureSelector["model"],
                        "produced 1 or fewer features with parameters:",
                        featureSelectorParameters,
                    )
            except Exception as e:
                print(
                    "!!!Experiment failed for:",
                    featureSelector["model"],
                    featureSelectorParameters,
                    model["model"],
                    modelParameters,
                    str(e),
                )
    return results


# %% [markdown]
# ## Results processing


# %%
def flattenArray(array):
    """Convert list of lists to a flat list"""
    return [element for list in array for element in list]


# %%
def processParameter(parameter):
    """Returns proper name of a paramter

    Some paramters (i.e. functions) need to be processed
    by extracting their name as str
    """
    if hasattr(parameter, "__name__"):
        return parameter.__name__
    else:
        return parameter


# %%
def getParameterName(column, parameter):
    """Helper function returns a new name for parameter to avoid collisions"""
    return column + "_" + parameter


# %%
def extractParamterInformation(df, parameterColumnName, parameterName):
    """
    Extract information about certain parameter and include it with
    the dataframe as a separate column in a format parameterColumnName+"_"+parameterName
    to avoid conflicts between model parameters and feature selector parameters
    MODIFIES THE PROVIDED DF
    """

    df[getParameterName(parameterColumnName, parameterName)] = df.apply(
        lambda x: (
            processParameter(x[parameterColumnName][parameterName])
            if parameterName in x[parameterColumnName].keys()
            else pd.NA
        ),
        axis=1,
    )


# %%
def extractParameterResults(resultsDf, models, featureSelectors):
    """Extract parameters from results for further results processing

    Prepares resulting parameters for further processing and appends
    resultsDf with columns corresponding to said parameters
    returns new dataframe and parameters
    """

    parameterName = []
    for featureSelector in featureSelectors:
        parameterName.append(list(featureSelector["parameters"][0].keys()))
    parameterName = list(set(flattenArray(parameterName)))

    parameterColumnName = ["selector_parameters"]

    parameters = generateParameters([parameterName, parameterColumnName], locals())

    parameterName = []
    for model in models:
        parameterName.append(list(model["parameters"][0].keys()))
    parameterName = list(set(flattenArray(parameterName)))
    parameterColumnName = ["model_parameters"]

    parameters = parameters + generateParameters(
        [parameterName, parameterColumnName], locals()
    )
    newDf = resultsDf.copy()
    for parameter in parameters:
        extractParamterInformation(newDf, **parameter)
    return newDf, parameters


def extractParameterResultsArr(resultsDf, modelsArr, parameterColumnNameArr):
    """Extract parameters from results for further results processing

    Prepares resulting parameters for further processing and appends
    resultsDf with columns corresponding to said parameters
    returns new dataframe and parameters
    """
    parameters = []
    for i, parameterColumnName in enumerate(parameterColumnNameArr):
        parameterColumnName = [parameterColumnName]

        parameterName = []
        for model in modelsArr[i]:
            parameterName.append(list(model["parameters"][0].keys()))
        parameterName = list(set(flattenArray(parameterName)))
        parameters = parameters + generateParameters(
            [parameterName, parameterColumnName], locals()
        )
    newDf = resultsDf.copy()
    for parameter in parameters:
        extractParamterInformation(newDf, **parameter)
    return newDf, parameters


# %%
def drawParameterResultsBoxplot(resultsDf, parameters):
    """Draw boxplots of how different parameters affect the score"""
    for parameter in parameters:
        grouped = resultsDf.groupby("feature_selector")
        for name, group in grouped:
            groupOnlyNa = group[group.isnull().any(axis=1)]
            group = group.dropna()

            if group.shape[0] > 0:
                plt.title(
                    "Score by " + str(parameter["parameterName"]) + " for " + name
                )
                sns.boxplot(
                    data=group.reset_index(),
                    x="model",
                    y="score",
                    hue=getParameterName(
                        parameter["parameterColumnName"], parameter["parameterName"]
                    ),
                )
                plt.show()
            if groupOnlyNa.shape[0] > 0:
                plt.title("Score for " + name)
                sns.boxplot(
                    data=groupOnlyNa.reset_index(),
                    x="model",
                    y="score",
                )
                plt.show()


# %%
def drawParameterResultsBarplot(resultsDf, parameters):
    """Draw barplots of how different parameters affect the score"""
    for parameter in parameters:

        if (
            parameter["parameterColumnName"]
            == RESULTS_COLUMNS[len(RESULTS_COLUMNS) - 1]
        ):
            plt.title("Score for parameter " + parameter["parameterName"])
            sns.barplot(
                data=resultsDf,
                x="model",
                y="score",
                hue=resultsDf[
                    [
                        "feature_selector",
                        getParameterName(
                            parameter["parameterColumnName"], parameter["parameterName"]
                        ),
                    ]
                ].apply(tuple, axis=1),
            )

            plt.tight_layout()
            plt.show()

            plt.title("Accuracy for parameter " + parameter["parameterName"])
            sns.barplot(
                data=resultsDf,
                x="model",
                y="accuracy",
                hue=resultsDf[
                    [
                        "feature_selector",
                        getParameterName(
                            parameter["parameterColumnName"], parameter["parameterName"]
                        ),
                    ]
                ].apply(tuple, axis=1),
            )

            plt.tight_layout()
            plt.show()

            plt.title("Precision for parameter " + parameter["parameterName"])
            sns.barplot(
                data=resultsDf,
                x="model",
                y="precision",
                hue=resultsDf[
                    [
                        "feature_selector",
                        getParameterName(
                            parameter["parameterColumnName"], parameter["parameterName"]
                        ),
                    ]
                ].apply(tuple, axis=1),
            )

            plt.tight_layout()
            plt.show()

            plt.title("True Positives for parameter " + parameter["parameterName"])
            sns.barplot(
                data=resultsDf,
                x="model",
                y="numberOfTruePositives",
                hue=resultsDf[
                    [
                        "feature_selector",
                        getParameterName(
                            parameter["parameterColumnName"], parameter["parameterName"]
                        ),
                    ]
                ].apply(tuple, axis=1),
            )

            plt.tight_layout()
            plt.show()

            plt.title("Number of features for parameter " + parameter["parameterName"])
            sns.barplot(
                data=resultsDf,
                x="feature_selector",
                y="numberOfFeatures",
                hue=resultsDf[
                    [
                        "feature_selector",
                        getParameterName(
                            parameter["parameterColumnName"], parameter["parameterName"]
                        ),
                    ]
                ].apply(tuple, axis=1),
            )

            plt.tight_layout()
            plt.show()
        else:
            plt.title("Score for parameter " + parameter["parameterName"])
            sns.barplot(
                data=resultsDf,
                x="feature_selector",
                y="score",
                hue=resultsDf[
                    [
                        "model",
                        getParameterName(
                            parameter["parameterColumnName"], parameter["parameterName"]
                        ),
                    ]
                ].apply(tuple, axis=1),
            )

            plt.tight_layout()
            plt.show()

            plt.title("Accuracy for parameter " + parameter["parameterName"])
            sns.barplot(
                data=resultsDf,
                x="feature_selector",
                y="accuracy",
                hue=resultsDf[
                    [
                        "model",
                        getParameterName(
                            parameter["parameterColumnName"], parameter["parameterName"]
                        ),
                    ]
                ].apply(tuple, axis=1),
            )

            plt.tight_layout()
            plt.show()

            plt.title("Precision for parameter " + parameter["parameterName"])
            sns.barplot(
                data=resultsDf,
                x="feature_selector",
                y="precision",
                hue=resultsDf[
                    [
                        "model",
                        getParameterName(
                            parameter["parameterColumnName"], parameter["parameterName"]
                        ),
                    ]
                ].apply(tuple, axis=1),
            )

            plt.tight_layout()
            plt.show()

            plt.title("True Positives for parameter " + parameter["parameterName"])
            sns.barplot(
                data=resultsDf,
                x="feature_selector",
                y="numberOfTruePositives",
                hue=resultsDf[
                    [
                        "model",
                        getParameterName(
                            parameter["parameterColumnName"], parameter["parameterName"]
                        ),
                    ]
                ].apply(tuple, axis=1),
            )

            plt.tight_layout()
            plt.show()


# %%
def drawResultsPerNumberOfFeatures(resultsDf):
    """Draws barplots with model type on x axis, accuracy/score on y axis, hue by number of features used"""
    plt.title("Accuracy by number of features for each model")
    sns.barplot(
        data=resultsDf,
        x="model",
        y="accuracy",
        hue="numberOfFeatures",
    )

    plt.tight_layout()
    plt.show()

    plt.title("Score by number of features for each model")
    sns.barplot(
        data=resultsDf,
        x="model",
        y="score",
        hue="numberOfFeatures",
    )

    plt.tight_layout()
    plt.show()

    plt.title("Correct predictions by number of features for each model")
    sns.barplot(
        data=resultsDf,
        x="model",
        y="numberOfTruePositives",
        hue="numberOfFeatures",
    )

    plt.tight_layout()
    plt.show()

    plt.title("Precision by number of features for each model")
    sns.barplot(
        data=resultsDf,
        x="model",
        y="precision",
        hue="numberOfFeatures",
    )

    plt.tight_layout()
    plt.show()


# %%
def filterDataframeByBestResultsForFeatureSelectors(resultsDf):
    """Extracts best results for each parameter for final comparison of feature selectors
    Returns a dataframe containing only the best results
    """
    filteredDf = resultsDf.copy()

    filteredDf = filteredDf.loc[
        filteredDf.groupby("feature_selector")["score"].idxmax()
    ]
    return filteredDf


# %%
def filterDataframeByBestResultsForModels(resultsDf):
    """Extracts best results for each parameter for final comparison of models
    Returns a dataframe containing only the best results
    """
    filteredDf = resultsDf.copy()

    filteredDf = filteredDf.loc[filteredDf.groupby("model")["score"].idxmax()]
    return filteredDf


# %%
def filterDataframeByBestResults(resultsDf):
    """Extracts best results for each parameter for final comparison
    Returns a dataframe containing only the best results
    """
    filteredDf = resultsDf.copy()
    return filteredDf[filteredDf["score"] == filteredDf["score"].max()]


def addColumnsScalerGenerator(resultsDf):
    """Conform the old version of dataset to new format"""
    resultsDf[RESULTS_COLUMNS[len(RESULTS_COLUMNS) - 1]] = {}
    resultsDf[RESULTS_COLUMNS[len(RESULTS_COLUMNS) - 2]] = (
        FeatureGenerator.NoFeatureGeneration.name
    )
    resultsDf[RESULTS_COLUMNS[len(RESULTS_COLUMNS) - 3]] = {}
    resultsDf[RESULTS_COLUMNS[len(RESULTS_COLUMNS) - 4]] = Scaler.NoScaling.name


def extractAllUniqueParameters(resultsDf, columns, paramColumns):
    result = {}
    for i, column in enumerate(columns):
        grouped = resultsDf.groupby(column)

        for name, group in grouped:
            paramArr = []

            group.apply(lambda x: paramArr.append(x[paramColumns[i]]), axis=1)
            paramArrDf = pd.DataFrame.from_dict(paramArr)

            result[name] = paramArrDf

    return result
