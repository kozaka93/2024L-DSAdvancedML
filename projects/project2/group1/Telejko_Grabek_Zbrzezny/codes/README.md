# AML-2024L-Offer-Acceptance-Prediction

---
**NOTE**

In our codebase we are indexing the variables from 0. However in the project report and final predictions submission files (`313547_vars.txt`, `313547_obs.txt`) the numeration of features and customers starts from 1. 

---

### Evaluation
The effectiveness of the prediction is assessed as follows:
- For each designated customer who actually took advantage of the offer, the company will pay you $€10$.
- For each variable used, you must pay $€200$ (the company bears the cost of obtaining information related to individual variables).

### Choosing and running the model
If you want to utilize a specific strategy, you may choose the feature selector and model
from any of the experiment defined in experiment_results. To do so, choose the desired model from `experiment_results` and load it:
```{python}
import os
import pickle

EXPERIMENT_PATH = "experiment_results"
EXPERIMENT_NAME = "exp_mlpc_rffis_905309_6850"  # example experiment filename

if __name__ == "__main__":
    with open(os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME), "rb") as file:
        experiment = pickle.load(file)  # follows the Experiment class

    # get the feature selector
    feature_selector = experiment.feature_selector(**experiment.feature_selector_config)
    # get and initialize the classifier
    model = experiment.classifier(**experiment.classifier_config)
    # continue...
```
The template above is available in `run_model_template.py`.

### Creating and running a new experiment (feature selector + model)
Prepare a model for the classification model and the feature selector.
```{python}
# Imports
from sklearn.svm import SVC  
from src.custom_feature_selectors.manual_feature_selector import ManualFeatureSelector
from src.experiment import Experiment
from src.experiment_utils import perform_experiments


# define the experiment config (you can include many experiments in it)
experiment_config = [
    Experiment(
        # creating example model for classification (SVM)
        classifier=SVC,
        classifier_config={
            "probability": True,
        },
        # creating example feature selector
        # here `ManualFeatureSelector` is used where the indices of columns are specified # directly
        feature_selector=ManualFeatureSelector,
        feature_selector_config={
            "indices": lasso_important_features
        },
    )
]

# run the experiments
# the configuration and results will be saved as a pickle file in `experiment_results`
# The results for each CV split and the overall score is printed immediately
scores, indices = perform_experiments(X, y, experiment_config)
```
**Example Output**
```
Experiment exp_svc_mfs_287d81 in progress...
Using 29 features, we properly classified 115/200 clients.
Using 29 features, we properly classified 100/200 clients.
Using 29 features, we properly classified 115/200 clients.
Using 29 features, we properly classified 112/200 clients.
Using 29 features, we properly classified 116/200 clients.
{'exp_svc_mfs_287d81': -220}
```
