# Advance Machine Learning project #2

## Running experiments

Running only files from `bin` folder is required to reproduce experiments. Please use the following `bash` script to run all experiments (using other tool than `bash` may require small adjustment in the following snippet). Please note that `boruta` is poorly deployed on PyPi and thus, may generate errors during execution. In this case you may consider removing `generate_boruta_features.py` from `run_expeirment.sh`.

```{bash}
conda create --name=aml2 python=3.9.19
conda activate aml2
pip install -r requirements.txt

chmod +x run_experiment.sh
./run_experiment.sh
```

Plots are automatically generated in `results/vis` when running `run_expeirment.sh`, but can be also viewed in `notebooks/result_analysis`.

## Methodology

* We select most important features using Boruta algorithm
* We research the impact of the different feature selection methods: RFE with various estimators: random forest, linear regression, etc. RFE creates a ranking of features selected by Boruta
* We optimize several ML models in terms of their hyperparameters and the number of features used.

## Technical aspects

We use hold-out split due to space we want to explore (feature selection method x number_of_features x model x hyperparameters). We use Optuna to search the space.

### Features selection flow

* Boruta_features: most important indexes in the original data (so `df.iloc[:,boruta_idx]`)
* RFE: indexes importance order AFTER Boruta's selection

### Remarks

`boruta` might not work due to `numpy` confilicts - removing `numpy` depricated types in source code solves the issue.

<!-- ## TODO -->

