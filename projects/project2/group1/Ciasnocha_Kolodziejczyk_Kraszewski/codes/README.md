# Project 2

Authors:
- Michał Ciasnocha
- Jerzy Kraszewski
- Filip Kołodziejczyk

## Problem description

The goal is to build a model that combines two advantages: has high accuracy and is parsimonious, i.e., is based on a small number of variables. Imagine that the company you work for has commissioned you to build
a predictive model whose purpose is to identify customers who use the bank’s marketing offer. The model should identify 1000 customers that are most likely to use the offer. The number is related to the fact that the company can send offer to maximally 1000 customers at once. It should be done using as few features as possible since using more features is more expensive. Main target is to maximize the net profit, which is expressed ib the following formula:

$$\text{Net profit} = 10 \times \text{TP} - 200 \times \text{number of variables}$$
    
where:
- TP is the number of true positives, i.e., the number of customers who actually used the offer and were correctly identified by the model
- 200 is the cost of using one variable
- 10 is the profit from one correctly identified customer

## Steps

Since the process is quite complex, we decided to create several notebooks, each dedicated to a different part of the process. The notebooks and steps are as follows:

1. Data Analysis (`data_analysis.ipynb`) - here we explore the data, manually checking for any interesting patterns or relationships between variables and the nature of the data itself. It is important step for the decision making in next steps, especially for the feature selection. In the summary we decided on the models we will use for this problem.
2. Initial model hyperparameter tuning (`rf_tuning.ipynb`) - here we tune the hyperparameters of one of the models we selected in the previous step. We tune Random Forest on all features to get the a good model for the feature selection methods which require a model to be trained.
3. Feature selection (`feature_selection.ipynb`) - here we use several feature selection methods to reduce the number of features. We are interested in finding the top candidates for the optimal subset, selecting only 11 features.
4. Final feature and model selection (`final_model.ipynb`) - here we use the features selected in the previous step to tune several models and select the best one. Wiht the best performing XGBoost we attempt the exhaustive search for the optimal subset of features. In parallel, feature selection methods are reevaluated. Search is done using the net profit formula. After the best subset is found, the model is trained on the subset and the final predictions of challenge data are made.