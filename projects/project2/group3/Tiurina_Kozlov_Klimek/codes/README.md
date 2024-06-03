# Project 2

## Overview of Project 2:
- The goal is to build a model that combines two advantages: has high accuracy and is parsimonious, i.e., is based on a small number of variables.
- We have 5000 historical training data. Each client is described with 500 variables (variables are anonymized). Your task is to build a model that predicts which customers in the test set took advantage of the offer.
    - That means that the number of variables in the model should be small, but the model should have high accuracy.
    - We need a good balance between the number of variables and the accuracy of the model, that is why we need the algorithm that can adapt the number of variables in the model to the specific minimization problem.
- Additionally, you should indicate the variables that were used to build the model. The effectiveness of your prediction will be assessed as follows:
    - For each designated customer who actually took advantage of the offer, the company will pay you €10.
    - For each variable used, you must pay €200 (the company bears the cost of obtaining information related to individual variables).
        - Our goal is to maximize the profit, i.e., to maximize the difference between the revenue from the correct prediction and the cost of the variables used in the model.
        - The profit is calculated as follows: `profit = 10 * (number of correctly predicted customers) - 200 * (number of variables used)`
        - **This will be the criterion for the effectiveness of the model, instead of it being the accuracy of the model.**

Students: TBD

To reproduce the results presented by our team, please use the following steps.

1. Copy data to /data/data to have the following structure.
```
codes
│   README.md   
│
└───data
│   │   eda.ipynb
│   │
│   └───data
│       │   x_test.txt
│       │   x_train.txt
│       │   y_train.txt
│   
└───...
```

2. (Optional) run /data/eda.ipynb to investigate project dataset.

3. To reproduce results of the submission solution, please, run /models/SVM.ipynb