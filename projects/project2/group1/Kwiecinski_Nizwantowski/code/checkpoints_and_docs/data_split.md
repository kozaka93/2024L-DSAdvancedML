# How do we split the data?

Machine learning models performance needs to be evaluated on a dataset different than the one used during training. Therefore, as a preparation for the **real** part of the project - model training, dimensionality reduction, etc. we split the original training dataset containing 5000 observations into two sets - training and validation.


Since now, if not mentioned otherwise, if we say training dataset, we mean the one separated by us containing 4000 observations.

## Why? 
Our main idea was to perform training, possibly using cross-validation on the smaller training set, and evaluate the model's performance on the validation dataset. This measure is taken because we need to know how well our model performs and estimate the error on the test dataset.

## Class imbalance
There was no class imbalance in the original training set of 5000 observations. As seen in the table below, there seems to be no class imbalance in the separated datasets either.


| Dataset | Count | Factor of positive labels |
|-------|-------| ------ |
| Original training | 0.4992 | 
| Training | 4000 | 0.49925 |
| Validation | 1000 | 0.499 |

