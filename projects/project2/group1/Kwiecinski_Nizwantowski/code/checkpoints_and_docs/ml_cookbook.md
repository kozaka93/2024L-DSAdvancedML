# ML cookbook
Apart from scoring an excellent grades, our primary goal during this class and the project is to learn something valuable. That's why in this file, we will gather some interesting insights and ideas learned during the second project.


## Data
The data splitting is essential to avoid *data leakage* - a situation where information outside the training dataset is used to create the model. The information may come from poorly prepared datasets, mistake of the data analyst etc. This situation should be avoided, since it overestimates the performance of the created model.

### Crossvalidation
Crossvalidation is very good tool to assess the model's performance, but when we also need to select the hyperparameters, there should be another dataset on which the choosen model's performance should be estimated.

## Project structure
Some tricks that may work to avoid chaos in the project:

1. Reporting in the mean time - create checkpoints that could be parts of the report
2. General notebook to preprocess data

## Model assesment
### Custom loss function
In the cases as here, where we need to optimize some criterion, we can define our loss function and optimize it using any model as usual

### Most common approach
The model could be trained on different train/test dataset splits, and there could be taken an average of its performance - this is a very stable solution

### Conservative approach
The safest way of evaluating the performance of the model is to take the minimum value achieved during the evaluation - for example, a minimal value of accuracy



