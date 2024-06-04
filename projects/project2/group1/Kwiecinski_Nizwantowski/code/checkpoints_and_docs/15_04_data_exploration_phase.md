Today we had a meet with Kasia. During the meeting we decided that from now on we will make notes. This is the first one and hopefully not the last. It will contain progress we made till this point.

A data was acquired. It contains 2 sets: train and test. Both have 5000 observations and 500 features. Additionally train set has binary target variable. It is a case of balanced classification problem. 

Data is artificially generated and is drawn from 4 different distributions:
    - the first 10 features are normally distributed and are strongly correlated to each other
    - the next 190 features are also drawn from a normal distribution but aren't correlated and have lower variance
    - the next 200 features are drawn from a uniform distribution [0,1]
    - the next 100 features are drawn from something that resembles a chi-squared distribution

It is worth to check up correlation matrix of first 120 features to get deeper insights. Why not all of them? Because that's the only part where anything interesting happens. Rest looks like noise. 

The same behaviors (marginal distributions of features and correlation patterns) are visible in test set.

There is no variable that has higher absolute value of Pearson correlation with target that 0.05, thus we are dealing with non linear relationships. Spearman correlation is even weaker: absolute value smaller than 0.04. 

Then simple model was created in order to estimate "problem difficulty". Train data was divided into two sets in ratio 4:1 with random state 213.
On top of it random forest was created wist random state of 213, 200 estimators and max depth of 10. This model had 63.5% accuracy. Thanks to this model we got access to feature importance score. 

We had an idea to check, what will happen if we build model, without some of the features with low importance. Again it was random forest, on the same train/test split. No max depth this time, random state 213 and 200 estimators. It achieved accuracy of 66.4%. 

Happy with the results we decided to take this idea to extreme. We set an arbitrary threshold on variable importance and using it we filtered variables. This procedure gave us 14 variables. They come form 2 distinct areas in dataset. One of them is first 10 variables with high corelation and the second is range of columns (100:105). Building the same model as previously but on smaller set of variables resulted in accuracy of 72.1%. Later it turned out we got lucky. 

Because we checked different model: SVM. Based on 40 different train/test we compared our forest build on 14 features with SVM with default hyperparameters. Random forest gave an average of 69% accuracy while SVM 71.5%. 

We tried other methods, for example xgboost, but with default hyperparameters it performed poorly (as expected) 66%. 

Then we performed two additional experiments. Using PCA before training the model - SVM. The average accuracy based on 40 train/test split decreased by around 0.5%. On this stage we won't be using it. 

The second experiment modified the kernel in SVM, the best results were obtain when using poly kernel with degree 2 (73% accuracy vs 71.5% obtained previously)