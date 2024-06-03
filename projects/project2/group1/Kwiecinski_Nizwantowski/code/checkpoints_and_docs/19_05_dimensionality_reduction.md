# Description of the approach to the problem
Initially, I wanted to check whether simple dimensionality reduction algorithms could be used for our problem. 

## Experiments
I tested some simple pipeline, which consisted of standard scaler, dimensionality reduction method - PCA or Feature Agglomeration with trained Random Forest on returned embeddings

I performed a simple crossvalidation grid search using the proposed pipeline, and saved the best parameters of the methods.

## Results

It turned out that the dimensionality reduction methods that were under investigation were not a good choice in our settings. The top 1 accuracy reported by the Random Forest on data transformed by PCA was only 54%, which is comparable to no dimensionality reduction technique at all.

## Conclusions

Since the objective of our problem was not to reduce the dimensionality of our data, but use as little features as possible, we won't be continuing investigation in this direction