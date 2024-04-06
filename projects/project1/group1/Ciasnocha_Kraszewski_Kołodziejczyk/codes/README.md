# Project 1

1. Task 1 - Datasets:
   - 9 datasets:
     - 3 small (at most 10 features)
     - 6 large (at least 10 features)
   - plus for original datasets
   - simple data preprocessing (handle missing values, remove collinear features)
2. Task 2 - Implementation:
    - 3 algorithms:
      - IWLS (Iteratively reweighted least squares)
      - SGD (Stochastic Gradient Descent)
      - ADAM (Adaptive Moment Estimation)
    - algorithms should support variable interactions
    - use default hyperparameters (recommended)
3. Task 3 - Experiments:
    - propose stopping criteria (same for all algorithms), brief literature review
    - Balanced Accuracy as a metric, avg over min. 5 runs, if does not converge stop after 500 iterations, training test resampling (?)
    - convergance analysis
    - compare algorithms (ours + sklearn: LDA, QDA, Decision Tree, Random Forest)
    - **small datasets only:** compare models with and without interactions (without sklearn models)


## Plan

1. Task 1 + 2 (deadline: 2024-03-07)
- [ ] Choice + setup of environment (Filip)
- [ ] Choice of datasets + description (Jerzy)
- [ ] Data preprocessing (Jerzy)
- [ ] Algorithms interface (Filip)
- [ ] IWLS implementation + check recommended hyperparams (Jerzy)
- [ ] SGD implementation + check recommended hyperparams (Michał)
- [ ] ADAM implementation + check recommended hyperparams (Filip)
- [ ] Variable interactions for algorithms (Michał)
2. Task 3 (deadline: 2024-03-21)
- [ ] Propose stopping criteria + prepare text (Michał)
- [ ] Training pipeline (Filip)
- [ ] Implement Balanced Accuracy metric + convergance analysis (Jerzy)
- [ ] Implement sklearn models - adapt to our interface (Michał)
- [ ] Set up experiments
  - [ ] all datasets with sklearn models (Filip)
  - [ ] small datasets with and without interactions (Jerzy)
3. Report + presentation (deadline: 2024-03-29)
- [ ] Report (every section started form new line, at most 2 pages, one text, one figures)
  - [ ] Metodology (Task1, stopping criteria, Balanced Accuracy with convergance)
  - [ ] Convergance analysis
  - [ ] Comparison of algorithms with sklearn models
  - [ ] Comparison of models with and without interactions 
- [ ] Record presentation 