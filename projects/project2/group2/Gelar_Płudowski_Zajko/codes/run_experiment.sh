#!/bin/bash

export PYTHONPATH=`pwd`

python bin/prepare_data.py
python bin/generate_boruta_features.py
python bin/generate_rde_feature_list.py

#!/bin/bash
mkdir logs

python bin/hpo.py --timeout=15 --objective=XgboostObjective --name=xgboost --generate-summary >> logs/xgboost.log &
python bin/hpo.py --timeout=15 --objective=LogisticRegressionObjective --name=logistic_regression --generate-summary >> logs/logistic_regression.log &
python bin/hpo.py --timeout=15 --objective=SvmObjective --name=svm --generate-summary >> logs/svm.log &
python bin/hpo.py --timeout=15 --objective=RandomForestObjective --name=random_forest --generate-summary >> logs/random_forest.log &
python bin/hpo.py --timeout=15 --objective=AdaboostObjective --name=adaboost --generate-summary >> logs/adaboost.log &

python bin/generate_test_predictions.py
python bin/generate_visualisations.py
