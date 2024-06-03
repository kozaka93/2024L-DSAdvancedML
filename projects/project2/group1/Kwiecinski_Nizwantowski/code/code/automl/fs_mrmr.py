from mrmr import mrmr_classif
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("..")

from metrics import make_competition_scorer, default_competition_metric

if __name__ == '__main__':
    X_train = np.load('../../data/x_train.npy')
    y_train = np.load('../../data/y_train.npy')
    X_val = np.load('../../data/x_val.npy')
    y_val = np.load('../../data/y_val.npy')
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    X = pd.DataFrame(X_train)
    y = pd.Series(y_train)

    selected_features = mrmr_classif(X, y, K=5, return_scores=False)
    print(selected_features)
    print("Number of selected features: ", len(selected_features))
    
    
    # train a model on selected features
    X_filtered = X_train[:, selected_features]
    X_val_filtered = X_val[:, selected_features]
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    rf.fit(X_filtered, y_train)
    print("Train score: ", default_competition_metric(y_train, rf.predict_proba(X_filtered)[:, 1], X_filtered.shape[1],  rf.predict_proba(X_filtered)[:, 1]))
    print("Val score: ", default_competition_metric(y_val, rf.predict_proba(X_val_filtered)[:, 1], X_val_filtered.shape[1], rf.predict_proba(X_val_filtered)[:, 1]))
    