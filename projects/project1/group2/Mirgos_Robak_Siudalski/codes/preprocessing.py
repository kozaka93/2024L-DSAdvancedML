import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from scipy.io import arff


def read_dataframe(path):
    #loading different data formats
    if path.endswith('.arff'):
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.xlsx'): # feautres case
        df = pd.read_excel(path)
        df = df.iloc[:, 2:]
    else:
        raise ValueError('File type not supported')
    #droping rows with missing values
    df.dropna(inplace=True)
    return df


def col_to_drop(X, threshold):
    importances = pd.DataFrame(data={
    'name': X.columns,
    'importance':  X.var(axis=0).values
    })
    importances.sort_values(by='importance', ascending=False, inplace=True)

    corr_matrix = X.corr().abs()
    upper_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []
    for column in upper_corr.columns:
        if any(upper_corr[column] > threshold):
            correlated = upper_corr[upper_corr[column] > threshold].index
            for col in correlated:
                if importances[importances['name'] == col]['importance'].values[0] < importances[importances['name'] == column]['importance'].values[0]:
                    to_drop.append(col)
                else:
                    to_drop.append(column)

    return np.unique(to_drop)


def preprocess_dataframe(df, test_size=0.2):
    #selecting X and y
    if df.iloc[:, -1].nunique() <= 3:
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
    elif df.iloc[:, 0].nunique() <= 3:
        X, y = df.iloc[:, 1:], df.iloc[:, 0]

    #encoding y
    label_encoder = LabelEncoder()

    # special case for maternal health dataset
    if y.nunique() == 3:
        y = label_binarize(y, classes=['low risk']).ravel()
    else:
        y = label_encoder.fit_transform(y)
    
    #encoding X
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col])

    #splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    #select high correlation columns to drop 
    to_drop = col_to_drop(X_train, 0.7)

    X_train = X_train.drop(columns=to_drop).to_numpy()
    X_test = X_test.drop(columns=to_drop).to_numpy()

    return X_train, X_test, y_train, y_test