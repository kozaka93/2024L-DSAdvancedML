from collections import namedtuple
from typing import Dict
from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import StandardScaler


# From: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
# Feature selection class to eliminate multicollinearity
class MultiCollinearityEliminator:

    # Class Constructor
    def __init__(self, df, target, threshold):
        self.df = df
        self.target = target
        self.threshold = threshold

    # Method to create and return the feature correlation matrix dataframe
    def createCorrMatrix(self, include_target=False):
        corrMatrix = None

        # Checking we should include the target in the correlation matrix
        if not include_target:
            df_temp = self.df.drop([self.target], axis=1)
            # Setting method to Pearson to prevent issues in case the default method for df.corr() gets changed
            # Setting min_period to 30 for the sample size to be statistically significant (normal) according to
            # central limit theorem
            corrMatrix = df_temp.corr(method='pearson', min_periods=30).abs()
        # Target is included for creating the series of feature to target correlation - Please refer the notes under the
        # print statement to understand why we create the series of feature to target correlation
        elif include_target:
            corrMatrix = self.df.corr(method='pearson', min_periods=30).abs()
        return corrMatrix

    # Method to create and return the feature to target correlation matrix dataframe
    def createCorrMatrixWithTarget(self):
        # After obtaining the list of correlated features, this method will help to view which variables
        # (in the list of correlated features) are least correlated with the target
        # This way, out the list of correlated features, we can ensure to elimate the feature that is
        # least correlated with the target
        # This not only helps to sustain the predictive power of the model but also helps in reducing model complexity

        # Obtaining the correlation matrix of the dataframe (along with the target)
        corrMatrix = self.createCorrMatrix(include_target=True)
        # Creating the required dataframe, then dropping the target row
        # and sorting by the value of correlation with target (in asceding order)
        corrWithTarget = pd.DataFrame(corrMatrix.loc[:, self.target]).drop([self.target], axis=0).sort_values(
            by=self.target)
        return corrWithTarget

    # Method to create and return the list of correlated features
    def createCorrelatedFeaturesList(self):
        # Obtaining the correlation matrix of the dataframe (without the target)
        corrMatrix = self.createCorrMatrix(include_target=False)
        colCorr = []
        # Iterating through the columns of the correlation matrix dataframe
        for column in corrMatrix.columns:
            # Iterating through the values (row wise) of the correlation matrix dataframe
            for idx, row in corrMatrix.iterrows():
                if (row[column] > self.threshold) and (row[column] < 1):
                    # Adding the features that are not already in the list of correlated features
                    if idx not in colCorr:
                        colCorr.append(idx)
                    if column not in colCorr:
                        colCorr.append(column)
        return colCorr

    # Method to eliminate the least important features from the list of correlated features
    def deleteFeatures(self, colCorr):
        # Obtaining the feature to target correlation matrix dataframe
        corrWithTarget = self.createCorrMatrixWithTarget()
        for idx, row in corrWithTarget.iterrows():
            if idx in colCorr:
                self.df = self.df.drop(idx, axis=1)
                break
        return self.df

    # Method to run automatically eliminate multicollinearity
    def autoEliminateMulticollinearity(self):
        # Obtaining the list of correlated features
        colCorr = self.createCorrelatedFeaturesList()
        while colCorr:
            # Obtaining the dataframe after deleting the feature (from the list of correlated features)
            # that is least correlated with the taregt
            self.df = self.deleteFeatures(colCorr)
            # Obtaining the list of correlated features
            colCorr = self.createCorrelatedFeaturesList()
        return self.df


def cleanup_vif(df, target, threshold=10.0):
    """
    Remove features with VIF greater than threshold
    """
    features = list(df.columns)
    features.remove(target)
    while True:
        vif = pd.DataFrame()
        vif["features"] = features
        vif["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
        max_vif = vif['VIF'].max()
        if max_vif > threshold:
            print(f"Removing feature {vif.loc[vif['VIF'].idxmax()]['features']} with VIF {max_vif}")
            features.remove(vif.loc[vif['VIF'].idxmax()]['features'])
        else:
            break
    return df[features + [target]]


def apply_standard_scaler(df, target):
    """
    Apply StandardScaler to the features
    """
    features = list(df.columns)
    features.remove(target)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def read_Rice_Cammeo_Osmancik(root: str = ""):
    """
    Rice (Cammeo and Osmancik) - https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
    """
    # 1 - Cammeo
    # 0 - Osmancik
    data, meta = arff.loadarff(root + 'data/rice+cammeo+and+osmancik/Rice_Cammeo_Osmancik.arff')
    df = pd.DataFrame(data)
    df['Class'] = df['Class'].astype(str)
    df['Class'] = df['Class'].replace('Cammeo', 1)
    df['Class'] = df['Class'].replace('Osmancik', 0)
    df['Class'] = df['Class'].astype(int)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Apply StandardScaler to the features
    apply_standard_scaler(df, 'Class')

    # Clean the data - remove collinear features
    MultiCollinearityEliminator(df, 'Class', 0.8).autoEliminateMulticollinearity()

    # Cleanup VIF
    df = cleanup_vif(df, 'Class', 10.0)

    X = df.drop(columns=['Class']).values
    y = df['Class'].values
    return X, y


def read_Online_Shoppers_intention(root: str = ""):
    """
    Online Shoppers intention Dataset - https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
    """
    filepath = root + 'data/online+shoppers+purchasing+intention+dataset/online_shoppers_intention.csv'

    df = pd.read_csv(filepath, sep=',', encoding='utf-8')
    df['Month'] = df['Month'].replace('Feb', 2)
    df['Month'] = df['Month'].replace('Mar', 3)
    df['Month'] = df['Month'].replace('May', 5)
    df['Month'] = df['Month'].replace('June', 6)
    df['Month'] = df['Month'].replace('Jul', 7)
    df['Month'] = df['Month'].replace('Aug', 8)
    df['Month'] = df['Month'].replace('Sep', 9)
    df['Month'] = df['Month'].replace('Oct', 10)
    df['Month'] = df['Month'].replace('Nov', 11)
    df['Month'] = df['Month'].replace('Dec', 12)
    df['Month'] = df['Month'].astype(int)

    df['VisitorType'] = df['VisitorType'].replace('Returning_Visitor', 1)
    df['VisitorType'] = df['VisitorType'].replace('New_Visitor', 2)
    df['VisitorType'] = df['VisitorType'].replace('Other', 0)
    df['VisitorType'] = df['VisitorType'].astype(int)

    df['Weekend'] = df['Weekend'].replace(False, 0)
    df['Weekend'] = df['Weekend'].replace(True, 1)
    df['Weekend'] = df['Weekend'].astype(int)

    df['Revenue'] = df['Revenue'].replace(False, 0)
    df['Revenue'] = df['Revenue'].replace(True, 1)
    df['Revenue'] = df['Revenue'].astype(int)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Apply StandardScaler to the features
    apply_standard_scaler(df, 'Revenue')

    # Clean the data - remove collinear features
    MultiCollinearityEliminator(df, 'Revenue', 0.8).autoEliminateMulticollinearity()

    # Cleanup VIF
    df = cleanup_vif(df, 'Revenue', 10.0)

    y = df['Revenue'].values
    X = df.drop(columns=['Revenue']).values
    return X, y


def read_Multiple_Disease_Prediction(root: str = ""):
    """
    Multiple Disease Prediction
    https://www.kaggle.com/datasets/ehababoelnaga/multiple-disease-prediction?select=Blood_samples_dataset_balanced_2%28f%29.csv
    """
    filepath = root + 'data/Multiple Disease Prediction/Blood_samples_dataset_balanced_2(f).csv'

    # 0 - Healthy
    # 1 - Any disease
    df = pd.read_csv(filepath)
    df['Disease'] = df['Disease'].replace('Healthy', 0)
    df[df['Disease'] != 0] = 1
    df['Disease'] = df['Disease'].astype(int)

    # Apply StandardScaler to the features
    apply_standard_scaler(df, 'Disease')

    # Clean the data - remove the rows with missing values
    MultiCollinearityEliminator(df, 'Disease', 0.8).autoEliminateMulticollinearity()

    # Cleanup VIF
    df = cleanup_vif(df, 'Disease', 10.0)

    y = df['Disease'].values
    X = df.drop(columns=['Disease']).values
    return X, y


def read_Web_Page_Phishing(root: str = ""):
    """
    Web Page Phishing
    https://www.kaggle.com/datasets/danielfernandon/web-page-phishing-dataset
    """
    filepath = root + 'data/Web Page Phishing/web-page-phishing.csv'
    df = pd.read_csv(filepath)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Apply StandardScaler to the features
    apply_standard_scaler(df, 'phishing')

    # Clean the data - remove collinear features
    MultiCollinearityEliminator(df, 'phishing', 0.8).autoEliminateMulticollinearity()

    # Cleanup VIF
    df = cleanup_vif(df, 'phishing', 10.0)

    y = df['phishing'].values.astype(int)
    X = df.drop(columns=['phishing']).values
    return X, y


def read_Dataset_for_Link_Phishing(root: str = ""):
    """
    Dataset for Link Phishing
    https://www.kaggle.com/datasets/winson13/dataset-for-link-phishing-detection
    """
    filepath = root + 'data/Dataset for Link Phishing/dataset_link_phishing.csv'
    df = pd.read_csv(filepath, low_memory=False)
    df = df.drop(columns=['Unnamed: 0', 'url'])
    df['domain_with_copyright'].unique()

    df['domain_with_copyright'] = df['domain_with_copyright'].replace('one', 1)
    df['domain_with_copyright'] = df['domain_with_copyright'].replace('One', 1)
    df['domain_with_copyright'] = df['domain_with_copyright'].replace('zero', 0)
    df['domain_with_copyright'] = df['domain_with_copyright'].replace('Zero', 0)

    df['domain_with_copyright'] = df['domain_with_copyright'].astype(int)

    df['status'] = df['status'].replace('phishing', 1)
    df['status'] = df['status'].replace('legitimate', 0)
    df['status'] = df['status'].astype(int)

    # Apply StandardScaler to the features
    apply_standard_scaler(df, 'status')

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Clean the data - remove collinear features
    MultiCollinearityEliminator(df, 'status', 0.8).autoEliminateMulticollinearity()

    # Cleanup VIF
    df = cleanup_vif(df, 'status', 5.0)

    y = df['status'].values
    X = df.drop(columns=['status']).values

    return X, y


def read_Statlog_Shuttle(root: str = ""):
    """
    Statlog (Shuttle) Data Set
    https://archive.ics.uci.edu/dataset/148/statlog+shuttle
    """
    filepaths = ['data/statlog+shuttle/shuttle.trn', 'data/statlog+shuttle/shuttle.tst']
    filepaths = [root + filepath for filepath in filepaths]
    df1 = pd.read_csv(filepaths[0], sep=' ', header=None)
    df2 = pd.read_csv(filepaths[1], sep=' ', header=None)
    df = pd.concat([df1, df2])

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Apply StandardScaler to the features
    apply_standard_scaler(df, 9)

    # Clean the data - remove collinear features
    MultiCollinearityEliminator(df, 9, 0.8).autoEliminateMulticollinearity()

    # Cleanup VIF
    df = cleanup_vif(df, 9, 10.0)

    y = df[9].values
    X = df.drop(columns=[9]).values

    # Remap the labels: 1 - first class, 0 - other classes
    y[y != 1] = 0

    return X, y


def read_Banknote_Authentication(root: str = ""):
    """
    Banknote Authentication Data Set
    https://archive.ics.uci.edu/dataset/267/banknote+authentication
    """
    filepath = root + 'data/banknote+authentication/data_banknote_authentication.txt'
    df = pd.read_csv(filepath, sep=',', header=None)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Apply StandardScaler to the features
    apply_standard_scaler(df, 4)

    # Clean the data - remove collinear features
    MultiCollinearityEliminator(df, 4, 0.8).autoEliminateMulticollinearity()

    # Cleanup VIF
    df = cleanup_vif(df, 4, 10.0)

    y = df[4].values
    X = df.drop(columns=[4]).values

    return X, y


def read_Airline_Passenger_Satisfaction(root: str = ""):
    """
    Airline Passenger Satisfaction
    https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
    """
    filepaths = ['data/Airline Passenger Satisfaction/train.csv', 'data/Airline Passenger Satisfaction/test.csv']
    filepaths = [root + filepath for filepath in filepaths]
    df1 = pd.read_csv(filepaths[0])
    df2 = pd.read_csv(filepaths[1])

    df = pd.concat([df1, df2])

    # drop unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'id'])

    # Encode the categorical features
    df['Gender'] = df['Gender'].replace('Male', 1)
    df['Gender'] = df['Gender'].replace('Female', 0)
    df['Gender'] = df['Gender'].astype(int)

    df['Customer Type'] = df['Customer Type'].replace('Loyal Customer', 1)
    df['Customer Type'] = df['Customer Type'].replace('disloyal Customer', 0)
    df['Customer Type'] = df['Customer Type'].astype(int)

    df['Type of Travel'] = df['Type of Travel'].replace('Personal Travel', 1)
    df['Type of Travel'] = df['Type of Travel'].replace('Business travel', 0)
    df['Type of Travel'] = df['Type of Travel'].astype(int)

    df['Class'] = df['Class'].replace('Eco Plus', 2)
    df['Class'] = df['Class'].replace('Business', 1)
    df['Class'] = df['Class'].replace('Eco', 0)
    df['Class'] = df['Class'].astype(int)

    df['satisfaction'] = df['satisfaction'].replace('neutral or dissatisfied', 0)
    df['satisfaction'] = df['satisfaction'].replace('satisfied', 1)
    df['satisfaction'] = df['satisfaction'].astype(int)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Apply StandardScaler to the features
    apply_standard_scaler(df, 'satisfaction')

    # Clean the data - remove collinear features
    MultiCollinearityEliminator(df, 'satisfaction', 0.8).autoEliminateMulticollinearity()

    # Cleanup VIF
    df = cleanup_vif(df, 'satisfaction', 10.0)

    # Reduce instance size for Airline_Passenger_Satisfaction by half
    df = df.sample(frac=0.5, random_state=0)

    y = df['satisfaction'].values
    X = df.drop(columns=['satisfaction']).values
    return X, y


def read_Optdigits(root: str = ""):
    """
    Optical Recognition of Handwritten Digits Data Set
    https://www.openml.org/search?type=data&sort=qualities.NumberOfNumericFeatures&status=active&order=desc&qualities.NumberOfFeatures=between_10_100&qualities.NumberOfClasses=%3D_2&id=980
    """
    filepath = root + 'data/optdigits/optdigits.arff'
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Remap the labels: 1 - Positive, 0 - Negative
    df['binaryClass'] = df['binaryClass'].replace(b'P', 1)
    df['binaryClass'] = df['binaryClass'].replace(b'N', 0)
    df['binaryClass'] = df['binaryClass'].astype(int)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Apply StandardScaler to the features
    apply_standard_scaler(df, 'binaryClass')

    # Clean the data - remove collinear features
    MultiCollinearityEliminator(df, 'binaryClass', 0.8).autoEliminateMulticollinearity()

    # Cleanup VIF
    df = cleanup_vif(df, 'binaryClass', 5.0)

    y = df['binaryClass'].values
    X = df.drop(columns=['binaryClass']).values
    return X, y


def read_EEG_Eye_State(root: str = ""):
    """
    EEG Eye State Data Set
    https://archive.ics.uci.edu/dataset/264/eeg+eye+state
    """
    filepath = root + 'data/eeg+eye+state/EEG Eye State.arff'
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Remap the labels: 1 - Eye detected, 0 - Eye not detected
    df['eyeDetection'] = df['eyeDetection'].replace(b'1', 1)
    df['eyeDetection'] = df['eyeDetection'].replace(b'0', 0)
    df['eyeDetection'] = df['eyeDetection'].astype(int)

    # Clean the data - remove the rows with missing values
    df = df.dropna()

    # Apply StandardScaler to the features
    apply_standard_scaler(df, 'eyeDetection')

    # Clean the data - remove collinear features
    MultiCollinearityEliminator(df, 'eyeDetection', 0.8).autoEliminateMulticollinearity()

    # Cleanup VIF
    df = cleanup_vif(df, 'eyeDetection', 10.0)

    y = df['eyeDetection'].values
    X = df.drop(columns=['eyeDetection']).values
    return X, y


XyPair = namedtuple('XyPair', ['X', 'y'])


def read_all_datasets(root: str = "") -> Dict[str, XyPair]:
    datasets = {
        'Rice_Cammeo_Osmancik': XyPair(*read_Rice_Cammeo_Osmancik(root)),
        'Online_Shoppers_intention': XyPair(*read_Online_Shoppers_intention(root)),
        # 'Multiple_Disease_Prediction': XyPair(*read_Multiple_Disease_Prediction(root)),
        'Dataset_for_Link_Phishing': XyPair(*read_Dataset_for_Link_Phishing(root)),
        'Banknote_Authentication': XyPair(*read_Banknote_Authentication(root)),
        'Optdigits': XyPair(*read_Optdigits(root)),
        'EEG_Eye_State': XyPair(*read_EEG_Eye_State(root)),
        'Web_Page_Phishing': XyPair(*read_Web_Page_Phishing(root)),
        'Statlog_Shuttle': XyPair(*read_Statlog_Shuttle(root)),
        'Airline_Passenger_Satisfaction': XyPair(*read_Airline_Passenger_Satisfaction(root)),
    }
    return datasets
