import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from palmerpenguins import load_penguins

### funs for optimizers and model
def sigmoid(x):
    if x<0:
        return np.exp(x)/(1+np.exp(x))
    else:
        return 1/(1+np.exp(-x))

def log_loss(coef, intercept, X, y):
    return -1*(y @ (coef @ X.T + intercept) - np.sum(np.log(1 + np.exp(-1*(coef @ X.T + intercept)))))

def logit(a):
        return np.log(a/(1-a))

def logit_derivative(a):
        return 1/(a*(1-a))

def log_likelihood(p,y):
        return np.sum(y*np.log(p)+(1-y+1e-8)*np.log(1-p+ 1e-8))
    
def get_beta(X,W,z):
    try:
        return np.linalg.inv(X.T@W@X)@(X.T@W@z)
    except:
        return np.linalg.pinv(X.T@W@X)@(X.T@W@z)

def add_interactions(X):
    X = np.array(X)
    products = np.array([X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i + 1, X.shape[1])]).T
    X = np.hstack((X, products))
    return X


class Stopper():
    def __init__(self,
                 n_iter_no_change: int,
                 epsilon_change: float = 1e-3) -> None:

        self.n_iter_no_change = n_iter_no_change
        self.epsilon_change = epsilon_change
        self.iters_count = 0
        self.best_coef = None
        self.best_intercept = None
        self.best_log_loss = np.inf

class EarlyStopping():
    def __init__(self,
                 n_iter_no_change: int,
                 epsilon_change: float = 1e-2) -> None:

        self.n_iter_no_change = n_iter_no_change
        self.epsilon_change = epsilon_change
        self.iters_count = 0
        self.best_coef = None
        self.best_intercept = None
        self.best_log_loss = np.inf
        
    def __call__(self, model, log_loss):
        if log_loss <= self.best_log_loss-self.epsilon_change:
            self.best_log_loss = log_loss
            self.iters_count = 0
            self.best_coef = model.coef
            self.best_intercept = model.intercept
            return False
        else:
            self.iters_count += 1
            if self.iters_count == self.n_iter_no_change:
                model.coef = self.best_coef
                model.intercept = self.best_intercept
                print("Early stopping!")
                return True
            else:
                return False
            
class ConvergenceChecker():
    def __init__(self,
                 n_iter_no_change: int = 500,
                 epsilon_change: float = 1e-2) -> None:

        self.n_iter_no_change = n_iter_no_change
        self.epsilon_change = epsilon_change
        self.iters_count = 0

    def __call__(self, train_loss):
        self.iters_count += 1
        if self.iters_count >= 10:
            results_window = train_loss[len(train_loss)-10:len(train_loss)]
            if np.max(results_window)-np.min(results_window) <= self.epsilon_change:
                    print(f"Convergence detected")
                    return True
        return False
                

class ResultsHolder():
    def __init__(self,
                  model
                  ) -> None:
          self.model = model
          self.train_loss = np.array([])
          self.val_loss = np.array([])
          self.log_like = np.array([])
    
    def evaluate_train(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        train_loss = log_loss(self.model.coef, self.model.intercept, X_train, y_train)
        self.train_loss = np.append(self.train_loss, train_loss)
    
    def evaluate_val(self,
                       X_val: np.ndarray,
                       y_val: np.ndarray
                       ) -> None:
        if X_val is not None:
            val_loss = log_loss(self.model.coef, self.model.intercept, X_val, y_val)
            self.val_loss = np.append(self.val_loss, val_loss)

    def evaluate_log_likelihood(self, 
                            p, # sigmoid of linear combination
                            y_train):
        log = log_likelihood(p, y_train)
        self.log_like = np.append(self.log_like, log)
         

### other funs
def get_most_corr_ind(df):
    corr_matrix = df.corr().abs()
    df_temp = pd.DataFrame(np.where(corr_matrix>0.8)).T
    df_temp.columns = ["val1", 'val2']
    df_temp = df_temp[df_temp['val1'] != df_temp['val2']].reset_index(drop=True)
    df_grouped = df_temp.groupby("val1").size().reset_index()
    df_grouped.sort_values([0], ascending=False, inplace=True)

    try:
        return df_grouped.iloc[0, 0]
    except:
        return None
    

def delete_corr_cols(df):
    for _ in range(df.shape[1]):
        ind = get_most_corr_ind(df)
        if ind:
            df = df.drop(df.columns[ind], axis=1)
        else: 
            break
    
    return df


### loading datasets
def load_magic_telescope(): # small dataset #1
    df = fetch_ucirepo(id=159) 

    X = df.data.features 
    y = df.data.targets

    X = delete_corr_cols(X)

    df_telescope = pd.DataFrame(pd.concat([X, y], axis=1))
    df_telescope["class"] = np.where(df_telescope["class"]=='g', 1, 0)

    return df_telescope


def load_penguins_df(): # small dataset #2
    penguins = load_penguins()

    # island: str -> int
    penguins = pd.get_dummies(penguins, columns=["island"], dtype='int')

    # sex: str -> int
    condlist = [penguins["sex"]=="male", penguins["sex"]=="female"]
    choicelist = [0, 1]
    penguins["sex"] = np.select(condlist, choicelist, 42)

    # target: str -> int
    # join Adelie and Chinstrap based on dataset's github
    condlist = [penguins["species"]=="Adelie", penguins["species"]=="Gentoo", penguins["species"]=='Chinstrap']
    choicelist = [0, 1, 0]
    penguins["species"] = np.select(condlist, choicelist, 42)

    X = penguins.iloc[:, 1:]
    y = penguins.iloc[:, 0]

    X = delete_corr_cols(X)

    df_penguins = pd.DataFrame(pd.concat([X,y], axis=1))
    df_penguins = df_penguins.dropna()

    return df_penguins


def load_banana(): # small dataset #3
    df = pd.read_csv("./data/banana.csv")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = delete_corr_cols(X)

    df_banana = pd.DataFrame(pd.concat([X, y], axis=1))
    df_banana["Quality"] = np.where(df_banana["Quality"]=='Good', 1, 0)

    return df_banana


def load_sonar(): # big dataset #1
    df = fetch_ucirepo(id=151) 
    
    X = df.data.features 
    y = df.data.targets 

    X = delete_corr_cols(X)

    df_sonar = pd.DataFrame(pd.concat([X, y], axis=1))
    df_sonar["class"] = np.where(df_sonar["class"]=='M', 1, 0)
    df_sonar = df_sonar.dropna()

    return df_sonar


def load_ionosphere(): # big dataset #2
    df = fetch_ucirepo(id=52) 
  
    X = df.data.features 
    y = df.data.targets 

    # Attribute2 has only 0
    X.drop("Attribute2", axis=1, inplace=True)
    X = delete_corr_cols(X)

    df_ion = pd.DataFrame(pd.concat([X, y], axis=1))
    df_ion["Class"] = np.where(df_ion["Class"]=='g', 1, 0)
    df_ion = df_ion.dropna()

    return df_ion


def load_covertype(): # big dataset #3
    df = fetch_ucirepo(id=31) 
  
    X = df.data.features 
    y = df.data.targets

    X = delete_corr_cols(X)

    df_cover = pd.DataFrame(pd.concat([X, y], axis=1))

    # to make classes not as unbalanced as they are now: {1, 3, 6}; {2, 4, 5, 7}.
    condlist = [df_cover["Cover_Type"]==1, df_cover["Cover_Type"]==2, df_cover["Cover_Type"]==3, 
            df_cover["Cover_Type"]==4, df_cover["Cover_Type"]==5, df_cover["Cover_Type"]==6, df_cover["Cover_Type"]==7]
    choicelist = [0, 1, 0, 1, 1, 0, 1]
    df_cover["Cover_Type"] = np.select(condlist, choicelist, 42)
    df_cover = df_cover.dropna()

    return df_cover


def load_breast_cancer(): # big dataset #4
    df = fetch_ucirepo(id=17) 

    X = df.data.features 
    y = df.data.targets 

    X = delete_corr_cols(X)

    df_breast = pd.DataFrame(pd.concat([X, y], axis=1))
    df_breast["Diagnosis"] = np.where(df_breast["Diagnosis"]=='M', 1, 0)
    df_breast = df_breast.dropna()

    return df_breast


def load_airline(): # big dataset #5
    df = pd.read_csv("./data/airline_train.csv")
    df = df.iloc[:, 1:]

    df["Gender"] = np.where(df["Gender"]=='Male', 1, 0)
    df["Customer Type"] = np.where(df["Customer Type"]=='Loyal Customer', 1, 0)
    df["Type of Travel"] = np.where(df["Type of Travel"]=='Personal Travel', 1, 0)

    condlist = [df['Class']=="Eco Plus", df['Class']=="Eco", df['Class']=="Business"]
    choicelist = [1, 0, 2]
    df['Class'] = np.select(condlist, choicelist, 42)

    df["satisfaction"] = np.where(df["satisfaction"]=='satisfied', 1, 0)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = delete_corr_cols(X)

    df_airline = pd.DataFrame(pd.concat([X,y], axis=1))
    df_airline = df_airline.dropna()

    return df_airline


def load_pcos(): # big dataset #6
    df = pd.read_excel("./data/PCOS.xlsx")
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X = delete_corr_cols(X)
    df_pcos = pd.DataFrame(pd.concat([X,y], axis=1))
    df_pcos = df_pcos.dropna()

    return df_pcos
