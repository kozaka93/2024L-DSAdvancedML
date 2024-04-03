import pandas as pd

from data.utils import get_openml_df


def get_blood_transfusion() -> pd.DataFrame:
    """
    https://www.openml.org/search?type=data&sort=runs&status=active&id=1464

    4 Variables:
    - Recency: months since last donation
    - Frequency: total number of donation
    - Monetary: total blood donated in c.c.
    - Time: months since first donation
    
    Target: a binary variable representing whether he/she donated blood in March 2007 
        (1 stand for donating blood; 0 stands for not donating blood).
    """
    df = get_openml_df('blood-transfusion-service-center')
    df = df.rename(columns={
        'V1': 'Recency',
        'V2': 'Frequency',
        'V3': 'Monetary',
        'V4': 'Time',
    })
    df['Target'] = df['Target'].map({'1': 0, '2': 1})
    return df


def get_diabetes() -> pd.DataFrame:
    """
    https://www.openml.org/search?type=data&sort=runs&status=active&id=37
    
    8 Variables:
    - preg: Number of times pregnant
    - plas: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    - pres: Diastolic blood pressure (mm Hg)
    - skin: Triceps skin fold thickness (mm)
    - insu: 2-Hour serum insulin (mu U/ml)
    - mass: Body mass index (weight in kg/(height in m)^2)
    - pedi: Diabetes pedigree function
    - age: Age (years)
    
    Target: 0 - tested negative; 1 - tested positive 
    """
    df = get_openml_df(37)
    new_target = df['Target'].map({'tested_negative': 0, 'tested_positive': 1})
    df['Target'] = new_target
    return df


def get_banknotes() -> pd.DataFrame:
    """
    https://www.openml.org/search?type=data&sort=runs&status=active&id=1462
    
    4 Variables:
    - V1: variance of Wavelet Transformed image
    - V2: skewness of Wavelet Transformed image
    - V3: curtosis of Wavelet Transformed image
    - V4: entropy of image
    
    Target: 0 - genuine; 1 - forged 
    """
    df = get_openml_df('banknote-authentication')
    # convert 1/2 to 0/1
    df['Target'] = df['Target'].map({'1': 0, '2': 1})
    return df
