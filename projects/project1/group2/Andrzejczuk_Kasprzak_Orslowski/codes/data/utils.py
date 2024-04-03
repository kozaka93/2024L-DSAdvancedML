from typing import Union

import pandas as pd
from sklearn.datasets import fetch_openml


def get_openml_df(openml_id: Union[str, int]) -> pd.DataFrame:
    if isinstance(openml_id, str):
        dataset = fetch_openml(openml_id, parser='auto')
    elif isinstance(openml_id, int):
        dataset = fetch_openml(data_id=openml_id, parser='auto')
    else:
        raise ValueError(f'{openml_id} is an invalid value of openml_id')
    df = dataset['data']
    df['Target'] = dataset['target']
    
    return df
