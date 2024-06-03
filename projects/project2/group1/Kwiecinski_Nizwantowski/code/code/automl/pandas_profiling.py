import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

X_train = np.load('../../data/X_train.npy')
y_train = np.load('../../data/y_train.npy')

X_df = pd.DataFrame(X_train)
y_df = pd.DataFrame(y_train, columns=['target'])
df = pd.concat([X_df, y_df], axis=1)

profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
profile.to_file("pandas_profiling.html")


