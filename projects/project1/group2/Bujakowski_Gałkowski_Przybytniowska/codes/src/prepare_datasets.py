import warnings

warnings.filterwarnings("ignore")


def prepare_data(df, cast_to_int, mapping, target_variable):
    if cast_to_int:
        df[target_variable] = df[target_variable].astype(int)
    if mapping != {}:
        df[target_variable] = df[target_variable].replace(mapping)

    # Filling missing values if column have more than 10% of NaNs
    for col in df.columns:
        if df[col].isnull().mean() > 0.1:
            df[col].fillna(df[col].mean(), inplace=True)

    correlated = df.corr().abs().map(lambda x: x > 0.8 and x < 1)
    if correlated.any().any():
        collinear_vars = set()
        for col in correlated.columns:
            if col not in collinear_vars:
                correlated_columns = correlated.loc[correlated[col], col].index.tolist()
                collinear_vars.update(correlated_columns)
        collinear_vars.discard(target_variable)
        df.drop(collinear_vars, axis=1, inplace=True)

    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    return X.to_numpy(), y.to_numpy()
