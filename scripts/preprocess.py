import pandas as pd

def preprocess_data(df_2324: pd.DataFrame, df_2425: pd.DataFrame):
    group_keys = ['Player', 'Born', 'Nation']

    sum_columns = [
        "MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt",
        "CrdY", "CrdR", "xG", "npxG", "xAG", "npxG+xAG", "PrgC", "PrgP", "PrgR"
    ]

    mean_columns = [
        "Age", "Gls_90", "Ast_90", "G+A_90", "G-PK_90",
        "G+A-PK_90", "xG_90", "xAG_90",
        "xG+xAG_90", "npxG_90", "npxG+xAG_90"
    ]

    # Drop only rows with missing values
    df_2324 = df_2324.dropna()
    df_2425 = df_2425.dropna()

    # --- Process 2023/24 ---
    df_2324_sum = df_2324[group_keys + sum_columns].groupby(group_keys, as_index=False).sum()
    df_2324_mean = df_2324[group_keys + mean_columns].groupby(group_keys, as_index=False).mean()
    df_2324_processed = pd.merge(df_2324_sum, df_2324_mean, on=group_keys)

    y_train = df_2324_processed["G+A"]
    X_train = df_2324_processed.drop(columns=["Player", "Born", "Nation", "G+A"])
    X_train = X_train.select_dtypes(include="number")

    # --- Process 2024/25 ---
    df_2425_sum = df_2425[group_keys + sum_columns].groupby(group_keys, as_index=False).sum()
    df_2425_mean = df_2425[group_keys + mean_columns].groupby(group_keys, as_index=False).mean()
    df_2425_processed = pd.merge(df_2425_sum, df_2425_mean, on=group_keys)

    y_test = df_2425_processed["G+A"]
    X_test = df_2425_processed.drop(columns=["Player", "Born", "Nation", "G+A"])
    X_test = X_test.select_dtypes(include="number")

    return X_train, y_train, X_test, y_test
