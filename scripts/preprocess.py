import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df_2324: pd.DataFrame, df_2425: pd.DataFrame,
                    excluded_features: list, target_feature: str):
    group_keys = ['Player', 'Born', 'Nation']

    sum_columns = [
        "MP", "Starts", "Min", "90s", "Gls", "Ast", target_feature, "G-PK", "PK", "PKatt",
        "CrdY", "CrdR", "xG", "npxG", "xAG", "npxG+xAG", "PrgC", "PrgP", "PrgR"
    ]

    mean_columns = [
        "Age", "Gls_90", "Ast_90", f"{target_feature}_90", "G-PK_90",
        "G+A-PK_90", "xG_90", "xAG_90",
        "xG+xAG_90", "npxG_90", "npxG+xAG_90"
    ]

    df_2324 = df_2324.dropna()
    df_2425 = df_2425.dropna()

    df_2324_sum = df_2324[group_keys + sum_columns].groupby(group_keys, as_index=False).sum()
    df_2324_mean = df_2324[group_keys + mean_columns].groupby(group_keys, as_index=False).mean()
    df_2324_processed = pd.merge(df_2324_sum, df_2324_mean, on=group_keys)

    y_train = df_2324_processed[target_feature]
    X_train = df_2324_processed.drop(columns=excluded_features + [target_feature])
    X_train = X_train.select_dtypes(include="number")

    df_2425_sum = df_2425[group_keys + sum_columns].groupby(group_keys, as_index=False).sum()
    df_2425_mean = df_2425[group_keys + mean_columns].groupby(group_keys, as_index=False).mean()
    df_2425_processed = pd.merge(df_2425_sum, df_2425_mean, on=group_keys)

    player_names_test = df_2425_processed["Player"]

    y_test = df_2425_processed[target_feature]
    X_test = df_2425_processed.drop(columns=excluded_features + [target_feature])
    X_test = X_test.select_dtypes(include="number")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print("Features scaled using StandardScaler.")

    return X_train_scaled, y_train, X_test_scaled, y_test, player_names_test