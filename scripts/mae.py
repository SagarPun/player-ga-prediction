import pandas as pd
from sklearn.metrics import mean_absolute_error

def calculate_mae_from_csv(file_path):
    df = pd.read_csv(file_path)
    actual_ga = df['Actual_G+A_24_25']
    predicted_ga = df['Predicted_G+A_24_25']
    mae = mean_absolute_error(actual_ga, predicted_ga)
    return mae


def calculate_group_mae(df_predictions, df_raw_data, group_column, prediction_col='Predicted_G+A_24_25',
                        actual_col='Actual_G+A_24_25'):

    df_player_info = df_raw_data[['Player', group_column]].drop_duplicates(subset=['Player'])
    df_merged = pd.merge(df_predictions, df_player_info, on='Player', how='left')

    if df_merged[group_column].isnull().any():
        print(
            f"Warning: Missing {group_column} data after merge. Ensure raw data contains these columns or preprocessing fills them.")
        df_merged.dropna(subset=[group_column], inplace=True)

    group_maes = {}
    for group_name in sorted(df_merged[group_column].unique()):
        subset = df_merged[df_merged[group_column] == group_name]
        if not subset.empty:
            mae = mean_absolute_error(subset[actual_col], subset[prediction_col])
            group_maes[group_name] = mae
    return group_maes