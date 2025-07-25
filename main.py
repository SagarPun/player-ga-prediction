import pandas as pd
import os
from config import data_config, model_config, train_config, shap_config

from scripts.preprocess import preprocess_data
from scripts.model_keras import train_and_evaluate_keras
from scripts.shap_explainer import explain_model_with_shap, explain_single_player


def load_raw_data(data_cfg):
    path_2324 = os.path.join("data", data_cfg['train_file'])
    path_2425 = os.path.join("data", data_cfg['test_file'])

    df_2324 = pd.read_excel(path_2324)
    df_2425 = pd.read_excel(path_2425)

    print(f"Loaded 2023/24 data shape: {df_2324.shape}")
    print(f"Loaded 2024/25 data shape: {df_2425.shape}")

    return df_2324, df_2425

def main():
    df_2324, df_2425 = load_raw_data(data_config)
    X_train, y_train, X_test, y_test, player_names_test = preprocess_data(
        df_2324, df_2425,
        excluded_features=data_config['excluded_features'],
        target_feature=data_config['target_feature']
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    train_and_evaluate_keras(
        X_train, y_train, X_test, y_test,
        player_names_test=player_names_test,
        model_config=model_config,
        train_config=train_config
    )

    # explain_model_with_shap(
    #     X_test, X_test.columns.tolist(),
    #     model_path=shap_config['model_path'],
    #     sample_size=shap_config['sample_size_model_explanation'],
    #     summary_plot_path=shap_config['summary_plot_path']
    # )
    explain_single_player(
        X_test,
        model_path=shap_config['model_path'],
        player_index=shap_config['single_player_index'],
        player_plot_path_prefix=shap_config['player_plot_path_prefix'],
        player_csv_path_prefix=shap_config['player_csv_path_prefix']
    )


if __name__ == "__main__":
    main()