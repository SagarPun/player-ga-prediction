import pandas as pd
import os
from config import data_config, model_config_keras, train_config_keras, \
                   model_config_sklearn_mlp, train_config_sklearn_mlp, \
                   shap_config_keras, shap_config_sklearn, visualization_config, project_config

from scripts.preprocess import preprocess_data
from scripts.model_keras import train_and_evaluate_keras
from scripts.model import train_and_evaluate as train_and_evaluate_sklearn
from scripts.shap_explainer import explain_model_with_shap
from scripts.visualize import plot_actual_vs_predicted, plot_residuals_distribution, \
                              plot_residuals_vs_predicted, plot_feature_correlation_heatmap


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

    selected_model = project_config["selected_model"]

    if selected_model == "keras":
        print("\n--- Running Keras Deep Learning Model ---")
        y_pred_keras = train_and_evaluate_keras(
            X_train, y_train, X_test, y_test,
            player_names_test=player_names_test,
            model_config=model_config_keras,
            train_config=train_config_keras
        )
        print("\n--- Running SHAP Explanations for Keras Model ---")
        explain_model_with_shap(
            X_test, X_test.columns.tolist(),
            model_path=shap_config_keras['model_path'],
            sample_size_explainer_background=shap_config_keras['sample_size_explainer_background'],
            num_players_for_summary_explanation=shap_config_keras['num_players_for_summary_explanation'],
            summary_plot_path=shap_config_keras['summary_plot_path'],
            model_type="keras"
        )
        print("\n--- Generating Performance Visualizations for Keras Model ---")
        plot_actual_vs_predicted(y_test, y_pred_keras, visualization_config['actual_vs_predicted_keras_path'])
        plot_residuals_distribution(y_test, y_pred_keras, visualization_config['residuals_distribution_keras_path'])
        plot_residuals_vs_predicted(y_test, y_pred_keras, visualization_config['residuals_vs_predicted_keras_path'])
    elif selected_model == "sklearn":
        print("\n--- Running Scikit-learn MLP Regressor ---")
        y_pred_sklearn = train_and_evaluate_sklearn(
            X_train, y_train, X_test, y_test,
            player_names_test=player_names_test,
            model_config=model_config_sklearn_mlp,
            train_config=train_config_sklearn_mlp
        )
        print("\n--- Running SHAP Explanations for Scikit-learn Model ---")
        explain_model_with_shap(
            X_test, X_test.columns.tolist(),
            model_path=shap_config_sklearn['model_path'],
            sample_size_explainer_background=shap_config_sklearn['sample_size_explainer_background'],
            num_players_for_summary_explanation=shap_config_sklearn['num_players_for_summary_explanation'],
            summary_plot_path=shap_config_sklearn['summary_plot_path'],
            model_type="sklearn"
        )
        print("\n--- Generating Performance Visualizations for Scikit-learn Model ---")
        plot_actual_vs_predicted(y_test, y_pred_sklearn, visualization_config['actual_vs_predicted_sklearn_path'])
        plot_residuals_distribution(y_test, y_pred_sklearn, visualization_config['residuals_distribution_sklearn_path'])
        plot_residuals_vs_predicted(y_test, y_pred_sklearn, visualization_config['residuals_vs_predicted_sklearn_path'])

    print("\n--- Generating Common Visualizations ---")
    plot_feature_correlation_heatmap(X_train, y_train, visualization_config['feature_correlation_heatmap_path'])


if __name__ == "__main__":
    main()