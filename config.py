data_config = {
    "train_file": "top5-players23-24.xlsx",
    "test_file": "top5-players24-25.xlsx",
    "excluded_features": ["Player", "Born", "Nation", "G+A"],
    "target_feature": "G+A"
}

model_config_keras = {
    "type": "CustomMLP",
    "layers": [
        {"units": 128, "activation": "elu", "dropout": 0.3},
        {"units": 64, "activation": "elu", "dropout": 0.2},
        {"units": 32, "activation": "elu", "dropout": 0.2}
    ],
    "output_units": 1,
    "optimizer": "Adam",
    "learning_rate": 0.0005,
    "loss": "mae"
}

model_config_sklearn_mlp = {
    "hidden_layer_sizes": (100, 50, 25),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "learning_rate": "adaptive",
    "tol": 1e-4,
    "random_state": 42
}

train_config_keras = {
    "epochs": 200,
    "batch_size": 32,
    "validation_split": 0.2,
    "callbacks": {
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 20,
            "restore_best_weights": True
        },
        "reduce_lr_on_plateau": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 10,
            "verbose": 1,
            "min_lr": 0.00001
        }
    },
    "model_save_path": "models/keras_ga_model",
    "predictions_save_path": "output/predictions_keras.csv"
}

train_config_sklearn_mlp = {
    "max_iter": 1000,
    "early_stopping": True,
    "n_iter_no_change": 20,
    "validation_fraction": 0.1,
    "model_save_path": "models/mlp_ga_predictor.pkl",
    "predictions_save_path": "output/predictions_sklearn.csv"
}

shap_config_keras = {
    "model_path": "models/keras_ga_model",
    "sample_size_explainer_background": 100,
    "num_players_for_summary_explanation": 100,
    "summary_plot_path": "output/shap_summary_keras.png"
}

shap_config_sklearn = {
    "model_path": "models/mlp_ga_predictor.pkl",
    "sample_size_explainer_background": 100,
    "num_players_for_summary_explanation": 100,
    "summary_plot_path": "output/shap_summary_sklearn.png"
}

visualization_config = {
    "actual_vs_predicted_keras_path": "output/actual_vs_predicted_keras.png",
    "residuals_distribution_keras_path": "output/residuals_distribution_keras.png",
    "residuals_vs_predicted_keras_path": "output/residuals_vs_predicted_keras.png",

    "actual_vs_predicted_sklearn_path": "output/actual_vs_predicted_sklearn.png",
    "residuals_distribution_sklearn_path": "output/residuals_distribution_sklearn.png",
    "residuals_vs_predicted_sklearn_path": "output/residuals_vs_predicted_sklearn.png",

    "feature_correlation_heatmap_path": "output/feature_correlation_heatmap.png"
}

# Set to "keras" or "sklearn" to choose which model to run
project_config = {
    "selected_model": "keras"
}