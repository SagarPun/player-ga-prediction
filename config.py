# config.py

# Data Configuration
data_config = {
    "train_file": "top5-players23-24.xlsx",
    "test_file": "top5-players24-25.xlsx",
    # Columns to drop during preprocessing for features
    "excluded_features": ["Player", "Born", "Nation", "G+A"],
    "target_feature": "G+A"
}

# Model Configuration
model_config = {
    "type": "CustomMLP", # Indicates using the custom defined MLP model
    "layers": [
        {"units": 128, "activation": "elu", "dropout": 0.3},
        {"units": 64, "activation": "elu", "dropout": 0.2},
        {"units": 32, "activation": "elu", "dropout": 0.2}
    ],
    "output_units": 1, # For regression
    "optimizer": "Adam",
    "learning_rate": 0.0005,
    "loss": "mae"
}

# Training Configuration
train_config = {
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
    "model_save_path": "models/keras_ga_model", # Updated: Removed .h5 extension for SavedModel format
    "predictions_save_path": "output/predictions.csv"
}

# SHAP Configuration
shap_config = {
    "model_path": "models/keras_ga_model", # Updated: Match the save path
    "sample_size_model_explanation": 100,
    "sample_size_single_player": 50,
    "single_player_index": 0,
    "summary_plot_path": "output/shap_summary.png",
    "player_plot_path_prefix": "output/shap_player_",
    "player_csv_path_prefix": "output/shap_player_"
}