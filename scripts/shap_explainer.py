import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # Import tensorflow for load_model and custom_objects
from scripts.model_keras import CustomMLP # Import your custom model class

def explain_model_with_shap(X, feature_names, model_path, sample_size, summary_plot_path):
    print("Loading model for SHAP analysis...")
    # FIX: Pass custom_objects to load_model for subclassed models
    model = tf.keras.models.load_model(model_path, custom_objects={'CustomMLP': CustomMLP})

    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    background = X_np[np.random.choice(X_np.shape[0], min(sample_size, len(X_np)), replace=False)]

    print("Initializing SHAP explainer...")
    explainer = shap.Explainer(model.predict, background)

    print("Calculating SHAP values...")
    shap_values = explainer(X_np)

    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, features=X_np, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path)
    print(f"SHAP summary saved to {summary_plot_path}")

def explain_single_player(X, model_path, player_index, player_plot_path_prefix, player_csv_path_prefix):
    print(f"Loading model for SHAP explanation for player index {player_index}...")
    # FIX: Pass custom_objects to load_model for subclassed models
    model = tf.keras.models.load_model(model_path, custom_objects={'CustomMLP': CustomMLP})
    player_data = X.iloc[[player_index]]

    background = X.sample(50, random_state=42) # Fixed sample size for single player explanation background
    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(player_data)

    print(f"SHAP explanation for player index {player_index}")

    shap.plots.bar(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(f"{player_plot_path_prefix}{player_index}.png")
    print(f"SHAP bar plot saved to {player_plot_path_prefix}{player_index}.png")

    impacts = pd.DataFrame({
        "Feature": shap_values.feature_names,
        "Impact": shap_values.values[0],
        "Feature Value": shap_values.data[0]
    }).sort_values("Impact", key=abs, ascending=False)

    impacts.to_csv(f"{player_csv_path_prefix}{player_index}.csv", index=False)
    print(f"SHAP values saved to {player_csv_path_prefix}{player_index}.csv")