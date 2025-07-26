import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from scripts.model_keras import CustomMLP


def explain_model_with_shap(X, feature_names, model_path, sample_size_explainer_background,
                            num_players_for_summary_explanation, summary_plot_path, model_type):
    print(f"Loading {model_type} model for SHAP analysis...")

    model = None
    if model_type == "keras":
        model = tf.keras.models.load_model(model_path, custom_objects={'CustomMLP': CustomMLP})
        predict_fn = model.predict
    elif model_type == "sklearn":
        model = joblib.load(model_path)
        predict_fn = model.predict
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Must be 'keras' or 'sklearn'.")

    if num_players_for_summary_explanation and num_players_for_summary_explanation > 0 and num_players_for_summary_explanation < len(
            X):
        print(f"Sampling {num_players_for_summary_explanation} players for SHAP summary explanation...")
        X_sampled = X.sample(n=num_players_for_summary_explanation, random_state=42)
    else:
        X_sampled = X

    X_np = X_sampled.to_numpy() if isinstance(X_sampled, pd.DataFrame) else X_sampled

    background_data_for_explainer = X_np[
        np.random.choice(X_np.shape[0], min(sample_size_explainer_background, len(X_np)), replace=False)]

    print("Initializing SHAP explainer...")
    if model_type == "sklearn":
        explainer = shap.KernelExplainer(predict_fn, background_data_for_explainer)
    elif model_type == "keras":
        explainer = shap.Explainer(predict_fn, background_data_for_explainer)

    print("Calculating SHAP values...")
    shap_values = explainer(X_np)

    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, features=X_np, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path)
    print(f"SHAP summary saved to {summary_plot_path}")