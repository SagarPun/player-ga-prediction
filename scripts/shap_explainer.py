import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def explain_model_with_shap(X, feature_names, model_path="models/keras_ga_model.h5", sample_size=100):
    print("🔍 Loading model for SHAP analysis...")
    model = load_model(model_path)

    # Convert to numpy if not already
    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    background = X_np[np.random.choice(X_np.shape[0], min(sample_size, len(X_np)), replace=False)]

    print("🧠 Initializing SHAP explainer...")
    explainer = shap.Explainer(model.predict, background)

    print("📈 Calculating SHAP values...")
    shap_values = explainer(X_np)

    # Summary plot
    print("📊 Generating SHAP summary plot...")
    shap.summary_plot(shap_values, features=X_np, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("output/shap_summary.png")
    print("✅ SHAP summary saved to output/shap_summary.png")

def explain_single_player(X, model_path="models/keras_ga_model.h5", player_index=0):
    import shap
    import matplotlib.pyplot as plt
    import pandas as pd
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    player_data = X.iloc[[player_index]]  # Keep shape (1, n)

    background = X.sample(50, random_state=42)
    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(player_data)

    print(f"🎯 SHAP explanation for player index {player_index}")

    # Plot impact
    shap.plots.bar(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(f"output/shap_player_{player_index}.png")
    print(f"✅ SHAP bar plot saved to output/shap_player_{player_index}.png")

    # Save impact values to CSV
    impacts = pd.DataFrame({
        "Feature": shap_values.feature_names,
        "Impact": shap_values.values[0],
        "Feature Value": shap_values.data[0]
    }).sort_values("Impact", key=abs, ascending=False)

    impacts.to_csv(f"output/shap_player_{player_index}.csv", index=False)
    print(f"📄 SHAP values saved to output/shap_player_{player_index}.csv")

