import os
import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

def train_and_evaluate(X_train, y_train, X_test, y_test,df_test_raw=None):
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n🔍 Mean Absolute Error (MAE) on 24/25: {mae:.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/mlp_ga_predictor.pkl")

    # Save predictions
    os.makedirs("output", exist_ok=True)
    df_out = pd.DataFrame({
        "Predicted_G+A_24_25": y_pred,
        "Actual_G+A_24_25": y_test.reset_index(drop=True)
    })

    if df_test_raw is not None and "Player" in df_test_raw.columns:
        df_out["Player"] = df_test_raw["Player"].reset_index(drop=True)

    df_out.to_csv("output/predictions.csv", index=False)
    print("✅ Predictions saved to output/predictions.csv")
