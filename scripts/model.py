import os
import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

def train_and_evaluate(X_train, y_train, X_test, y_test,
                       player_names_test=None, model_config=None, train_config=None):
    if model_config is None:
        raise ValueError("model_config must be provided for MLPRegressor.")
    if train_config is None:
        raise ValueError("train_config must be provided for MLPRegressor.")

    model = MLPRegressor(
        hidden_layer_sizes=model_config.get('hidden_layer_sizes', (64, 32)),
        activation=model_config.get('activation', 'relu'),
        solver=model_config.get('solver', 'adam'),
        alpha=model_config.get('alpha', 0.0001),
        learning_rate_init=model_config.get('learning_rate_init', 0.001),
        learning_rate=model_config.get('learning_rate', 'constant'),
        max_iter=train_config.get('max_iter', 500),
        early_stopping=train_config.get('early_stopping', False),
        n_iter_no_change=train_config.get('n_iter_no_change', 10),
        validation_fraction=train_config.get('validation_fraction', 0.1),
        tol=model_config.get('tol', 1e-4),
        random_state=model_config.get('random_state', 42)
    )

    print("Training Scikit-learn MLPRegressor...")
    model.fit(X_train, y_train)
    print("Training complete.")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE) on 24/25: {mae:.2f}")

    os.makedirs(os.path.dirname(train_config['model_save_path']), exist_ok=True)
    joblib.dump(model, train_config['model_save_path'])
    print(f"Model saved to {train_config['model_save_path']}")

    os.makedirs(os.path.dirname(train_config['predictions_save_path']), exist_ok=True)
    df_out = pd.DataFrame({
        "Actual_G+A_24_25": y_test.reset_index(drop=True),
        "Predicted_G+A_24_25": y_pred
    })

    if player_names_test is not None:
        df_out["Player"] = player_names_test.reset_index(drop=True)

    if "Player" in df_out.columns:
        df_out = df_out[["Player", "Actual_G+A_24_25", "Predicted_G+A_24_25"]]

    df_out.to_csv(train_config['predictions_save_path'], index=False)
    print(f"Predictions saved to {train_config['predictions_save_path']}")

    return y_pred