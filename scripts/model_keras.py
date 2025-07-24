import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

def train_and_evaluate_keras(X_train, y_train, X_test, y_test,df_test_raw=None):
    input_dim = X_train.shape[1]

    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)  # output layer for regression
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    # Predict
    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n📊 Keras Model MAE (on 24/25 set): {mae:.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/keras_ga_model.h5")

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
