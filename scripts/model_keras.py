import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
# No need to import load_model here as it's not used for saving
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define a custom Keras Model by subclassing tf.keras.Model
class CustomMLP(Model):
    def __init__(self, input_dim, layer_configs, output_units):
        super(CustomMLP, self).__init__()
        self.hidden_layers = []
        for i, config in enumerate(layer_configs):
            # Dynamically create layers based on config
            self.hidden_layers.append(layers.Dense(config['units'], name=f'dense_{i+1}'))
            self.hidden_layers.append(layers.BatchNormalization(name=f'bn_{i+1}'))
            self.hidden_layers.append(layers.Activation(config['activation'], name=f'activation_{i+1}'))
            self.hidden_layers.append(layers.Dropout(config['dropout'], name=f'dropout_{i+1}'))

        self.output_layer = layers.Dense(output_units, name='output_layer')

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            # Pass training flag for BatchNormalization and Dropout
            if isinstance(layer, (layers.BatchNormalization, layers.Dropout)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return self.output_layer(x)

def train_and_evaluate_keras(X_train, y_train, X_test, y_test, player_names_test=None,
                             model_config=None, train_config=None):
    if model_config is None:
        raise ValueError("model_config must be provided.")
    if train_config is None:
        raise ValueError("train_config must be provided.")

    input_dim = X_train.shape[1]

    # Instantiate our custom model with parameters from config
    model = CustomMLP(
        input_dim=input_dim,
        layer_configs=model_config['layers'],
        output_units=model_config['output_units']
    )

    # Configure optimizer and loss from config
    optimizer = Adam(learning_rate=model_config['learning_rate'])
    model.compile(optimizer=optimizer, loss=model_config['loss'])

    callbacks = []
    if 'early_stopping' in train_config['callbacks']:
        es_config = train_config['callbacks']['early_stopping']
        callbacks.append(EarlyStopping(
            monitor=es_config['monitor'],
            patience=es_config['patience'],
            restore_best_weights=es_config['restore_best_weights']
        ))
    if 'reduce_lr_on_plateau' in train_config['callbacks']:
        lrp_config = train_config['callbacks']['reduce_lr_on_plateau']
        callbacks.append(ReduceLROnPlateau(
            monitor=lrp_config['monitor'],
            factor=lrp_config['factor'],
            patience=lrp_config['patience'],
            verbose=lrp_config['verbose'],
            min_lr=lrp_config['min_lr']
        ))

    # Build the model by calling it on dummy input to create weights
    model.build(input_shape=(None, input_dim))
    model.summary() # Print model summary to verify structure

    model.fit(
        X_train, y_train,
        validation_split=train_config['validation_split'],
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    # Predict
    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Keras Model MAE (on 24/25 set): {mae:.2f}")

    os.makedirs(os.path.dirname(train_config['model_save_path']), exist_ok=True)
    # FIX: Change saving format for subclassed models to TensorFlow SavedModel format
    model.save(train_config['model_save_path'], save_format="tf")
    print(f"Model saved to {train_config['model_save_path']} in TensorFlow SavedModel format.")

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