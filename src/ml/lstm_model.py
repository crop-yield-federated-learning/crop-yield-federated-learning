from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_lstm_model(input_features=38):
    """
    Build the LSTM regression model for crop-yield prediction.

    Input:
        One timestep containing 38 processed features.

    Output:
        Predicted crop yield in kg/hectare.
    """

    model = keras.Sequential([
        layers.Input(shape=(1, input_features)),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_tabular_mlp_model(input_features=38, seed=42):
    """Build a regularized feed-forward regressor for a 38-value feature row.

    The dataset contains independent tabular examples rather than ordered
    timesteps. This architecture therefore models the input as one feature
    vector instead of reshaping it into a synthetic sequence.
    """
    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(input_features,), name="processed_features")
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(inputs)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    outputs = layers.Dense(1, name="scaled_yield")(x)
    model = keras.Model(inputs, outputs, name="crop_yield_tabular_mlp")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model
