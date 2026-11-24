import sys
from pathlib import Path

# Thêm đường dẫn gốc để import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import tensorflow as tf
from src.keras_preprocessing import create_preprocessor

def build_model( train_ds, learning_rate = 1e-3):
    inputs_dict, features = create_preprocessor(train_ds)
    x = tf.keras.layers.Dense(128, activation ="relu")(features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(64, activation= 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs=inputs_dict, outputs=outputs, name="CreditRiskModel")
    
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate = learning_rate),
        loss = tf.losses.BinaryCrossentropy(),
        metrics = [            
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
            ]
    )

    return model

if __name__ == "__main__":
    from src.dataset import get_dataset

    print("--- Test khoi tao Model ---")
    train_path = "data/processed/train.csv"
    ds_train = get_dataset(train_path, batch_size=128, is_training=False).take(10)

    model = build_model(ds_train)
    model.summary() 