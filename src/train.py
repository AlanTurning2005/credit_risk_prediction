import sys
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import tensorflow as tf
from src.dataset import get_dataset
from src.model import build_model
from src.config import BATCH_SIZE


def calculate_class_weights_dynamic(train_csv_path="data/processed/train.csv"):

    df_target = pd.read_csv(train_csv_path, usecols=['TARGET'])
    counts = df_target['TARGET'].value_counts()
    
    total_neg = counts[0]
    total_pos = counts[1]
    total = total_neg + total_pos
    weight_for_0 = (1 / total_neg) * (total / 2.0)
    weight_for_1 = (1 / total_pos) * (total / 2.0)
    
    return {0: weight_for_0, 1: weight_for_1}

def train(epochs=20):
    train_path = "data/processed/train.csv"
    valid_path = "data/processed/valid.csv"
    model_save_dir = Path("models")
    model_save_dir.mkdir(exist_ok=True)
    best_model_path = model_save_dir / "best_model.keras"

    print("Nạp dữ liệu Train và Validation")
    # Dataset dùng để adapt tiền xử lý
    adapt_ds = get_dataset(train_path, batch_size=2048, is_training=False).take(100)
    
    # Dataset chính để huấn luyện
    train_ds = get_dataset(train_path, batch_size=BATCH_SIZE, is_training=True)
    valid_ds = get_dataset(valid_path, batch_size=BATCH_SIZE, is_training=False)

    print("Khởi tạo mô hình...")
    model = build_model(adapt_ds, learning_rate=1e-3)

    print("Thiết lập Callbacks & Class Weights...")
    class_weights = calculate_class_weights_dynamic()
    print("Class weights:", class_weights)

    callbacks = [
        # Dừng sớm nếu sau 5 epoch val_auc không cải thiện
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Lưu file mô hình tốt nhất
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        # Giảm learning rate khi loss chững lại
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    print("Bắt đầu huấn luyện ")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    print(f"\nMô hình tốt nhất đã được lưu tại: {best_model_path}")
    return history


if __name__ == "__main__":
    train(epochs=20)