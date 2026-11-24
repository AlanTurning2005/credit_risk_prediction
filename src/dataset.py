import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
import pandas as pd 
import numpy as np

from src.config import BATCH_SIZE # (1024)
from src.config import SHUFFLE_BUFFER_SIZE #(5000)
from src.config import RECORD_DEFAULTS, CSV_COLUMNS, TARGET_COL
def parse_csv_line(line): 
    """
    input: Nhan 1 dong csv, so dong 
    output: tuple chua features_dict, target
    
    su dung encode_csv

    """
    fields = tf.io.decode_csv(line, record_defaults= RECORD_DEFAULTS)

    features_dict = dict(zip(CSV_COLUMNS, fields))
    target = features_dict.pop(TARGET_COL)

    return features_dict, target
    pass

def get_dataset(file_pattern, batch_size = BATCH_SIZE, is_training =True) : 
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle = is_training)
    dataset = dataset.interleave(
        lambda file_path: tf.data.TextLineDataset(file_path).skip(1),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(parse_csv_line, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training == True: 
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)

    dataset = dataset.batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)




if __name__ == "__main__":
    print("--- TESTING TF.DATA PIPELINE ---")
    train_path = "data/processed/train.csv"
    
    if Path(train_path).exists():
        ds = get_dataset(train_path, batch_size=4, is_training=False)
        for X_batch, y_batch in ds.take(1):
            print("Batch loaded successfully!")
            print(f"Number of features in batch: {len(X_batch)}")
            print(f"Target y shape: {y_batch.shape}")
            print(f"Sample 'AGE' tensor: {X_batch['AGE']}")
    else:
        print(f"File not found: {train_path}")