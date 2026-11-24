import numpy as np 
import pandas as pd
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import keras
import tensorflow as tf
from src.dataset import get_dataset
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS

def get_normalization_layer(name, dataset):
    """"
    Normalize Numerical + Text 
    """
    normalizer = tf.keras.layers.Normalization(axis = None)
    feature_ds = dataset.map(lambda x, y: x[name])
    normalizer.adapt(feature_ds)
    return normalizer

def get_category_encoding_layer(name, dataset):
    onehot_encoder = tf.keras.layers.StringLookup(output_mode= "one_hot", name =f"{name}_lookup")
    feature_ds = dataset.map(lambda x,y: x [name])
    onehot_encoder.adapt(feature_ds)
    return onehot_encoder

def create_preprocessor(dataset): 
    input_dict = {}
    encoded_features = []

    for col in NUMERICAL_COLS :
        input_col = tf.keras.Input(shape = (1,), name = col, dtype= tf.float32)
        norm_layer =get_normalization_layer(col, dataset)
        encoded_col = norm_layer(input_col)
        input_dict[col] = input_col
        encoded_features.append(encoded_col)

    for col in CATEGORICAL_COLS :
        input_col = tf.keras.Input(shape =(1,), name = col, dtype = "string")
        cat_layer = get_category_encoding_layer(col, dataset)
        encoded_col = cat_layer(input_col)
        input_dict[col] = input_col
        encoded_features.append(encoded_col)

    concated_features = tf.keras.layers.concatenate(encoded_features)
    return input_dict, concated_features

# test 

if __name__ == "__main__":
    train_path = "data/processed/train.csv"
    ds_train = get_dataset(train_path, batch_size= 128, is_training =False).take(10)
    # age_normalizer = get_normalization_layer("AGE", ds_train)
    # print(age_normalizer.mean.numpy())
    # print(age_normalizer.variance.numpy())
    # print("\n--- Testing Category Encoding for CODE_GENDER ---")
    # gender_encoder = get_category_encoding_layer("CODE_GENDER", ds_train)
    # vocab = gender_encoder.get_vocabulary()
    # print("Vocabulary:", vocab)
    # test_tensor = tf.constant(["M", "F", "Alien"])
    # print("Encoded test:\n", gender_encoder(test_tensor).numpy())
    inputs_dict, features = create_preprocessor(ds_train)
    preprocessor_model = tf.keras.Model(inputs=inputs_dict, outputs=features)
    print(f"Input: {len(inputs_dict)}")
    print(f"output Shape: {features.shape[-1]}")