import torch.nn as nn
import tensorflow as tf

def Network_tf(width=128, input_shape=(1,)):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(width, activation='relu', input_shape=input_shape),  # Input Layer
        tf.keras.layers.Dense(width, activation='relu'),  # Hidden layer 1
        tf.keras.layers.Dense(width, activation='relu'),  # Hidden layer 2
        tf.keras.layers.Dense(1)  # Output layer
    ])