from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from airt.keras.layers import MonoDense

def CMNN(input_shape=(1,), monotonicity_indicator=[1]):
    return Sequential([
                    Input(shape=input_shape, name='input_1'),
                    MonoDense(128, activation="relu", monotonicity_indicator=monotonicity_indicator),
                    MonoDense(128, activation="relu"),
                    MonoDense(128, activation="relu"),
                    MonoDense(1)
                ])