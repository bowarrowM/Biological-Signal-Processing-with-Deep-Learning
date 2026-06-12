"""
Model architecture.

build_model() is the only public interface. Swap or extend the architecture
here without touching data loading or training logic.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional, Conv1D, Dense,
    Dropout, Input, LSTM, MaxPooling1D, MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from config import load_config

cfg = load_config()


def build_model(input_shape: tuple, n_classes: int) -> Model:
    reg       = l2(cfg["model"]["l2_reg"])
    dr        = cfg["model"]["dropout_rate"]
    attn_heads   = cfg["model"]["attention_heads"]
    attn_key_dim = cfg["model"]["attention_key_dim"]

    inp = Input(shape=input_shape)

    x = Conv1D(64,  kernel_size=3, activation="relu", kernel_regularizer=reg)(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dr)(x)

    x = Conv1D(128, kernel_size=3, activation="relu", kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dr)(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(dr)(x)

    x = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim)(x, x)
    x = Dropout(dr)(x)

    x = Bidirectional(LSTM(64))(x)
    x = Dropout(dr)(x)

    out = Dense(n_classes, activation="softmax", kernel_regularizer=reg)(x)

    return Model(inputs=inp, outputs=out)
