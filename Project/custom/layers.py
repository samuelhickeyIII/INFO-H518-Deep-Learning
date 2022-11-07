import tensorflow as tf
import math as m
from tensorflow import keras
import numpy as np
import math

def positional_encoding(seq, embedding) -> tf.Tensor:
    depth = embedding/2

    positions = np.arange(seq)[:, np.newaxis]
    depths = np.arange(embedding)[np.newaxis, :] / depth

    angle_rads = positions * (1 / 10000**depths)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=np.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_embedding, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.d_embedding = d_embedding
        self.max_seq = kwargs.get('max_seq', 2048)
        self.embedding = keras.layers.Embedding(vocab_size, d_embedding)
        self.pos_encoding = positional_encoding(self.max_seq, d_embedding)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        inputs = self.embedding(inputs)
        inputs *= tf.math.sqrt(tf.cast(self.d_embedding, tf.float32))
        return tf.add(inputs + self.pos_encoding[:, :length, :])
