import tensorflow as tf
from keras.layers import Flatten


class PositionalEncoding():
    def __init__(self, max_sequence_len: int, vector_len: int):
        self.max_sequence_len = max_sequence_len
        self.vector_len = vector_len

    def __call__(self) -> tf.Tensor:
        i_even = tf.range(0, self.vector_len, delta=2, dtype=tf.float32)
        i_odd = tf.range(1, self.vector_len, delta=2, dtype=tf.float32)

        denominator = tf.math.pow(10000, i_even/self.vector_len )

        position = tf.range(0, self.max_sequence_len, dtype=tf.float32)
        position = tf.reshape(position, [self.max_sequence_len, 1])
        
        i_even = tf.math.sin(position/denominator)
        i_odd = tf.math.cos(position/denominator)

        x = tf.stack([i_even, i_odd], axis=2)
        x = Flatten()(x)
        return x

