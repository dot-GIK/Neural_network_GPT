import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras.layers import Flatten
from keras.models import Sequential
from keras import Model as M

class PositionalEncoding(tf.Module):
    def __init__(self, sequence_length=10, w_vector_length=6):
        super().__init__()
        self.sequence_length = sequence_length
        self.w_vector_length = w_vector_length 
        
    def __call__(self):
        i_even = tf.range(0, self.w_vector_length, delta=2, dtype=tf.float32)
        i_odd = tf.range(1, self.w_vector_length, delta=2, dtype=tf.float32)

        denominator = tf.math.pow(10000, i_even/self.w_vector_length )

        position = tf.range(0, self.sequence_length, dtype=tf.float32)
        position = tf.reshape(position, [self.sequence_length, 1])
        
        i_even = tf.math.sin(position/denominator)
        i_odd = tf.math.cos(position/denominator)

        stacked = tf.stack([i_even, i_odd], axis=2)
        stacked = Flatten()(stacked)
        return stacked