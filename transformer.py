import tensorflow as tf
# from input_embedding import InputEmbedding


# class Transformer(tf.keras.Model):
#     def __init__(self, data, sequence_len, vector_len):
#         super(Transformer, self).__init__()
#         self.input_embedding = InputEmbedding(data, sequence_len, vector_len)

#     def call(self, x):
#         x = self.input_embedding(x)
#         return x
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())