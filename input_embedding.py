import tensorflow as tf
from positional_encoding import PositionalEncoding
from tokenizer import Tokenizer
from keras.layers import Embedding, Dropout
from keras.layers import Layer


class InputEmbedding(Layer):
    def __init__(self, max_sequence_len: int, vector_len: int):
        super().__init__()
        self.embedding = Embedding(input_dim=max_sequence_len, output_dim=vector_len, mask_zero=True)
        self.pos_enc = PositionalEncoding(max_sequence_len, vector_len)
        self.dropout = Dropout(.1)

    def __call__(self, x):
        x = self.embedding(x)
        output = self.dropout(x + self.pos_enc())
        return x

