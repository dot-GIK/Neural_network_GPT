import tensorflow as tf
from positional_encoding import PositionalEncoding
from tokenizer import Tokenizer
from keras.layers import Embedding, Dropout


class InputEmbedding:
    def __init__(self, data, sequence_len, vector_len):
        self.tokenizer = Tokenizer(data, sequence_len)
        self.embedding = Embedding(input_dim=sequence_len, output_dim=vector_len)
        self.pos_enc = PositionalEncoding(sequence_len, vector_len)
        self.dropout = Dropout(.1)

    def __call__(self, x):
        x = self.tokenizer(x)
        x = self.embedding(x)
        output = self.dropout(x + self.pos_enc())
        return x

