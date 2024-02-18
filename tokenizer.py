from tensorflow import data
from keras.layers import TextVectorization


class Tokenizer:
    def __init__(self, data, sequence_len):
        self.data = data
        self.sequence_len = sequence_len

    def __call__(self, x):
        dataset = data.Dataset.from_tensor_slices([self.data])
        tokenizer = TextVectorization(self.sequence_len, output_mode="int")
        tokenizer.adapt(dataset.batch(32))
        return tokenizer(x)

