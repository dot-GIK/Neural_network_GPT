import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class InputData():
    def __init__(self, data, is_it_file):
        self.data = data
        self.sequence_len = 0
        self.is_it_file = is_it_file

        if self.is_it_file:
            self.data = tf.Variable(self.text_preparation_file())
        else:
            self.sequence_len = len(self.data.split(' '))
            self.data = tf.Variable(self.data)

    def text_preparation_file(self):
        with open(self.data, 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.replace('\\ufeff', '')
            self.sequence_len = len(text.split())
        return text

