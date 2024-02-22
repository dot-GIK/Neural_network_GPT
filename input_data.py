import tensorflow as tf


class InputData(tf.keras.layers.Layer):
    def __init__(self, dataset: str, BATCHES: int, is_it_file: bool):
        super().__init__()
        if is_it_file: self.dataset = self.text_preparation_file(dataset)
        else: self.dataset = tf.data.Dataset.from_tensor_slices([dataset])

        vectorizer = tf.keras.layers.TextVectorization()
        vectorizer.adapt(self.dataset.batch(BATCHES))

        model = tf.keras.models.Sequential()
        model.add(vectorizer)
        self.dataset = tf.Variable([model.predict(self.dataset)])

    def get_dataset(self) -> tf.Variable:
        return self.dataset

    def text_preparation_file(self, dataset: str) -> tf.data.Dataset:
        with open(dataset, 'r', encoding='utf-8') as f:
            data = f.read()
            data = data.replace('\\ufeff', '')
        return tf.data.Dataset.from_tensor_slices([data])
    
