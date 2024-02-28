import tensorflow as tf


class InputData(tf.keras.layers.Layer):
    def __init__(self, dataset: str, is_it_file: bool):
        super().__init__()
        BATCH_SIZE = 3
        if is_it_file: dataset = self.text_is_file(dataset)

        dataset = self.punctuation_marks_processing(dataset)
        tokens_dataset = tf.Variable(dataset.split(' '))
        dataset = self.division_into_sentences(dataset)

        vectorizer = tf.keras.layers.TextVectorization(standardize=None)
        vectorizer.adapt(tokens_dataset)

        model = tf.keras.models.Sequential()
        model.add(vectorizer)

        dataset = tf.Variable(model.predict(dataset))
        n = dataset.shape[0] - BATCH_SIZE
        self.dataset_x = [dataset[i: i+BATCH_SIZE] for i in range(n)]
        self.dataset_y = [dataset[BATCH_SIZE:]]


    def get_dataset(self) -> tf.Variable:
        # Return Input-Output datasets 
        return (self.dataset_x, self.dataset_y)
    

    def division_into_sentences(self, dataset: str) -> list[list[str]]:
        # Dividing text into sentences
        sentences_data = [['']]
        cell = 0
        size_d = len(dataset)
        for i in range(size_d):
            sentences_data[cell][0] += dataset[i]
            if dataset[i] == '.' and i != size_d-1:
                sentences_data.append([''])
                cell+=1
        return sentences_data


    def punctuation_marks_processing(self, dataset: str) -> str:
        # Highlighting all punctuation marks
        dataset_processed = ''
        size_d = len(dataset)

        for char in range(size_d):
            if dataset[char] in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' and dataset[char-1] != ' ':
                dataset_processed += ' ' + dataset[char]
            else:
                dataset_processed += dataset[char]
        return dataset_processed


    def text_is_file(self, dataset: str) -> str:
        # Check: is a text a file?
        with open(dataset, 'r', encoding='utf-8') as f:
            data = f.read()
            data = data.replace('\\ufeff', '')
        return data
    
