import tensorflow as tf
from errors import DatasetSizeSmallerThenBatchSize


class DataPreparation():
    def __init__(self, dataset: str, batch_size: int, is_it_file: bool):
        self.BATCH_SIZE = batch_size
        MAX_TOKENS=60
        if is_it_file: dataset = self.text_is_file(dataset)

        dataset = self.punctuation_marks_processing(dataset)
        self.dataset, tokens_dataset, self.dataset_y_input, self.dataset_y_output = self.division_into_sentences(dataset)

        self.vectorize = tf.keras.layers.TextVectorization(standardize=None, output_sequence_length=MAX_TOKENS)
        self.vectorize.adapt(tokens_dataset)

        self.model = tf.keras.models.Sequential()
        self.model.add(self.vectorize)


    def encoder_tks(self) -> tf.data.Dataset:
        self.dataset = tf.Variable(self.model.predict(self.dataset))
        self.dataset_y_input = tf.Variable(self.model.predict(self.dataset_y_input))
        self.dataset_y_output = tf.Variable(self.model.predict(self.dataset_y_output))

        if self.dataset.shape[0] < self.BATCH_SIZE*2:
            raise DatasetSizeSmallerThenBatchSize(f'Размер датасета(dataset.shape): {self.dataset.shape} меньше, чем удвоенный размер партии(batch_size): {self.BATCH_SIZE*2}')
        elif self.dataset.shape[0] > self.BATCH_SIZE*2:
            self.dataset = self.dataset[:self.BATCH_SIZE*2]     
            self.dataset_y_input = self.dataset_y_input[:self.BATCH_SIZE*2]
            self.dataset_y_output = self.dataset_y_output[:self.BATCH_SIZE*2]

        n = self.dataset.shape[0] - self.BATCH_SIZE
        self.dataset = [self.dataset[i: i+self.BATCH_SIZE] for i in range(0,n,self.BATCH_SIZE)][0]
        self.dataset_y_input = [self.dataset_y_input[i: i+self.BATCH_SIZE] for i in range(self.BATCH_SIZE, self.dataset_y_input.shape[0], self.BATCH_SIZE)][0]
        self.dataset_y_output = [self.dataset_y_output[i: i+self.BATCH_SIZE] for i in range(self.BATCH_SIZE, self.dataset_y_output.shape[0], self.BATCH_SIZE)][0]

        max_dataset = tf.data.Dataset.from_tensor_slices((([self.dataset], [self.dataset_y_input]), [self.dataset_y_output]))
        max_dataset = max_dataset.shuffle(20000).prefetch(buffer_size=tf.data.AUTOTUNE)
        return max_dataset


    def get_dataset_shape(self) -> tuple[int, int]:
        return self.dataset.shape
    

    def division_into_sentences(self, dataset: str) -> tuple[list[list[str]], tf.Tensor]:
        # Dividing text into sentences
        if dataset[-1] != '.': dataset += ' .'
        cell = 0
        size_d = len(dataset)
        sentences_data = [['[START] ']]
        tokens_dataset = ['[START] ']
        dataset_y_input = [['[START] ']]
        dataset_y_output = [['']]

        for i in range(size_d): 
            tokens_dataset[0] += dataset[i]
            sentences_data[cell][0] += dataset[i]
            dataset_y_input[cell][0] += dataset[i]
            dataset_y_output[cell][0] += dataset[i]

            if dataset[i] == '.' and i != size_d-1:
                dataset_y_output[cell][0] += ' [END]'
                tokens_dataset[0] += ' [END] [START]'
                sentences_data[cell][0] += ' [END]'
                dataset_y_input.append(['[START] '])
                dataset_y_output.append([''])
                sentences_data.append(['[START] '])
                cell+=1

            elif i == size_d-1: 
                dataset_y_output[cell][0] += ' [END]'
                tokens_dataset[0] += ' [END]'
                sentences_data[cell][0] += ' [END]'

        return sentences_data, tokens_dataset, dataset_y_input, dataset_y_output


    def punctuation_marks_processing(self, dataset: str) -> str:
        # Highlighting all punctuation marks
        dataset_processed = ''
        size_d = len(dataset)  
        for char in range(size_d):
            if (dataset[char] in '!#$%&*+,-.)]}/:";<=>?@\\^_`|~\t\n' or dataset[char] == "'") and dataset[char-1] != ' ':
                dataset_processed += ' ' + dataset[char]
            elif (dataset[char] in '!#$%&*+,-.([{/":;<=>?@\\^_`|~\t\n' or dataset[char] == "'") and char != size_d-1:
                if dataset[char+1] != ' ':
                    dataset_processed += dataset[char] + ' '
            else:
                dataset_processed += dataset[char]
        return dataset_processed


    def text_is_file(self, dataset: str) -> str:
        # Check: is a text a file?
        with open(dataset, 'r', encoding='utf-8') as f:
            data = f.read()
            data = data.replace('\\ufeff', '')
        return data