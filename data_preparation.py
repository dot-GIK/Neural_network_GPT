import tensorflow as tf
from errors import DatasetSizeSmallerThenBatchSize


class DataPreparation():
    def __init__(self, dataset: str, size_sentence: int, batch_size: int, max_tokens: int, is_it_file: bool):
      self.size_sentence = size_sentence
      self.BATCH_SIZE = batch_size
      self.MAX_TOKENS = max_tokens

      if is_it_file: dataset = self.text_is_file(dataset)
      if dataset[-1] != '.': dataset += '.'

      dataset = self.punctuation_marks_processing(dataset)
      self.tokens_dataset = ['[START] ' + '[END] ' + dataset]
      dataset = self.size_of_sentences_and_division_into_sentences(dataset)
      self.dataset, self.dataset_y_input, self.dataset_y_output = self.preparing_datasets(dataset)

      self.vectorize = tf.keras.layers.TextVectorization(standardize=None, output_sequence_length=self.MAX_TOKENS)
      self.vectorize.adapt(self.tokens_dataset)

      self.model = tf.keras.models.Sequential()
      self.model.add(self.vectorize)

    def encoder_tks(self) -> tuple[tf.Tensor, tf.Tensor]:
      print(self.dataset, '\n', self.dataset_y_input, '\n', self.dataset_y_output)

      self.dataset = tf.Variable(self.model.predict(self.dataset))
      self.dataset_y_input = tf.Variable(self.model.predict(self.dataset_y_input))
      self.dataset_y_output = tf.Variable(self.model.predict(self.dataset_y_output))
      
      if self.dataset.shape[0] < self.size_sentence:
          raise DatasetSizeSmallerThenBatchSize(f'Размер датасета(dataset.shape): {self.dataset.shape} меньше, чем установленное количество предложений(size_sentence): {self.size_sentence}')
      elif self.dataset.shape[0] > self.size_sentence:
          self.dataset = self.dataset[:self.size_sentence]     
          self.dataset_y_input = self.dataset_y_input[:self.size_sentence]
          self.dataset_y_output = self.dataset_y_output[:self.size_sentence]

      n = self.dataset.shape[0] - self.BATCH_SIZE
      self.dataset = [self.dataset[i: i+self.BATCH_SIZE] for i in range(0,n,self.BATCH_SIZE)][0]
      self.dataset_y_input = [self.dataset_y_input[i: i+self.BATCH_SIZE] for i in range(self.BATCH_SIZE, self.dataset_y_input.shape[0], self.BATCH_SIZE)][0]
      self.dataset_y_output = [self.dataset_y_output[i: i+self.BATCH_SIZE] for i in range(self.BATCH_SIZE, self.dataset_y_output.shape[0], self.BATCH_SIZE)][0]

      print(self.dataset, '\n', self.dataset_y_input, '\n', self.dataset_y_output)
      max_dataset = tf.data.Dataset.from_tensor_slices((([self.dataset], [self.dataset_y_input]), [self.dataset_y_output]))
      max_dataset = max_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) #.shuffle(50000)
      return max_dataset

    def get_sentence(self, sentence):
      if sentence[-1] != '.': sentence += '.'
      sentence = self.punctuation_marks_processing(sentence)
      sentence = [['[START] ' + sentence + ' [END]']]
      return tf.Variable(self.model.predict(sentence))
    
    def get_token_dictionary(self):
      return self.vectorize.get_vocabulary()

    def preparing_datasets(self, dataset: list) -> tuple[list[list[str]],list[list[str]],list[list[str]]]:
      dataset_y_input = []
      dataset_y_output = []
      size_d = len(dataset)
      for i in range(size_d):
        if i == 0: dataset_y_input.append(['[START] '+dataset[i][0]])
        else: dataset_y_input.append(['[START]'+dataset[i][0]])

        dataset_y_output.append([dataset[i][0]+' [END]'])
          
        if i == 0: dataset[i][0] = '[START] ' + dataset[i][0] + ' [END]'
        else: dataset[i][0] = '[START]' + dataset[i][0] + ' [END]'

      return dataset, dataset_y_input, dataset_y_output

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

    def size_of_sentences_and_division_into_sentences(self, dataset: str) -> list[list]:
      dataset_preprocessed = []
      data = ['']
      size_d = len(dataset)
      for i in range(size_d):
          data[0] += dataset[i]
          if dataset[i] == '.' and len(data[0].split(' ')) <= self.MAX_TOKENS-3:
            dataset_preprocessed.append(data)
            data = ['']
      return dataset_preprocessed  
    
    def text_is_file(self, dataset: str) -> str:
        # Check: is a text a file?
        with open(dataset, 'r', encoding='utf-8') as f:
            data = f.read()
            data = data.replace('\\ufeff', '')
        return data
    
