import tensorflow as tf
from errors import TheRemainderOfTheDivision, TheQuotientIsNotDividedIntoTwo
from keras.layers import Dropout
import numpy as np
import keras


class DataPreparation():
    def __init__(self, data: str, batch_size: int, buffer_size: int, is_it_file: bool):
      '''
      Creating a dataset for training the Transformer model.
      '''

      self.BATCH_SIZE = batch_size
      self.BUFFER_SIZE= buffer_size
      
      # Checking Batch_size and Buffer_size
      if self.BUFFER_SIZE % self.BATCH_SIZE != 0:
        raise TheRemainderOfTheDivision(f'Размер датасета(buffer_size) должен быть кратен размеру партии(batch_size). ERROR ваш остаток: {self.BUFFER_SIZE % self.BATCH_SIZE}')
      elif (self.BUFFER_SIZE // self.BATCH_SIZE) % 2 != 0:
        raise TheQuotientIsNotDividedIntoTwo(f'Размер датасета(buffer_size) должен быть кратен размеру партии(batch_size) и частное дложно быть кратно двум. ERROR ваше частное: {self.BUFFER_SIZE // self.BATCH_SIZE}') 
      
      # Variable data is a file
      if is_it_file: data = self.text_is_file(data)      

      data = self.punctuation_marks_processing(data)
      data = data.split('.')
      print(f'Количество предложений в датасете: {len(data)}')

      dataset = ''
      for i in range(self.BUFFER_SIZE):
        dataset += ' ' + data[i] + ' .'
      
      # Creating datasets
      self.tokens_dataset = ['[START] ' + '[END] ' + dataset]
      # self.dataset, self.dataset_y_input, self.dataset_y_output, size_of_largest_sent = self.creating_datasets_from_sentences(dataset)
      self.dataset, self.dataset_y_input, self.dataset_y_output, size_of_largest_sent = self.creating_datasets_from_phrases(dataset)

      print(f'Размер самого большого предложения: {size_of_largest_sent}')

      # Creating Tokenizer
      self.vectorize = tf.keras.layers.TextVectorization(standardize=None, output_sequence_length=self.BATCH_SIZE)
      self.vectorize.adapt(self.tokens_dataset)

      self.model = tf.keras.models.Sequential()
      self.model.add(self.vectorize)

    def encoder_tks(self) -> tf.data.Dataset:
      '''
      Processing datasets for further tokenization and creating final dataset.
      '''

      self.dataset = tf.Variable(self.model.predict(self.dataset))
      self.dataset_y_input = tf.Variable(self.model.predict(self.dataset_y_input))
      self.dataset_y_output = tf.Variable(self.model.predict(self.dataset_y_output))

      print(self.dataset, '\n', self.dataset_y_input, '\n', self.dataset_y_output)

      final_dataset = tf.data.Dataset.from_tensor_slices((([self.dataset], [self.dataset_y_input]), [self.dataset_y_output]))
      final_dataset = final_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
      return final_dataset

    def get_sentence(self, sentence: str) -> np.array:
      '''
      Processing the entered sentence.
      '''

      if sentence[-1] != '.': sentence += '.'
      sentence = self.punctuation_marks_processing(sentence)
      sentence = [['[START] ' + sentence + ' [END]']]
      return self.model.predict(sentence)

    def creating_datasets_from_sentences(self, data: str) -> tuple[list[list[str]],list[list[str]],list[list[str]], int]:
      data = data.split('.')
      dataset, dataset_y_input, dataset_y_output  = [], [], []
      flag = 0
      size_of_largest_sent = 0
      for i in range(0, len(data)): # длинна самого большого предложения
        if flag == 0 and i != len(data) - 1:
           dataset.append(['[START] ' + data[i] + ' . [END]'])
           flag = 1
        elif i != len(data) - 1:
          dataset_y_input.append(['[START] ' + data[i] + ' .'])
          dataset_y_output.append([data[i] + ' . [END]'])
          flag = 0

      if len(data) > size_of_largest_sent:
        size_of_largest_sent = len(data)

      return dataset, dataset_y_input, dataset_y_output, size_of_largest_sent

    def creating_datasets_from_phrases(self, data: str):
      pass


    def punctuation_marks_processing(self, dataset: str) -> str:
        '''
        Selecting all punctuation marks for further text processing.
        '''

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
        '''
        Does it check that the text is a file?
        '''

        with open(dataset, 'r', encoding='utf-8') as f:
            data = f.read()
            data = data.replace('\\ufeff', '')
        if data[-1] != '.': data += '.'
        return data
    

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    
    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)
    self.dropout = Dropout(.1)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = self.dropout(x + self.pos_encoding[tf.newaxis, :length, :])
    return x


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)
    
    self.last_attn_scores = attn_scores
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])

    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x
  

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x


class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
    super().__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    x = self.pos_embedding(x) 
    x = self.dropout(x)
    for i in range(self.num_layers):
        x = self.enc_layers[i](x)
    return x


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)
    
    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)
    self.last_attn_scores = self.cross_attention.last_attn_scores
    x = self.ffn(x)  
    return x
  

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [DecoderLayer(d_model=d_model, num_heads=num_heads,dff=dff, dropout_rate=dropout_rate) for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    x = self.pos_embedding(x) 
    x = self.dropout(x)
    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)
    self.last_attn_scores = self.dec_layers[-1].last_attn_scores
    return x
  

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    context, x  = inputs
    context = self.encoder(context) 
    x = self.decoder(x, context) 

    logits = self.final_layer(x)  

    try:
      del logits._keras_mask
    except AttributeError:
      pass

    return logits


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  

def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred
  mask = label != 0
  match = match & mask
  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

class Model(tf.Module):
  def __init__(self, transformer, datasets, training_dataset, epochs):
    super().__init__()
    self.epochs = epochs
    self.datasets = datasets
    self.training_dataset = training_dataset
    self.transformer = transformer

  def __call__(self, sentence, max_length):
    
    start = tf.constant(self.datasets.get_token_dictionary().index('[START]'), dtype=tf.int64)[tf.newaxis]
    end = tf.constant(self.datasets.get_token_dictionary().index('[END]'), dtype=tf.int64)[tf.newaxis]

    sentence = self.datasets.get_sentence(sentence)
    print(sentence)

    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([sentence, output], training=False)
      predictions = predictions[:, -1:, :] 
      predicted_id = tf.argmax(predictions, axis=-1)
      output_array = output_array.write(i+1, predicted_id[0])
      if predicted_id == end:
        break
    output = tf.transpose(output_array.stack())
    print(output)
    result = ''
    for i in range(output.shape[1]):
      token = int(output[0][i])
      result += ' ' + self.datasets.get_token_dictionary()[token]
    return result
    
  

def main():
    sentence = 'Правда ли, что Шаимов лох'
    datasets = DataPreparation('dataset.txt', 60, 240, True)
    training_data = datasets.encoder_tks()


    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=20000,
    target_vocab_size=20000,
    dropout_rate=dropout_rate)

    transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])
    transformer.fit(training_data, epochs=1000)

    model = Model(transformer, datasets, training_data, 1000)
    print(model(sentence, 30))

if __name__ == '__main__':
    main()