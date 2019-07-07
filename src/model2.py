import tensorflow as tf
import re
import unicodedata
import numpy as np
import matplotlib.pyplot as plt

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w

class Encoder(tf.keras.Model):
    def __init__(self, num_units):
        super(Encoder, self).__init__()
        self.units = num_units
        self.seq = tf.keras.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = num_units, return_sequences=True, recurrent_initializer='glorot_uniform'), merge_mode='concat'),
            tf.keras.layers.LSTM(units = num_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
            ResidualLSTM(units = num_units),
            ResidualLSTM(units = num_units),
            ResidualLSTM(units = num_units),
            ResidualLSTM(units = num_units),
            ResidualLSTM(units = num_units),
            ResidualLSTM(units = num_units)
            ])

    def call(self, input_tensor):
        output_tensor = self.seq(input_tensor)

        return output_tensor

class ResidualLSTM(tf.keras.Model):
    def __init__(self, units, return_state = False, actual_units=-1):
        super(ResidualLSTM, self).__init__()
        self.lstm_layer = tf.keras.layers.LSTM(units = units, return_sequences=True, return_state=return_state, recurrent_initializer='glorot_uniform')
        self.return_state = return_state
        if actual_units == -1:
            self.num_actual_units = units
        else:
            self.num_actual_units = actual_units

    def call(self, input_tensor, initial_state = 0):
        if (self.return_state):
            output_tensor, state_h, state_c = self.lstm_layer(input_tensor)
        else:
            output_tensor = self.lstm_layer(input_tensor)

        actual_input_tensor = input_tensor[:, :, :self.num_actual_units]

        residual_ouput_tensor = tf.add(output_tensor, actual_input_tensor)
        # print('input tensor: {}, output tensor: {}, residual tensor: {}'.format(input_tensor.shape, output_tensor.shape, residual_ouput_tensor.shape))
        if (self.return_state):
            return residual_ouput_tensor, state_h, state_c
        else:
            return residual_ouput_tensor

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, num_units, vocab_size):
        super(Decoder, self).__init__()
        
        self.units = num_units

        self.seq_list = [tf.keras.layers.LSTM(units = num_units, return_sequences=True, return_state = True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.LSTM(units = num_units, return_sequences=True, return_state = True, recurrent_initializer='glorot_uniform'),
            ResidualLSTM(units = num_units, return_state = True),
            ResidualLSTM(units = num_units, return_state = True),
            ResidualLSTM(units = num_units, return_state = True),
            ResidualLSTM(units = num_units, return_state = True),
            ResidualLSTM(units = num_units, return_state = True),
            tf.keras.layers.LSTM(units = num_units, return_sequences=True, return_state = True, recurrent_initializer='glorot_uniform'),
            ]

        self.dense = tf.keras.layers.Dense(vocab_size)

        self.softmax = tf.keras.layers.Softmax()

        self.attention = BahdanauAttention(num_units)
        # self.attention = tf.keras.layers.AdditiveAttention()

    def initialize_hidden_states(self, batch_size):
        hidden_state_list = [[tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))],
                            [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))],
                            [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))],
                            [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))],
                            [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))],
                            [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))],
                            [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))],
                            [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]
                            ]

        return hidden_state_list

    def initialize_hidden_state(self, batch_size):
        hidden_state = tf.zeros((batch_size, 1, self.units))
        return hidden_state

    def call(self, enc_output, dec_input, hidden_states):
        
        context_vector, attention_weights = self.attention(hidden_states[0][0], enc_output)
        context_vector = tf.expand_dims(context_vector, 1)

        for i, layer in enumerate(self.seq_list):
            # print('Context_vector: {}'.format(context_vector.shape))
            # print('Decoder Input: {}'.format(dec_input.shape))
            dec_input = tf.concat([dec_input, context_vector], axis=-1)
            # output, state = layer(dec_input, initial_state = hidden_states[i])

            # print('Loop {} shape: {}'.format(i, dec_input.shape))

            output, state_h, state_c = layer(dec_input)
            # print('Loop {} shape: {}'.format(i, output.shape))
            # print('Loop {} shape: {}'.format(i, state_h.shape))
            # print('Loop {} shape: {}'.format(i, state_c.shape))

            # hidden_states[i] = tf.reshape(output, [-1, self.units])
            hidden_states[i] = [state_h, state_c]

            dec_input = output 
            # print(output.shape)

        output = self.dense(output)

        output = self.softmax(output)

        return output, hidden_states, attention_weights


class GNMT:
    def __init__(self, embedding_dim, num_units, src_tknzr, trg_tknzr):

        self.source_tokenizer = src_tknzr
        self.target_tokenizer = trg_tknzr

        self.src_vocab_size = len(self.source_tokenizer.word_index) + 1
        self.trg_vocab_size = len(self.target_tokenizer.word_index) + 1
        
        self.enc_embedding = tf.keras.layers.Embedding(self.src_vocab_size, embedding_dim[0])
        self.dec_embedding = tf.keras.layers.Embedding(self.trg_vocab_size, embedding_dim[1])
        self.encoder = Encoder(num_units[0])
        self.decoder = Decoder(num_units[1], self.trg_vocab_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


    @tf.function
    def train(self, inputs, targets):
    # def train(self, inputs, targets):

        with tf.GradientTape() as tape:

            batch_size = inputs.shape[0]

            embedded = self.enc_embedding(inputs)
            # print(embedded.shape)
            encoded = self.encoder(embedded)
            # print(encoded.shape)

            dec_input_token = tf.expand_dims([self.target_tokenizer.word_index['<start>']] * batch_size, 1)
            # hidden_state = self.decoder.initialize_hidden_state(batch_size)
            hidden_states = self.decoder.initialize_hidden_states(batch_size)

            loss = 0

            for t in range(targets.shape[1]):
                dec_input = self.dec_embedding(dec_input_token) 

                # print('dec_input: {}'.format(dec_input.shape))

                decoded, hidden_states, _ = self.decoder(encoded, dec_input, hidden_states)

                # for state in hidden_states:
                #     print(state[0].shape)

                # print('Decoded: {}'.format(decoded.shape))
                # print('Target[:, t]: {}'.format(targets[:, t].shape))
                loss += self.loss_function(targets[:, t], decoded)

                # no teacher forcing
                # dec_input_token = tf.argmax(decoded).numpy()

                # using teacher forcing
                dec_input_token = tf.expand_dims(targets[:, t], axis=-1)
                # print('dec_input_token: {}'.format(dec_input_token.shape))



        batch_loss = (loss / int(targets.shape[1]))

        variables = self.enc_embedding.trainable_variables + self.encoder.trainable_variables + self.dec_embedding.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    # def test(self, sentence, max_length_targ, max_length_inp):

    #     return result, sentence, attention_plot

    # def plot_attention(self, attention, sentence, predicted_sentence):


    # def translate(self, sentence, max_length_targ, max_length_inp):

    def loss_function(self, real, pred):

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_obj(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def translate(self, sentence, trg_len):
        result, sentence, attention_plot = self.test(sentence, trg_len)

        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        self.plot_attention(attention_plot, sentence.split(' '), result.split(' '))

        return

    def plot_attention(self, attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)
        cax = ax.matshow(attention, cmap='viridis')
        fig.colorbar(cax)

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

        plt.show()

    def test(self, sentence, trg_len=30):

        sentence = preprocess_sentence(sentence)

        inputs = [self.source_tokenizer.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=len(sentence),
                                                               padding='post')

        attention_plot = np.zeros((trg_len, len(sentence)))

        inputs = tf.convert_to_tensor(inputs)
        # inputs = tf.expand_dims(inputs, 1)

        # print('inputs: {}'.format(inputs.shape))

        result = ''

        batch_size = inputs.shape[0]

        embedded = self.enc_embedding(inputs)
        encoded = self.encoder(embedded)
        # print('embedded: {}'.format(embedded.shape))
        # print('encoded: {}'.format(encoded.shape))

        dec_input_token = tf.expand_dims([self.target_tokenizer.word_index['<start>']] * batch_size, 1)
        # hidden_state = self.decoder.initialize_hidden_state(batch_size)
        hidden_states = self.decoder.initialize_hidden_states(batch_size)

        # print('dec_input_token: {}'.format(dec_input_token.shape))
        # print('hidden_state: {}'.format(hidden_state.shape))

        for t in range(trg_len):
            dec_input = self.dec_embedding(dec_input_token) 
            decoded, hidden_state, attention_weights = self.decoder(encoded, dec_input, hidden_states)

            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            # print('decoded: {}'.format(decoded.shape))
            pred_token = tf.argmax(decoded[0, 0, :]).numpy()
            # print('pred_token: {}'.format(pred_token.shape))
            if pred_token != 0:
                result += self.target_tokenizer.index_word[pred_token] + ' '

                if self.target_tokenizer.index_word[pred_token] == '<end>':
                    return result, sentence, attention_plot

            else:
                result += 'NONE' + ' '


            # print('pred_token: {}'.format(pred_token.shape))
            # no teacher forcing
            dec_input_token = np.asarray(pred_token).reshape([1, 1])

        return result, sentence, attention_plot