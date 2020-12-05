import numpy as np
import tensorflow as tf


class RNN_multichannel_dropout(tf.keras.Model):
    def __init__(self, window_size, vocab_size):
        super(RNN_multichannel_dropout, self).__init__()
        # Defining the hyperparameters
        self.batch_size = 100
        self.embedding_size = 60
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.vocab_size = vocab_size
        self.window_size = window_size

        # Embedding vector
        self.embedding_matrix = tf.Variable(
            tf.random.truncated_normal(
                [self.vocab_size, self.embedding_size], stddev=0.01
            )
        )
        # GRU Layers
        self.text1_encoder = tf.keras.layers.GRU(100)
        self.text2_encoder = tf.keras.layers.GRU(100)
        self.text3_encoder = tf.keras.layers.GRU(100)
        # Dense layer 1
        self.text1_dense1 = tf.keras.layers.Dense(100, activation="relu")
        self.text2_dense1 = tf.keras.layers.Dense(100, activation="relu")
        self.text3_dense1 = tf.keras.layers.Dense(100, activation="relu")
        # Dense layer 2
        self.text1_dense2 = tf.keras.layers.Dense(100, activation="relu")
        self.text2_dense2 = tf.keras.layers.Dense(100, activation="relu")
        self.text3_dense2 = tf.keras.layers.Dense(100, activation="relu")
        # Dense layer 3
        self.concat_dense = tf.keras.layers.Dense(50, activation="relu")
        # Output layer
        self.concat_regress = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, text_col1, text_col2, text_col3, is_test=False):
        # Creating embedding table. Size of (BATCH_SZ, SENTENCE_LENGTH, EMBEDDING_SZ)
        embedded_text_col1 = tf.nn.embedding_lookup(self.embedding_matrix, text_col1)
        embedded_text_col2 = tf.nn.embedding_lookup(self.embedding_matrix, text_col2)
        embedded_text_col3 = tf.nn.embedding_lookup(self.embedding_matrix, text_col3)
        # Conducting Dropout
        if is_test == False:
            embedded_text_col1 = tf.nn.dropout(embedded_text_col1, rate=0.1)
            embedded_text_col2 = tf.nn.dropout(embedded_text_col2, rate=0.1)
            embedded_text_col3 = tf.nn.dropout(embedded_text_col3, rate=0.1)
        # Output of RNN layers, represents the summarized sentence
        text1_encoder_output = self.text1_encoder(embedded_text_col1)
        text2_encoder_output = self.text2_encoder(embedded_text_col2)
        text3_encoder_output = self.text3_encoder(embedded_text_col3)
        # Dense layer outputs using the RNN output
        text1_dense1_output = self.text1_dense1(text1_encoder_output)
        text2_dense1_output = self.text2_dense1(text2_encoder_output)
        text3_dense1_output = self.text3_dense1(text3_encoder_output)
        # Second dense layer
        text1_dense2_output = self.text1_dense2(text1_dense1_output)
        text2_dense2_output = self.text2_dense2(text2_dense1_output)
        text3_dense2_output = self.text3_dense2(text3_dense1_output)
        # Concatenation of the three different outputs into a single output
        concatenation = tf.concat(
            [text1_dense2_output, text2_dense2_output, text3_dense2_output], axis=1
        )
        # Putting through the third dense layer
        concatenated_output = self.concat_dense(concatenation)
        # Putting through the regression layer
        regression_output = self.concat_regress(concatenated_output)
        return regression_output

    def loss_function(self, predicted_ratings, true_ratings):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(predicted_ratings, true_ratings)
