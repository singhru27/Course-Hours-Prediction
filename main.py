import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from preprocess_all import *
import sys
import random
from multichannelRNNmodel import RNN_multichannel
from multichannelRNNmodelDropout import RNN_multichannel_dropout
from multichannelNumerical import multichannel_numerical

########## DO NOT CHANGE EXCEPT BY RUJUL #####################
WINDOW_SIZE = 125
EPOCHS = 15
NUMERICAL_EPOCHS = 15
########## DO NOT CHANGE EXCEPT BY RUJUL #####################

random.seed(0)
tf.random.set_seed(1)


def train(model, train_labels, train_text1, train_text2, train_text3):
    """
    Trains the model
    Parameters:
        - model, train_labels, train_text1, train_text2, train_text3
    Return:
        - None
    """
    batch_size = model.batch_size
    num_examples = len(train_labels)
    # Shuffling the training inputs
    indices = tf.range(num_examples)
    indices = tf.random.shuffle(indices)
    shuffled_train_labels = tf.gather(train_labels, indices)
    shuffled_train_text1 = tf.gather(train_text1, indices)
    shuffled_train_text2 = tf.gather(train_text2, indices)
    shuffled_train_text3 = tf.gather(train_text3, indices)
    # Iterating through the matches
    num_batches = num_examples // batch_size
    for i in range(num_batches):
        batched_train_labels = shuffled_train_labels[
            i * batch_size : (i + 1) * batch_size
        ]
        batched_train_text1 = shuffled_train_text1[
            i * batch_size : (i + 1) * batch_size
        ]
        batched_train_text2 = shuffled_train_text2[
            i * batch_size : (i + 1) * batch_size
        ]
        batched_train_text3 = shuffled_train_text3[
            i * batch_size : (i + 1) * batch_size
        ]
        # Creating the predictions
        with tf.GradientTape() as tape:
            predictions = model.call(
                batched_train_text1, batched_train_text2, batched_train_text3
            )
            loss = model.loss_function(predictions, batched_train_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(loss)


def train_numerical(
    model, train_labels, train_text1, train_text2, train_text3, train_numerical_data
):
    """
    Trains the model
    Parameters:
        - model, train_labels, train_text1, train_text2, train_text3, train_numerical_data
    Return:
        - None
    """
    batch_size = model.batch_size
    num_examples = len(train_labels)
    # Shuffling the training inputs
    indices = tf.range(num_examples)
    indices = tf.random.shuffle(indices)
    shuffled_train_labels = tf.gather(train_labels, indices)
    shuffled_train_text1 = tf.gather(train_text1, indices)
    shuffled_train_text2 = tf.gather(train_text2, indices)
    shuffled_train_text3 = tf.gather(train_text3, indices)
    shuffled_train_numerical_data = tf.gather(train_numerical_data, indices)
    # Iterating through the matches
    num_batches = num_examples // batch_size
    for i in range(num_batches):
        batched_train_labels = shuffled_train_labels[
            i * batch_size : (i + 1) * batch_size
        ]
        batched_train_text1 = shuffled_train_text1[
            i * batch_size : (i + 1) * batch_size
        ]
        batched_train_text2 = shuffled_train_text2[
            i * batch_size : (i + 1) * batch_size
        ]
        batched_train_text3 = shuffled_train_text3[
            i * batch_size : (i + 1) * batch_size
        ]
        batched_train_numerical_data = shuffled_train_numerical_data[
            i * batch_size : (i + 1) * batch_size
        ]
        # Creating the predictions
        with tf.GradientTape() as tape:
            predictions = model.call(
                batched_train_text1,
                batched_train_text2,
                batched_train_text3,
                batched_train_numerical_data,
            )
            loss = model.loss_function(predictions, batched_train_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(loss)


def test(model, test_labels, test_text1, test_text2, test_text3):
    batch_size = model.batch_size
    num_examples = len(test_labels)
    mse = []
    # Iterating through the matches
    num_batches = num_examples // batch_size
    for i in range(num_batches):
        batched_test_labels = test_labels[i * batch_size : (i + 1) * batch_size]
        batched_test_text1 = test_text1[i * batch_size : (i + 1) * batch_size]
        batched_test_text2 = test_text2[i * batch_size : (i + 1) * batch_size]
        batched_test_text3 = test_text3[i * batch_size : (i + 1) * batch_size]

        predictions = model.call(
            batched_test_text1,
            batched_test_text2,
            batched_test_text3,
            is_test=True,
        )
        loss = model.loss_function(predictions, batched_test_labels)
        mse.append(loss)
    return sum(mse) / num_batches


def test_numerical(
    model, test_labels, test_text1, test_text2, test_text3, test_numerical_data
):
    batch_size = model.batch_size
    num_examples = len(test_labels)
    mse = []
    # Iterating through the matches
    num_batches = num_examples // batch_size
    for i in range(num_batches):
        batched_test_labels = test_labels[i * batch_size : (i + 1) * batch_size]
        batched_test_text1 = test_text1[i * batch_size : (i + 1) * batch_size]
        batched_test_text2 = test_text2[i * batch_size : (i + 1) * batch_size]
        batched_test_text3 = test_text3[i * batch_size : (i + 1) * batch_size]
        batched_test_numerical_data = test_numerical_data[
            i * batch_size : (i + 1) * batch_size
        ]

        predictions = model.call(
            batched_test_text1,
            batched_test_text2,
            batched_test_text3,
            batched_test_numerical_data,
            is_test=True,
        )
        loss = model.loss_function(predictions, batched_test_labels)
        mse.append(loss)
    return sum(mse) / num_batches


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"RNN", "RNN_DROPOUT", "RNN_ALL"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [RNN/RNN_DROPOUT/RNN_ALL]")
        exit()

    print("Running preprocessing...")
    # Model Selection
    if sys.argv[1] == "RNN":
        (
            train_labels,
            train_text1,
            train_text2,
            train_text3,
            test_labels,
            test_text1,
            test_text2,
            test_text3,
            vocab_dict,
        ) = preprocess_data("2020-2019 Review Data.csv")
        VOCAB_SIZE = len(vocab_dict.keys())
        print("Preprocessing complete.")
        model = RNN_multichannel(WINDOW_SIZE, VOCAB_SIZE)
        # Training for EPOCH number of iterations
        for i in range(EPOCHS):
            print(i)
            train(model, train_labels, train_text1, train_text2, train_text3)
        print(
            "Final Loss", test(model, test_labels, test_text1, test_text2, test_text3)
        )

    # Running the dropout model
    elif sys.argv[1] == "RNN_DROPOUT":
        (
            train_labels,
            train_text1,
            train_text2,
            train_text3,
            test_labels,
            test_text1,
            test_text2,
            test_text3,
            vocab_dict,
        ) = preprocess_data("2020-2019 Numerical Review Data.csv")
        VOCAB_SIZE = len(vocab_dict.keys())
        print("Preprocessing complete.")

        model = RNN_multichannel_dropout(WINDOW_SIZE, VOCAB_SIZE)
        # Training for EPOCH number of iterations
        for i in range(EPOCHS):
            print(i)
            train(model, train_labels, train_text1, train_text2, train_text3)
        print(
            "Final Loss", test(model, test_labels, test_text1, test_text2, test_text3)
        )

    # Running the model with numerical data included
    elif sys.argv[1] == "RNN_ALL":
        (
            train_labels,
            train_text1,
            train_text2,
            train_text3,
            train_numerical_data,
            test_labels,
            test_text1,
            test_text2,
            test_text3,
            test_numerical_data,
            vocab_dict,
        ) = preprocess_data_numerical("2020-2019 Numerical Review Data.csv")
        VOCAB_SIZE = len(vocab_dict.keys())
        print("Preprocessing complete.")
        ################################################################################################

        ################################v################################################################################################
        # Creating the model
        model = multichannel_numerical(WINDOW_SIZE, VOCAB_SIZE)
        # Training for EPOCH number of iterations
        for i in range(NUMERICAL_EPOCHS):
            train_numerical(
                model,
                train_labels,
                train_text1,
                train_text2,
                train_text3,
                train_numerical_data,
            )
        print(
            "Final Loss",
            test_numerical(
                model,
                test_labels,
                test_text1,
                test_text2,
                test_text3,
                test_numerical_data,
            ),
        )


if __name__ == "__main__":
    main()
