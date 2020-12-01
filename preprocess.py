import tensorflow as tf
import numpy as np
from functools import reduce
import csv
import random


def get_all_data(file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with all words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """
    # data = pd.read_csv(file)
    all_data = []
    with open(file) as csvfile:
        datareader = csv.reader(csvfile, delimiter=",")
        for row in datareader:
            all_data.append(row)
    dict = {}

    # Separating all punctuation in the dataset. Creating a list of words for each entry field for each datapoint
    all_data = all_data[2:]
    for i in range(len(all_data)):
        all_data[i][1] = str(all_data[i][1]).lower()
        all_data[i][2] = str(all_data[i][2]).lower()
        all_data[i][3] = str(all_data[i][3]).lower()

        all_data[i][2] = all_data[i][2].replace(".", " .")
        all_data[i][2] = all_data[i][2].replace(",", " ,")
        all_data[i][2] = all_data[i][2].replace("!", " !")
        all_data[i][2] = all_data[i][2].replace("?", " ?")
        all_data[i][2] = all_data[i][2].replace(")", " )")
        all_data[i][2] = all_data[i][2].replace("(", "( ")

        all_data[i][3] = all_data[i][3].replace(".", " .")
        all_data[i][3] = all_data[i][3].replace(",", " ,")
        all_data[i][3] = all_data[i][3].replace("!", " !")
        all_data[i][3] = all_data[i][3].replace("?", " ?")
        all_data[i][3] = all_data[i][3].replace(")", " )")
        all_data[i][3] = all_data[i][3].replace("(", "( ")

        all_data[i][1] = all_data[i][1].split(",")
        all_data[i][2] = all_data[i][2].split()
        all_data[i][3] = all_data[i][3].split()
        all_data[i][3].append(all_data[i][4])
        all_data[i] = [
            all_data[i][0],
            all_data[i][1],
            all_data[i][2],
            all_data[i][3],
        ]
        # Creating a dictionary mapping words to word counts, to be used for unkING later on
        for j in range(2, 4):
            for word in all_data[i][j]:
                if (word in dict) == False:
                    dict[word] = 0
                else:
                    dict[word] = dict[word] + 1
    # UNKing the words in the dataset
    for i in range(len(all_data)):
        for j in range(2, 4):
            for k in range(len(all_data[i][j])):
                if dict[all_data[i][j][k]] < 2:
                    all_data[i][j][k] = "UNK"
    data_id = {}
    id = 0
    # Creating a vocabulary dictionary, mapping from words to their ids
    id_num = 0
    vocab_dict = {}
    for i in range(len(all_data)):
        for j in range(2, 4):
            for word in all_data[i][j]:
                if word not in vocab_dict:
                    vocab_dict[word] = id_num
                    id_num += 1

    return all_data, vocab_dict


def get_id_list(vocab_dict):
    """
    Returns
    """


def split_data(all_data):
    """
    Splits the entire set of data into a training set and a testing set
    Parameters:
        - all_data: The full dataset
    Returns:
        - train_data: The training dataset
        - test_data: The testing dataset
    """
    num_elements = len(all_data)
    train_percentage = 0.8
    test_percentage = 0.2
    random.shuffle(all_data)
    # Creating the train and test datasets, in list form
    num_in_train = int(train_percentage * num_elements)
    train_data = all_data[0:num_in_train]
    test_data = all_data[num_in_train:num_elements]
    return (train_data, test_data)


if __name__ == "__main__":
    all_data, vocab_dict = get_all_data("2020-2019 Review Data.csv")
    train_data, test_data = split_data(all_data)
