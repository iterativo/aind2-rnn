import numpy as np
import math
import re

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# DONE: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    windows_count = len(series) - window_size
    for i in range(windows_count):
        X.append(series[i:i + window_size])

    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


# DONE: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### DONE: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    unique_chars = set(text)
    allowed_chars = {'a','b','c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        ' ', '!', ',', '.', ':', ';', '?'}
    to_remove = unique_chars - allowed_chars

    for r in to_remove:
        text = text.replace(r, ' ')
    return text


### DONE: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    windows_count = math.ceil((len(text) - window_size) / step_size)
    for i in range(windows_count):
        idx_start = i * step_size
        idx_end = idx_start + window_size
        inputs.append(text[idx_start:idx_end])
        outputs.append(text[idx_end])

    return inputs,outputs


# DONE build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
