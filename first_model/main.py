import numpy as np
import requests
import tensorflow
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
from models import ltsm
from sklearn import preprocessing
from plot_data import plot_forecast
from utils import preprocess_data

np.random.seed(5)
tensorflow.random.set_seed(3)

history_points = 20

historical_data_normalised, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = preprocess_data('DIS')

test_split = 0.9
n = int(historical_data_normalised .shape[0] * test_split)

historical_data_train = historical_data_normalised [:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]

historical_data_test = historical_data_normalised[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

y_test_predicted, y_predicted = ltsm(historical_data_train, y_train, historical_data_normalised, historical_data_test, technical_indicators, tech_ind_train, tech_ind_test, y_normaliser, history_points)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100

print(scaled_mse)
plot_forecast(unscaled_y_test, y_test_predicted)


