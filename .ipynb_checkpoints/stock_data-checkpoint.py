{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'preprocess_data'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-100-154177b160d4>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mrequests\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtensorflow\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 7\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mpreprocess_data\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      8\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpreprocessing\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'preprocess_data'"
     ]
    }
   ],
   "source": [
    "from keras.models import Model\n",
    "from keras.layers import Dense, Dropout, LSTM, Input, Activation\n",
    "from keras import optimizers\n",
    "import numpy as np\n",
    "import requests\n",
    "import tensorflow\n",
    "import preprocess_data\n",
    "from sklearn import preprocessing\n",
    "\n",
    "np.random.seed(5)\n",
    "tensorflow.random.set_seed(3)\n",
    "\n",
    "history_points = 20\n",
    "# dataset\n",
    "# url = 'https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/2015-01-01/2020-12-18?apiKey=jXuS6onzWtfNEkQhMJyrCty0ZQIgPsi5'.format('SPY')\n",
    "# response_data = requests.get(url)\n",
    "# results = response_data.json()['results']\n",
    "# results = map(lambda day: [day['o'], day['h'], day['l'], day['c'], day['v']], results)\n",
    "# results = list(results)\n",
    "# results = np.array(results)\n",
    "\n",
    "# data_normaliser = preprocessing.MinMaxScaler()\n",
    "# data_normalised = data_normaliser.fit_transform(results)\n",
    "    \n",
    "# # using the last {history_points} open close high low volume data points, predict the next open value\n",
    "# ohlcv_histories = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])\n",
    "\n",
    "# next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])\n",
    "# next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)\n",
    "\n",
    "# next_day_open_values = np.array([results[:, 0][i + history_points].copy() for i in range(len(results) - history_points)])\n",
    "# next_day_open_values = np.expand_dims(next_day_open_values, -1)\n",
    "\n",
    "# y_normaliser = preprocessing.MinMaxScaler()\n",
    "# y_normaliser.fit(next_day_open_values)\n",
    "\n",
    "ohlcv_histories, next_day_open_values, unscaled_y, y_normaliser = preprocess_data(ticker)\n",
    "\n",
    "test_split = 0.9\n",
    "n = int(ohlcv_histories.shape[0] * test_split)\n",
    "\n",
    "ohlcv_train = ohlcv_histories[:n]\n",
    "y_train = next_day_open_values[:n]\n",
    "\n",
    "ohlcv_test = ohlcv_histories[n:]\n",
    "y_test = next_day_open_values[n:]\n",
    "\n",
    "unscaled_y_test = unscaled_y[n:]\n",
    "\n",
    "# model architecture\n",
    "\n",
    "lstm_input = Input(shape=(history_points, 5), name='lstm_input')\n",
    "x = LSTM(100, name='lstm_0')(lstm_input)\n",
    "x = Dropout(0.2, name='lstm_dropout_0')(x)\n",
    "x = Dense(64, name='dense_0')(x)\n",
    "x = Activation('sigmoid', name='sigmoid_0')(x)\n",
    "x = Dense(1, name='dense_1')(x)\n",
    "output = Activation('linear', name='linear_output')(x)\n",
    "\n",
    "model = Model(inputs=lstm_input, outputs=output)\n",
    "adam = optimizers.Adam(lr=0.0005)\n",
    "model.compile(optimizer=adam, loss='mse')\n",
    "model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)\n",
    "\n",
    "\n",
    "# evaluation\n",
    "\n",
    "y_test_predicted = model.predict(ohlcv_test)\n",
    "y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)\n",
    "y_predicted = model.predict(ohlcv_histories)\n",
    "y_predicted = y_normaliser.inverse_transform(y_predicted)\n",
    "\n",
    "assert unscaled_y_test.shape == y_test_predicted.shape\n",
    "real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))\n",
    "scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100\n",
    "print(scaled_mse)\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.gcf().set_size_inches(22, 15, forward=True)\n",
    "\n",
    "start = 0\n",
    "end = -1\n",
    "\n",
    "real = plt.plot(unscaled_y_test[start:end], label='real')\n",
    "pred = plt.plot(y_test_predicted[start:end], label='predicted')\n",
    "\n",
    "# real = plt.plot(unscaled_y[start:end], label='real')\n",
    "# pred = plt.plot(y_predicted[start:end], label='predicted')\n",
    "\n",
    "plt.legend(['Real', 'Predicted'])\n",
    "\n",
    "plt.show()\n",
    "\n",
    "from datetime import datetime\n",
    "model.save(f'basic_model.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
