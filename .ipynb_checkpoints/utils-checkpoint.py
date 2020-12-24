import requests


history_points = 20

def preprocess_data(ticker):
    data = get_data_from_source(self, ticker);
    
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    historical_data_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)
    
    assert historical_data_normalised.shape[0] == next_day_open_values_normalised.shape[0]
    return ohlcv_histories_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

def get_data_from_source(self, ticker):
    url = 'https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/2015-01-01/2020-12-18?apiKey=jXuS6onzWtfNEkQhMJyrCty0ZQIgPsi5'.format(ticker)
    response_data = requests.get(url)
    results = response_data.json()['results']
    results = map(lambda day: [day['o'], day['h'], day['l'], day['c'], day['v']], results)
    results = list(results)

    return np.array(results)