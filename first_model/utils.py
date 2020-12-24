import numpy as np
import requests
from sklearn import preprocessing


history_points = 20

def return_vs_market(ticker, spy):
  return (ticker['o'] - ticker['c']) - (spy['o'] - spy['c'])

def fetch_ticker_data(ticker):
  return requests.get('https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/2015-01-01/2020-12-18?apiKey=jXuS6onzWtfNEkQhMJyrCty0ZQIgPsi5'.format(ticker));

def transform_data(data):
  results = data.json()['results']
  results = map(lambda day: [day['o'], day['h'], day['l'], day['c'], day['v'], day['t'], day['vw']], results)
  results = list(results)

  return np.array(results)

def get_data_from_source(ticker):
    spy_data = transform_data(fetch_ticker_data('SPY'))
    ticker_data = transform_data(fetch_ticker_data(ticker))
    # ticker_vs_sp_return = np.array([return_vs_market(ticker_data[i], spy_data[i]) for i in range(len(ticker_data))])
    # ticker_data = np.concatenate((ticker_data, ticker_vs_sp_return), axis = 1)
    # print(np.shape(ticker_data))
    return ticker_data

def preprocess_data(ticker):
    data = get_data_from_source(ticker);
    
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    historical_data_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data[:, 0][i + history_points].copy() 
      for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    for his in historical_data_normalised:
        sma = np.mean(his[:, 3])
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        technical_indicators.append(np.array([sma]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)
    
    assert historical_data_normalised.shape[0] == next_day_open_values_normalised.shape[0]
    return historical_data_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser
