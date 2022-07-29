import yfinance as yf
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from finta import TA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split


class Stock:
    
    # Class variables
    # Indicators for use with finta api
    indicators = ['RSI', 'MACD', 'STOCH', 'ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']
    smoothing = 2

    # Random Forest Classifier Parameters
    # n_estimators=300
    # min_samples_split=200
    n_estimators=100
    min_samples_split=200

    # Define Backetesting Parameters
    # Smaller steps will take longer to run but may yield better accuracy
    start = 1000
    step = 1000
    # step = 750
    # step = 50

    # Adjust the below value to determine the threshold for backtesting. Setting > 0.6 will throw out more potential correct trades
    backtest_threshold = 0.6

    def __init__(self, ticker_id='', data_path='data/yahoo_data'):
        self.ticker = ticker_id
        self.path = data_path
        self.ticker_file = f'{self.path}/{self.ticker}_data.json'
        self.training_data_file = f'data/{self.ticker}_training_data.csv'
        self.prediction_file = f'data/{self.ticker}_predictions.csv'

    # Calculate Exponential Moving Average
    def calculate_ema(self, data, period, smoothing=2):
        ''' Calculate Exponential Moving Average '''
        self.ema = data.close.ewm(com=period).mean()
        return self.ema

    def normalize_data(self, data):
        ''' Normalizes data for single column to moving average'''
        return (data / data.ewm(5).mean())

    def retrieve_stock_data(self):
        ''' Return dataframe for ticker and write to file '''
        if os.path.exists(self.ticker_file):
            with open(self.ticker_file) as f:
                self.data = pd.read_json(self.ticker_file)
        else:
            self.ticker = yf.Ticker(self.ticker)
            self.data = self.ticker.history(period='max')

            self.data.to_json(self.ticker_file)
        
        return self.data

    def retrieve_indicator_data(self, data):
        for indicator in self.indicators:
            self.indicator_data = eval('TA.' + indicator + '(self.data)')
            if not isinstance(self.indicator_data, pd.DataFrame):
                self.indicator_data = pd.DataFrame(self.indicator_data)

            # Merge FINTA Indicator Data with Yahoo Data
            self.data = self.data.merge(self.indicator_data, left_index=True, right_index=True)

        #Calculate Exponential Moving Averages for 50 day, 21 day, 15 day, and 5 day
        self.data['ema50'] = self.calculate_ema(self.data, 50, smoothing=self.smoothing)
        self.data['ema21'] = self.calculate_ema(self.data, 21, smoothing=self.smoothing)
        self.data['ema15'] = self.calculate_ema(self.data, 15, smoothing=self.smoothing)
        self.data['ema5'] = self.calculate_ema(self.data, 5, smoothing=self.smoothing)

        # Normalize volume to moving volume average
        self.data['normalized_volume'] = self.normalize_data(self.data['volume'])

        # Open/Close, High/Close, and Low/Close Ratios
        self.data['open_close_ratio'] = self.data['open'] / self.data['close']
        self.data['high_close_ratio'] = self.data['high'] / self.data['close']
        self.data['low_close_ratio'] = self.data['low'] / self.data['close']

        return self.data

    def basic_ticker_info(self):
        ''' Generate head for dataframe and display plot graph of historical rpice '''
        self.pd = self.retrieve_stock_data()

        print(self.pd.head(5))
        self.pd.plot.line(y="Close", use_index=True)
        # plt.show()

    def prep_training_data(self):

        # Retrieve data and format for Finta API
        # Info here: https://github.com/peerchemist/finta

        self.data = self.retrieve_stock_data()
        self.data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        del (self.data['Dividends'], self.data['Stock Splits'])

        # Call fundtion to retrieve Indicator Data and merge with Yahoo Data
        # Requirement 1 and 2
        self.data = self.retrieve_indicator_data(self.data)

        # Shift data to prevent look ahead
        self.shifted_data = self.data.copy()
        self.shifted_data = self.shifted_data.shift(1)

        # Save columns for join later
        self.market_indicators = self.data.columns

        # Calculate target (either 1 or 0) based on close price.
        # This is the truth value
        self.data = self.data[['close']]
        self.data = self.data.rename(columns= {'close':'actualclose'})
        self.data['target'] = self.data.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])['actualclose']
        self.data['target'].value_counts()

        # Joing shifted data with target data
        self.training_data = self.data.join(self.shifted_data[self. market_indicators]).iloc[1:]

        # Drop NAN values
        self.training_data.dropna(inplace=True)
        # Save training data to file for potential future use
        self.training_data.to_csv(self.training_data_file)

        return self.market_indicators, self.training_data


    def create_model(self, market_indicators, training_data):
        ''' Create model and fit to data '''
        self.market_indicators = market_indicators
        self.training_data = training_data

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            random_state=1
        )

    def fit_model(self):
        train = self.training_data.iloc[:-100]
        test = self.training_data.iloc[-100:]        
        self.model.fit(train[self.market_indicators], train['target'])

        return train, test, self.model

        

    def check_precision_score(self):
        ''' Check precision score of model '''
        self.train, self.test, self.model = self.fit_model()

        m_indicators = self.model.predict(self.test[self.market_indicators])
        m_indicators = pd.Series(m_indicators, index=self.test.index)
        self.precision = precision_score(self.test['target'], m_indicators)

        # Round for easier viewing in console
        self.precision = round(self.precision * 100, 2)

        self.combined = pd.concat({'target': self.test['target'], 'prediction': m_indicators}, axis=1)
        # DF of True Target and Predicted Target
        print(self.combined.tail(10))

        return self.precision

    def backtesting(self):
        ''' Backtesting Function '''
        predictions = []
        # Backtest loop
        for i in range(self.start, self.training_data.shape[0], self.step):
            # Create training and test sets
            train = self.training_data.iloc[0:i].copy()
            test = self.training_data.iloc[i:i+self.step].copy()

            print(f'Computing predictions for {i} to {i+self.step}... of {self.training_data.shape[0]}')

            self.model.fit(train[self.market_indicators], train['target'])

            m_ind = self.model.predict_proba(test[self.market_indicators])[:, 1]
            m_ind = pd.Series(m_ind, index=test.index)
            m_ind[m_ind > self.backtest_threshold] = 1
            m_ind[m_ind <= self.backtest_threshold] = 0

            # Combine pred and test
            combined = pd.concat({'target': test['target'], 'prediction': m_ind}, axis=1)

            predictions.append(combined)
        predictions = pd.concat(predictions)
        predictions.to_csv(self.prediction_file)
        
        # Calculate new precision score
        ps = precision_score(predictions['target'], predictions['prediction'])
        ps = round(ps * 100, 2)

        return predictions, ps