import yfinance as yf
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


class Stock:
    def __init__(self, ticker_id='', data_path='data/yahoo_data'):
        self.ticker = ticker_id
        self.path = data_path
        self.ticker_file = f'{self.path}/{self.ticker}_data.json'


    def retrieve_stock_data(self):
        ''' Return dataframe for ticker and write to file '''
        if os.path.exists(self.ticker_file):
            with open(self.ticker_file) as f:
                self.hist = pd.read_json(self.ticker_file)
        else:
            self.ticker = yf.Ticker(self.ticker)
            self.hist = self.ticker.history(period="max")

            self.hist.to_json(self.ticker_file)
        
        return self.hist

    def basic_ticker_info(self):
        ''' Generate head for dataframe and display plot graph of historical rpice '''
        self.pd = self.retrieve_stock_data()

        print(self.pd.head(5))
        self.pd.plot.line(y="Close", use_index=True)
        # plt.show()

    def prep_training_data(self):
        self.data = self.retrieve_stock_data()
        self.shifted_data = self.data.copy()
        self.shifted_data = self.shifted_data.shift(1)
        self.data = self.data[['Close']]
        self.data = self.data.rename(columns= {'Close':'ActualClose'})
        # print(self.data.head(5))
        self.data['Target'] = self.data.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])['ActualClose']

        # Shift prices forward
        self.market_predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
        self.training_data = self.data.join(self.shifted_data[self.market_predictors]).iloc[1:]

        print(self.training_data.head(10))

        return self.market_predictors, self.training_data


class Learning_Model:
    def __init__(self, predictors, training_data):
        self.predictors = predictors
        self.training_data = training_data
        self.model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=2000,
            random_state=1
        )
        self.train = self.training_data.iloc[:-100]
        self.test = self.training_data.iloc[-100:]

        self.model.fit(self.train[predictors], self.train["Target"])

    def check_precision_score(self):
        self.predictors = self.model.predict(self.test[self.predictors])
        self.predictors = pd.Series(self.predictors, index=self.test.index)
        self.precision = precision_score(self.test['Target'], self.predictors)

        # Round
        self.precision = round(self.precision * 100, 2)
        return self.precision

    # def backetesting(self, training_data):