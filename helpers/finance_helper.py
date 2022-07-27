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

        # Define backtesting parameters
        self.start = 1000
        self.step = 750

        

    def check_precision_score(self):
        self.predictors = self.model.predict(self.test[self.predictors])
        self.predictors = pd.Series(self.predictors, index=self.test.index)
        self.precision = precision_score(self.test['Target'], self.predictors)

        # Round
        self.precision = round(self.precision * 100, 2)
        return self.precision

    def backtesting(self):
        self.predictions = []
        # Loop
        for i in range(self.start, self.training_data.shape[0], self.step):
            # Create training and test sets
            self.train = self.training_data.iloc[0:i].copy()
            self.test = self.training_data.iloc[i:i+self.step].copy()

            # Fit model
            # self.model.fit(self.train[self.predictors], self.train["Target"])  

            self.preds = self.model.predict_proba(self.test[self.predictors])[:, 1]
            self.preds = pd.Series(self.preds, index=self.test.index)
            self.preds[self.preds > 0.6] = 1
            self.preds[self.preds <= 0.6] = 0

            # Combine pred and test
            # self.combined = pd.concat([self.preds, self.test['Target']], axis=1)
            self.combined = pd.concat({'Target': self.test['Target'], 'Predictions': self.preds}, axis=1)

            self.predictions.append(self.combined)
        
        return pd.concat(self.predictions)  # Concatenate all predictions   
