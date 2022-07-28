from helpers.finance_helper import Stock
from helpers.finance_helper import Learning_Model
import matplotlib.pyplot as plt

def ticker_prompt(ticker=''):
    if ticker == '':
        ticker = input('Input stock ticker for analysis: ')
        return ticker, Stock(ticker)
    else:
        return ticker, Stock(ticker)

# MSFT
# AAPL
# AMZN
# GOOG
ticker_name, stock1 = ticker_prompt('AAPL')
print(f'Generating results for {ticker_name}')
# stock1.basic_ticker_info()
predictors, training_data = stock1.prep_training_data()
print(stock1.generate_indicator_data())
# print(training_data.head(5))

stock_model1 = Learning_Model(predictors, training_data)
print(f'Model is directionally accurate {stock_model1.check_precision_score()}% of the time.')


#stock_model2 = Learning_Model(predictors, training_data)

# print(stock_model1.predictors)

#print(stock_model1.check_precision_score())
# print(stock_model1.training_data.head(5))
# print(stock_model1.backtesting())

