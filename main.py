from helpers.stock_ml_helper import Stock
import matplotlib.pyplot as plt

def ticker_prompt(ticker=''):
    if ticker == '':
        ticker = input('Input stock ticker for analysis: ')
        return ticker, Stock(ticker)
    else:
        return ticker, Stock(ticker)

def line_break():
    print('------------------------------------------------')

# MSFT
# AAPL
# AMZN
# GOOG

def main():
    ticker_name, stock1 = ticker_prompt()
    print(f'Generating results for {ticker_name}')
    line_break()

    market_indicators, training_data = stock1.prep_training_data()
    stock1.basic_ticker_info()
    print(training_data.tail(10))
    line_break()

    print('Check for class inbalance')
    print('This is important to ensure the truth value is not skewed in one direction')
    print(training_data['target'].value_counts())
    line_break()

    stock1.create_model(market_indicators, training_data)

    score1 = stock1.check_precision_score()
    line_break()
    print(f'Initial Precision score before backtesting: {score1}')
    line_break()

    predictions, score2 = stock1.backtesting()
    line_break()
    print(f'Backtesting score: {score2}')
    line_break()
    print('Backtesting Predictions:\n')
    print(predictions)
    line_break()

    print('Plotting Target vs Prediction')
    print('1:1 line is the ideal prediction')
    stock1.pred_heatmap()


if __name__ == "__main__":
    main()    

