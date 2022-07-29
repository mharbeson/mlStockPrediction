# Machine Learning Stock Analysis
<p align=center>
<a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
<img src="https://img.shields.io/badge/os-linux-brightgreen">
<img src="https://img.shields.io/badge/os-mac-brightgreen"></a>
<img src="https://img.shields.io/badge/os-windows-brightgreen"></a>
<br>
</p>

- The objective of this project was to allow a user to select any stock ticker and build a machine learning model. 
- Data is pulled from the yfinance and finta API libraries in Python and utilizes the RandomForestClassifier algorithm. 
- This model is designed to maximize true positive value results or when the algorithm predicts the price will go up. This results in a large number of false negatives, however in the case of the stock market this would be minimizing potential losses rather than maximizing gains. 
- The model can further be improved through tweaking market indicators, reduced step size, a different machine learning algorithm. 

- The resulting heatmap shows the results of the Confusion Matrix.


<h1 align="center">
	Showcase
</h1>

<img src=assets/initial_pred.png width="100%">
</p>

<img src=assets/backtesting_pred.png width="100%">
</p>


<h1 align="center">
    Heatmap Key
</h1>

- Target 0: Prediction 0 
    -   True Negative - Days the model correctly chooses not to buy
- Target 1: Prediction 0 
    - False Negative - Days the model incorrectly chooses not to buy 
    - In the real world this equates to a missed opportunity to trade.
- Target 0: Prediction 1
    - False Positive - Days the model incorrectly chooses to buy 
    - In the real world this equates to a loss.
- Target 1: Prediction 1
    - True Positive - Days the model correctly chooses to buy
    - In the real world this equats to a gain
    - This is the ideal outcome

<img src=assets/targetvspred_heatmap.png width="100%">
</p>

## Table of Contents

- [Requirements](#Requirements)
- [Install](#Install)
  - [Linux](#Linux)
  - [Mac](#Mac)
  - [Windows](#Windows)
- [Running](#Running)
- [Project Requirements](#project-requirements)



## Requirements

- Python 3.8.10
- Requirements are outlined in requirements.txt
- Linux users may require python3-tk and python3-gi-cairo. These can be installed by:
```
apt-get install python3-tk python3-gi-cairo
```


## Install

### Linux

- Clone the repository: 
```bash
git clone https://github.com/mharbeson/mlStockPrediction.git
``` 

- Create and activate the virtual environment from the cloned directory:
```bash
python3 -m venv mlStockPrediction
source mlStockPrediction/bin/activate
```

- Install requirements:
```bash
cd mlStockPrediction
pip install -r requirements.txt
```

- Execute main.py. Users will be prompted to select a stock ticker for analysis
```bash
python3 main.py
```


### Mac

- Clone the repository: 
```bash
git clone https://github.com/mharbeson/mlStockPrediction.git
``` 

- Create and activate the virtual environment from the cloned directory:
```bash
python3 -m venv mlStockPrediction
source mlStockPrediction/bin/activate
```

- Install requirements:
```bash
cd mlStockPrediction
pip install -r requirements.txt
```

- Execute main.py. Users will be prompted to select a stock ticker for analysis
```bash
python3 main.py 
```

### Windows

- Clone the repository: 
```
git clone https://github.com/mharbeson/mlStockPrediction.git
``` 

- Create and activate the virtual environment from the cloned directory:
```CMD
python -m venv mlStockPrediction
mlStockPrediction\Scripts\activate.bat
```

```PowerShell
python -m venv mlStockPrediction
mlStockPrediction\Scripts\Activate.ps1
```

- Install requirements:
```
cd mlStockPrediction
pip install -r requirements.txt
```

- Execute main.py. Users will be prompted to select a stock ticker for analysis
```
python main.py
```

## Running
- In the directory run, python3 main.py
- The program will prompt you to enter a stock ticker then provide data results.


## Project Requirements

- Category 1: Loading Data:
    - Read two data sets in with an API. 
    - This was accomplished through the use of the yfinance and finta API in Python.
        - See helper/stock_ml_helper.py

- Category 2: Clean and operate on data:
    - Clean your data and perform a pandas merge with your two data sets, then calculate some new values based on the new data set.  
    - This was accomplished through the prep_training_data() and retrieve_indicator_data() methods, joining the yfinance and finta dataframes and generating new columns such as EMA(Exponential Moving Average).
        - See helper/stock_ml_helper.py

- Category 3: Visualize / Present
    - Visualize data in a graph, chart, or other visual representation of data.
        - See function pred_heatmap(), check_precision_score(), and basic_ticker_info()


- Category 4: Best Practices
    - The program should utilize a virtual environment and document library dependencies in a requirements.txt file.
        - See readme for requirements.txt

- Future Stretch Items
    - Impliment paper trading to allow the algorithim to learn from its decisions.
    - Tailor market indicators to yield highest possible precision score.