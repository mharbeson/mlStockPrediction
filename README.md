# Machine Learning Stock Analysis
<p align=center>
<a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
<img src="https://img.shields.io/badge/os-linux-brightgreen">
<img src="https://img.shields.io/badge/os-mac-brightgreen"></a>
<img src="https://img.shields.io/badge/os-windows-brightgreen"></a>
<br>
</p>

-
-
-
-

<h1 align="center">
	Showcase
</h1>

<img src=assets/---.png width="100%">
</p>


## Table of Contents

- [Requirements](#Requirements)
- [Install](#Install)
  - [Linux](#Linux)
  - [Mac](#Mac)
  - [Windows](#Windows)
- [Running](#Running)
- [Notes](#Notes)
- [Project Requirements](#project-requirements)



## Requirements

- Python 3.8.10
- Stuff
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
- 


## Notes

- 

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
        - See function trackFeatureHeatmap()
    - Display data in tabular form
        - See function releaseYearHistogram()

- Category 4: Best Practices
    - The program should utilize a virtual environment and document library dependencies in a requirements.txt file.
        - See readme for requirements.txt

- Future Stretch Items
    - Impliment paper trading to allow the algorithim to learn from its decisions.
    - Tailor market indicators to yield highest possible precision score.