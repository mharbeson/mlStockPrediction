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
- Linux users will require python3-tk and python3-gi-cairo. These can be installed by:
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

- Category 1: Python Programming Basics:
    - Create a dictionary or list, populate it with several values, retrieve at least one value, and use it in your program.
        - See function trackFeaturesGenerator()
    - Create and call at least 3 functions or methods, at least one of which must return a value that is used somewhere else in your code. To clarify, at least one function should be called in your code, that function should calculate, retrieve, or otherwise set the value of a variable or data structure, return a value to where it was called, and use that value somewhere else in your code.
        - See main.py

- Category 2: Utilize External Data:
    - Connect to an external/3rd party API and read data into your app
        - See main.py

- Category 3: Data Display
    - Visualize data in a graph, chart, or other visual representation of data.
        - See function trackFeatureHeatmap()
    - Display data in tabular form
        - See function releaseYearHistogram()

- Category 4: Best Practices
    - The program should utilize a virtual environment and document library dependencies in a requirements.txt file.
        - See readme for requirements.txt

- Future Stretch Items
    - Impliment paper trading to allow the algorithim to learn from it's decisions.
    - 