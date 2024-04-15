# DSA4263 Fraud Hackathon Project
Fraud is a very common issue in the world of finance. Common types of fraud involving bank accounts include, account opening fraud, money laundering and fraudulent transactions etc. In this report, we present an innovative approach to enhancing fraud detection amongst bank accounts using machine learning techniques.

## Installation 
Prior to installation, you can clone the repository. We use python 3.12.2, and the requirements file specifies versions of all other packages. A virtual environment ul-env (replace ul-env with any other name) can be created in the command line:


## Files Structure
Fraud-Hackathon
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│        └── Base.zip  <- Compressed version of the dataset used
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│   │                  the creator's initials, and a short `-` delimited description, e.g.
│   │                  `1.0-jqp-initial-data-exploration`. For more details on what each notebook
│   │                  is about, refer to `notebooks/README.md`
│   │ 
│   ├── AdaBoost_Model.ipynb 
│   ├── DecisionTree_Model.ipynb
│   ├── eda.ipynb
│   ├── GaussianNB_Model.ipynb
│   ├── hypo_eda.ipynb
│   ├── LogisticRegression_Model.ipynb
│   ├── pipeline.ipynb
│   ├── README.md
│   ├── resampling.ipynb
│   └── XGBoost_Model.ipynb
│  
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└──  src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py  <- unzips Base.zip to Base.csv
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py <- Runs the full data processing pipeline and generates train and test datasets.
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── train_model.py    <- Script to train a sklearn model or xgboost model
    │   ├── predict_model.py  <- Script to make a prediction using a saved model
    │   └── evaluate_model.py <- Script to evaluate model performanced using predictions
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py      <- Script to generate visualizations used in the report


## Instructions to Run
There are 2 methods to run the code in the files, 1) Running using `make` and 2) Running the python scripts directly.

### Method 1 : Running using `make`
```bash
# To run the full pipeline
1. cd into the Fraud-Hackathon/ directory
2. make
```
```bash
# To run specific parts of the pipeline
1. cd into the Fraud-Hackathon/ directory
2. make <part>  
Available options: 
    venv  <- Creates virtual environment and install packages   
    dataset <- Unzips the dataset
    features <- Preprocesses the data and generate train and test sets
    train <- Trains a model and saves it
    predict <- Makes predictions with a saved model
    evaluate <- Evalaute the performance of the models with its predictions
    visualize <- Generate diagrams used in the report
    ]  e.g make train 
```

### Method 2 : Running the scripts directly
```text
# Activate environment if necessary
1. Install packages required 
pip install -r requirements.txt

2. Change directory into src/ folder
cd src/

Refer to `Fraud-Hackathon\src\README.md` for more details to run each script.
```