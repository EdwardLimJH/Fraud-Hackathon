# Notebook Repository

Welcome to the Model Notebook Repository! This repository contains Jupyter notebooks implementing our baseline and challenger machine learning models. Each notebook follows a standardized workflow pipeline for data preprocessing (handling of missing values and categorical features, feature scaling and feature selection, and resampling), model training, hyperparameter tuning, and evaluation.

## Overview

The repository includes the following notebooks:

1. **DecisionTree_Model.ipynb**: Implements the Decision Tree model, the baseline model for the project, following our standardized workflow pipeline.

2. **AdaBoost_Model_Tuning.ipynb**: Implements the AdaBoost model, following our standardized workflow pipeline. Additionally, it includes explanations using Lime and PDP (Partial Dependence Plots).

3. **Gaussian_Model_Tuning.ipynb**: Implements the Gaussian Naive Bayes model, following our standardized workflow pipeline. Additionally, it includes explanations using Lime and PDP (Partial Dependence Plots).

4. **LogisticRegression_Model_Tuning.ipynb**: Implements the Logistic Regression model, following our standardized workflow pipeline. Additionally, it includes explanations using Lime and Shap values.

5. **XGBoost_Model_Tuning.ipynb**: Implements the XGBoost model, following our standardized workflow pipeline.

## Usage
  
### Installation and Setup
1. Clone the repository to your local machine:
```bash
git clone <repository-url>
```
2. Navigate to the project directory:
```bash
cd <repository-directory>
```
4. Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
### Running Notebooks
1. Download the Jupyter notebooks from the repository.
2. Open the notebooks using Jupyter Notebook or JupyterLab.
3. Run the cells sequentially in each notebook to execute the code and explore the models.

### Using Saved Models
1. Navigate to the "models" folder in the repository.
2. Download the saved model files (pickle files) for each model.
3. In your Python environment, load the saved models using the pickle library.
