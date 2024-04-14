# Notebook Repository

Welcome to the Notebook Repository! This repository contains Jupyter notebooks implementing various machine learning models and analysis for a predictive modeling task.

## Overview

The repository includes the following notebooks:

1. **AdaBoost_Model.ipynb**: Implements the AdaBoost model, following the same pipeline as our baseline model. Additionally, it includes explanations using Lime and PDP (Partial Dependence Plots) that contribute to hypothesis testing.

2. **DecisionTree_Model.ipynb**: Implements the Decision Tree model, the baseline model for the project, following our standardized workflow pipeline for data preprocessing (handling of missing values and categorical features, feature scaling and feature selection, and resampling), model training, hyperparameter tuning, and evaluation.

3. **GaussianNB_Model.ipynb**: Implements the Gaussian Naive Bayes model, following the same pipeline as our baseline model. Additionally, it includes explanations using Lime and PDP (Partial Dependence Plots) that contribute to hypothesis testing.

4. **LogisticRegression_Model.ipynb**: Implements the Logistic Regression model, following the same pipeline as our baseline model. Additionally, it includes explanations using Lime and Shap values.

5. **XGBoost_Model_Tuning.ipynb**: Implements the XGBoost model, following the same pipeline as our baseline model.

6. **eda.ipynb**: Conducts exploratory data analysis (EDA) to gain insights into the dataset. It experiments with the effect of scaling on correlation, feature importance, and multicollinearity. 

7. **hypo_eda.ipynb**: This notebook explores hypotheses related to fraudulent activities, analyzing factors such as employment status, income, and phone validity. We employed Seaborn and Matplotlib for data visualization, alongside t-tests to validate hypotheses regarding fraudulent activities.

8. **pipeline.ipynb**: Demonstrates our standardized machine learning workflow pipeline for data preprocessing, model training, and evaluation. 

9. **resampling.ipynb**: This notebook explores feature selection and resampling techniques. Methods include Pearson correlation, variance thresholding, RFE, LASSO with CV, and stepwise selection. Resampling methods such as random undersampling, Tomek links, SMOTE, and combinations are evaluated using decision tree classifiers, comparing various performance metrics.

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
