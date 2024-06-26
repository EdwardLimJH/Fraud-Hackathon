# Documentation of Scripts 

## Folder stucture 

```
├── src 
│ ├── __init__.py <- Makes src a Python module
│ │
│ ├── data <- Script to download data
│ │ └── make_dataset.py
│ │
│ ├── features <- Script to turn raw data into features for modeling
│ │ └── build_features.py
│ │
│ ├── models <- Scripts to train models and then use trained models to make
│ │ ├── evaluate_model.py
│ │ ├── predict_model.py
│ │ └── train_model.py
│ │
│ └── visualization <- Scripts to create exploratory and results oriented visualizations
│   ├── create_xml.py
│   └── create_visualizations.py
```
### data folder 
In the data folder there is only 1 python script (`make_dataset.py`). The script unzips the Base.zip file and saves a Base.csv file. 

To run the script simply navigate to the directory in the command prompt. 
```bash
cd ..\..\Fraud-Hackathon\src\data
```

Run the following code.
```bash
python make_dataset.py
```

### features folder 
In the features folder there is only 1 python script (`build_features.py`). 

There are 7 functions in the script.

1) reverse_fillna(). The function replaces negative values with NaN. 
2) generate_new_features(). The function creates new columns for new features we are generating.
3) handle_missing_values(). The function first drops columns with high proportion of missing values and then drops the rows with missing values. 
4) handle_categorical_features(). The function filters out features that are of type String and encodes it. 
5) backward_stepwise_selection(). The function selects features using backward stepwise selection method.
6) MinMaxScaler(). Scale the numeric features 
7) SMOTE(). SMOTE does oversampling for the imbalance dataset.


To run the script simply navigate to the directory in the command prompt.
```bash
cd ..\..\Fraud-Hackathon\src\features
```

Run the following code.
```bash
python build_features.py
```

### models folder
In the models folder there are 3 python scripts (evaluate_model.py, predict_model.py, train_model.py). The order of running each script is train_model.py -> predict_model.py -> evaluate_model.py. 

### train model
`train_model.py` is the script to train the model of choice to specify the model of choice add -m <"Model name"> to the back of the command. The trained model will be saved as a pickle file under "..\..\Fraud-Hackaton\models\<model name.pkl>"

There is only 1 function in the script.

get_model(). Specify the model name when running the script. Some explamples are -m "AdaBoostClassifier", -m "DecisionTreeClassifier", -m "GaussianNB", -m "LogisticRegression", -m "XGBClassifier".

To run the script simply navigate to the directory in the command prompt.
```bash
cd ..\..\Fraud-Hackathon\src\models
```

Run the following code to train the model. Change the model choice accordingly  
```bash
python train_model.py -m <"Model name">
```
### predict model 
`predict_model.py` generates prediction of the test set usng the trained models. Specify the file path of the pickle file for the model at the end of the command. -s "..\..\Fraud-Hackathon\models\<model name.pkl>". 

load_model(). Loads the trained model to generate predictions. Specify the file path at the end of the command -s "..\..\Fraud-Hackathon\models\<model name.pkl>".  

Run the following code in the same directory to generate predictions. Change the trained model pickle file accordingly. 
```bash
python predict_model.py -s "..\..\Fraud-Hackathon\models\<model name.pkl>"
```
### evaluate model
`evaluate_model.py` generates evaluation score of the prediction against the ground truth. Specify the location of the prediction csv file. -yp "..\..\Fraud-Hackathon\models\prediction.csv"

Run the following code in the same directory to evaluate the model.  
```bash
python evaluate_model.py -yp "..\..\Fraud-Hackathon\models\prediction.csv"
```

### visualization folder
In the visualization folder there are 2 python script (create_visualization.py, create_xml.py). `create_visualization.py` will generate all the plots we created for Exploratory Data Analysis. `create_xml.py` will generate all the plots related to explanable ML techniques.

`create_visualizations.py`:
There are 5 main functions to generate plots.

1) fraud_distribution(). Creates the plot of the ratio of fraud to non-fraud 
2) missing_value_percentage(). Bar plots of percentage of missing values for each feature for fraud and non-fraud cases
3) box_plt_numeric_feature(). Box plot for each numeric feature
4) KDE_plot_numeric(). KDE plot for each numeric feature

To run the script simply navigate to the directory in the command prompt.
```bash
cd ..\..\Fraud-Hackathon\src\visualizations
```

Run the following code.
```bash
python create_visualizations.py
```

`create_xml.py` :
There is only 1 function in the script.

create_xml(). Specify the path to the saved train model when running the script. An example is -s ../../models/logistic.pkl.

To run the script simply navigate to the directory in the command prompt.
```bash
cd ..\..\Fraud-Hackathon\src\visualizations
```

Run the following code to train the model. Change the model choice accordingly  
```bash
python create_xml.py -s "../../models/logistic.pkl"
```