# Documentation of Scripts 

## Folder stucture 

```bash
├── src 
│ ├── __init__.py 
│ │
│ ├── data 
│ │ └── make_dataset.py
│ │
│ ├── features 
│ │ └── build_features.py
│ │
│ ├── models 
│ │ ├── evaluate_model.py
│ │ ├── predict_model.py
│ │ └── train_model.py
│ │
│ └── visualization 
│   └── visualize.py
```
### data folder 
In the data folder there is only 1 python script (make_dataset.py). The script unzips the Base file. 

To run the script simply navigate to the directory in the command prompt. 
```bash
cd ..\..\Fraud-Hackathon\src\data
```

Run the following code.
```bash
python make_dataset.py
```

### features folder 
In the features folder there is only 1 python script (build_features.py). The script processes the Base.csv dataset by generating new features, handling missing values, handiling categorical data, splitting into train test set, scaling numerical features, feature selection, and resampling of data. 

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
train_model.py is the script to train the model of choice to specify the model of choice add -m <"Model name"> to the back of the command.
predict_model.py generates prediction of the test set usng the trained models. Specify the file path of the pickle file for the model at the end of the command. -s "...\...\models\<model name.pkl>"
evaluate_model.py generates evaluation score of the prediction against the ground truth. Specify the location of the prediction csv file. -yp "...\data\predicitons\prediction.csv"

To run the script simply navigate to the directory in the command prompt.
```bash
cd ..\..\Fraud-Hackathon\src\features
```

Run the following code to train the model. Change the model choice accordingly  
```bash
python train_model.py -m <"Model name">
```

Run the following code to generate predictions. Change the trained model pickle file accordingly. 
```bash
python predict_model.py -s "...\...\models\<model name.pkl>"
```

Run the following code to evaluate the model.  
```bash
python evaluate_model.py -yp "...\data\predicitons\prediction.csv"
```
