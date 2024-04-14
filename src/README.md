# Documentation of Scripts 

## Folder stucture 

```bash
├── src <- Source code for use in this project.
│ ├── __init__.py <- Makes src a Python module
│ │
│ ├── data <- Scripts to download or generate data
│ │ └── make_dataset.py
│ │
│ ├── features <- Scripts to turn raw data into features for modeling
│ │ └── build_features.py
│ │
│ ├── models <- Scripts to train models and then use trained models to make
│ │ │ predictions
│ │ ├── predict_model.py
│ │ └── train_model.py
│ │
│ └── visualization <- Scripts to create exploratory and results orientedvisualizations
│   └── visualize.py
```
### data folder 
In the data data folder there is only 1 python script. The script unzips the Base file. To run the script simply navigate to the directory in the command prompt and run the following code.

```bash
python make_dataset.py
```