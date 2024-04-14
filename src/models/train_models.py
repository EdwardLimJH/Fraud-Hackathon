import argparse
from os.path import join as pathjoin
from sklearn.utils import all_estimators
import xgboost
from sklearn.base import ClassifierMixin
import json
from json.decoder import JSONDecodeError
import pandas as pd
import pickle

class ModelNotSupported(Exception):
    def __init__(self):
        self.message = "The model input is not supported. Please try another classifier from sklearn or xgboost"
        super().__init__(self.message)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ix", "--xtrain_path", help="X_train file input")
    parser.add_argument("-iy", "--ytrain_path", help="X_train file input")
    parser.add_argument("-s", "--saved_model_path", help="Filepath to save trained model")
    parser.add_argument("-m", "--model_name", help="Model name. A classifier from sklearn or xgboost", required=True)
    parser.add_argument("-d", "--dictionary", help="Input model parameters as JSON string")
    args = parser.parse_args()

    # Convert the JSON string of model parameters to a dictionary
    try:
        if args.dictionary:
            json_string = args.dictionary
            json_string = json_string.replace(", ", ",").replace("{", '{"').replace(":",'":').replace(",",',"')
            print(json_string)
            args.dictionary = json.loads(json_string)
    except JSONDecodeError as err:
        print(args.dictionary)
        raise JSONDecodeError("Model parameter format incorrect.\nPlease enter a valid json string",
                              args.dictionary, err.pos)

    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return args_dict

def get_model(model_name):
    CLASSIFIERS = [
        est
        for est in all_estimators()
        if (issubclass(est[1], ClassifierMixin))
    ]

    CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
    CLASSIFIER_MAP = {name.lower():model for (name,model) in CLASSIFIERS}
    return CLASSIFIER_MAP.get(model_name.lower(), None)

def save_pkl(object_tosave, filename):
    pickle.dump(object_tosave, open(filename,"wb"))


def train_model():
    model = get_model(parsed_args.get("model_name"))
    if not model :
        raise ModelNotSupported

    data_dir = pathjoin("..","..","data")
    DEFAULT_XTRAIN_RESAMPLED_PATH = pathjoin(data_dir,"processed","X_train_resampled.csv")
    DEFAULT_YTRAIN_RESAMPLED_PATH = pathjoin(data_dir,"processed","y_train_resampled.csv")
    
    X_train =  pd.read_csv(parsed_args.get("xtrain_path", DEFAULT_XTRAIN_RESAMPLED_PATH))
    y_train = pd.read_csv(parsed_args.get("ytrain_path", DEFAULT_YTRAIN_RESAMPLED_PATH))

    print("============= Initiating model =============")
    model_params = parsed_args.get("dictionary")
    if model_params:
        model = model(**model_params)
    else:
        model = model()
    
    print("============= Training model =============")
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train, y_train)
    print("============= Finished training model =============")
    print("============= Saving trained model =============")
    DEFAULT_SAVE_MODEL_FILEPATH = pathjoin("..","..","models",f'{parsed_args.get("model_name")}.pkl',)
    save_pkl(model, parsed_args.get("saved_model_path", DEFAULT_SAVE_MODEL_FILEPATH))



if __name__ == "__main__":
    parsed_args = parse_arguments()
    train_model()