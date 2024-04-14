import argparse
from os.path import join as pathjoin
import pandas as pd
import pickle


def load_model(model_filename):
    return pickle.load(open(model_filename, "rb"))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ix", "--xtest_path", help="X_test file input")
    parser.add_argument("-o", "--output_path", help="path to save predictions output")
    parser.add_argument("-s", "--saved_model_path", help="Filepath to saved trained model", required=True)
    args = parser.parse_args()

    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return args_dict

def make_prediction():
    data_dir = pathjoin("..","..","data")
    DEFAULT_XTEST_PATH = pathjoin(data_dir,"processed","X_test.csv")
    print("============= Reading in CSV data =============")    
    X_test = pd.read_csv(parsed_args.get("xtest_path",DEFAULT_XTEST_PATH))
    print("============= Loading saved model =============")
    saved_model = load_model(parsed_args.get("saved_model_path"))
    trained_columns = saved_model.feature_names_in_.tolist()
    X_test = X_test[trained_columns]
    print("============= Making Predictions =============")
    predictions = saved_model.predict(X_test)
    positive_class_prob = saved_model.predict_proba(X_test)[:, 1]
    
    results = pd.DataFrame({"is_fraud":predictions,
                            "probability_of_fraud":positive_class_prob})
    print("============= Saving output into csv file =============")
    DEFAULT_OUTPUT_PATH = pathjoin("..","..","data","predictions","predictions.csv")
    results.to_csv(parsed_args.get("output_path", DEFAULT_OUTPUT_PATH), index=False)

if __name__ == "__main__":
    parsed_args = parse_arguments()
    make_prediction()