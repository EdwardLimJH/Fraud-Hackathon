import argparse
from os.path import join as pathjoin
import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, fbeta_score, f1_score, average_precision_score, precision_recall_curve, confusion_matrix,balanced_accuracy_score


def load_model(model_filename):
    return pickle.load(open(model_filename, "rb"))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-yp", "--ypred_path", help="y_pred file path", required=True)
    parser.add_argument("-yt", "--ytest_path", help="y_test file path")
    parser.add_argument("-o", "--output_path", help="path to save evaluation results")
    args = parser.parse_args()

    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return args_dict


def evaluate_predictions():
    print("============= Reading in CSV data =============")    
    data_dir = pathjoin("..","..","data")
    DEFAULT_YTEST_PATH = pathjoin(data_dir,"processed","y_test.csv")
    y_test = pd.read_csv(parsed_args.get("ytest_path",DEFAULT_YTEST_PATH))
    y_pred = pd.read_csv(parsed_args.get("ypred_path"))
    y_pred = y_pred["fraud_bool"]
    print("============= Evaluating performance =============")    
    score_results = {}
    score_results["accuracy_score"] = accuracy_score(y_test, y_pred)
    score_results["classification_report"] = classification_report(y_test, y_pred)
    score_results["recall_score"] = recall_score(y_test, y_pred)
    score_results["precision_score"] = precision_score(y_test, y_pred)
    score_results["F2-score"] = fbeta_score(y_test, y_pred, beta =2)
    score_results["F1-score"] = f1_score(y_test, y_pred)
    score_results["average_precision_score"] = average_precision_score(y_test, y_pred)
    score_results["PR-AUC"] = precision_recall_curve(y_test, y_pred)
    score_results["balanced_accuracy_score"] = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    TN, FP, FN, TP = cm.ravel()
    TPR = TP/(TP+FN)
    FNR = FN/(TP+FN)
    score_results["TPR"] = TPR
    score_results["FNR"] = FNR

    print("============= Saving evaluation results to csv =============")    
    evaluation_results = pd.DataFrame(score_results)
    DEFAULT_EVALUATION_RESULTS_PATH = pathjoin(data_dir,"predictions","evaluation.csv")
    evaluation_results.to_csv(parsed_args.get("output_path",DEFAULT_EVALUATION_RESULTS_PATH), index=False)
    

if __name__ == "__main__":
    parsed_args = parse_arguments()
    evaluate_predictions()