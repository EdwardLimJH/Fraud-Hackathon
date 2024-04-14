import argparse
import pandas as pd
import numpy as np
from os.path import join as pathjoin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",'--raw_csv_path', help='Input csv file path')
    parser.add_argument("-ttss",'--tts_seed', type=int, help='Random State for train-test split')
    parser.add_argument("-r",'--reverse_na_filename', help='Filepath to save interrim data (reverse missing values)')
    parser.add_argument("-g",'--new_features_filename', help='Filepath to save interrim data (generate new features)')
    parser.add_argument("-n",'--no_missing_filename', help='Filepath to save interrim data (Handle missing values)')
    parser.add_argument("-e",'--encoded_filename', help='Filepath to save interrim data (Encode categorical features)')
    parser.add_argument("-xn",'--xtrain_no_resample_path', help='File path to save Processed X_train')
    parser.add_argument("-yn",'--ytrain_no_resample_path', help='File path to save Processed y_train')
    parser.add_argument("-bsst",'--bss_threshold', type=float, help='Backward Stepwise Selection threshold')
    parser.add_argument("-sr",'--smote_sampling', type=float, help='Sampling ratio for smote')
    parser.add_argument("-ss",'--smote_seed', type=int, help='Random State for train-test split')
    parser.add_argument("-xtr",'--xtrain_resampled_path', help='File path to save Processed and resampled X_train')
    parser.add_argument("-ytr",'--ytrain_resampled_path', help='File path to save Processed and resampled y_train')
    parser.add_argument("-xt",'--xtest_path', help='File path to save Processed X_test')
    parser.add_argument("-yt",'--ytest_path', help='File path to save Processed y_test')
    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return args_dict

def reverse_fillna(df):
    missing_features = ['prev_address_months_count', 'current_address_months_count', 'intended_balcon_amount', 
                    'bank_months_count', 'session_length_in_minutes', 'device_distinct_emails_8w']
    # Replace negative values with NaN
    for feature in missing_features:
        df[feature] = df[feature].apply(lambda x: x if x >= 0 else np.nan)
    return df


def generate_new_features(df):
    features_to_drop = ['prev_address_months_count', 
                        'intended_balcon_amount', 
                        'bank_months_count']

    for col in features_to_drop:
        missing_column_name = f'{col}_missing'
        df[missing_column_name] = np.where(df[col].isna(), 1, 0)
    return df

def handle_missing_values(df):
    features_to_drop = ['prev_address_months_count', 
                            'intended_balcon_amount', 
                            'bank_months_count']
    # drop columns with high proportions of missing values
    df.drop(features_to_drop, axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True) # drop rows containing NaN values
    return df

def handle_categorical_features(df):
    # Only features with String data type need to be encoded
    encoded_features = [feature for feature in df.columns if df[feature].dtype == 'object']
    df = pd.get_dummies(df, columns=encoded_features, drop_first=True, dtype=int)
    return df

def save_pkl(object_tosave, filename):
    pickle.dump(object_tosave, open(filename,"wb"))


def backward_stepwise_selection(X, y, p_threshold=0.05):
    features = X.columns.tolist()
    num_features = len(features)
    
    for i in range(num_features, 0, -1):
        model = sm.Logit(y, X[features]).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > p_threshold:
            remove_feature = p_values.idxmax()
            print(f"Removing '{remove_feature}' with p-value: {max_p_value:.4f}")
            features.remove(remove_feature)
        else:
            break   
    return features


def process_dataset():
    data_dir = pathjoin("..","..","data")
    print("============= Parsing Arguments =============")
    DEAFULT_RAW_CSV_PATH = pathjoin(data_dir,"raw","base.csv")
    raw_csv_path = parsed_args.get("raw_csv_path", DEAFULT_RAW_CSV_PATH)
    print("============= Reading in raw csv file =============")
    df = pd.read_csv(raw_csv_path, usecols=lambda x: x != 'device_fraud_count')
    print("============= Reversing missing values =============")
    df = reverse_fillna(df)
    print("============= Saving interim data to csv =============")
    DEFAULT_REVERSE_NA_PATH = pathjoin(data_dir,"interim","reverse_na.csv")
    df.to_csv(parsed_args.get("reverse_na_filename", DEFAULT_REVERSE_NA_PATH), index=False)
    print("============= Generating New features =============")
    df = generate_new_features(df)
    print("============= Saving interim data to csv =============")
    DEFAULT_NEW_FEATURES_PATH = pathjoin(data_dir,"interim","new_features.csv")
    df.to_csv(parsed_args.get("new_features_filename", DEFAULT_NEW_FEATURES_PATH), index=False)
    print("============= Handle missing values =============")
    df = handle_missing_values(df)
    DEFAULT_NO_MISSING_PATH = pathjoin(data_dir,"interim","no_missing.csv")
    print("============= Saving interim data to csv =============")
    df.to_csv(parsed_args.get("no_missing_filename", DEFAULT_NO_MISSING_PATH), index=False)
    print("============= Encode categorical features =============")
    df = handle_categorical_features(df)
    print("============= Saving interim data to csv =============")
    DEFAULT_ENCODED_PATH = pathjoin(data_dir,"interim","processed.csv")
    df.to_csv(parsed_args.get("encoded_filename", DEFAULT_ENCODED_PATH), index=False)
    # Separate the feature matrix and target variable
    X = df.drop('fraud_bool', axis=1)
    y = df['fraud_bool']

    print("============= Train Test Split =============")
    DEFAULT_TTS_SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=parsed_args.get("tts_seed", DEFAULT_TTS_SEED), 
                                                        stratify=y)

    numeric_features = ['income', 'name_email_similarity', 'current_address_months_count', 'customer_age', 'days_since_request', 'zip_count_4w', 'velocity_6h', 'velocity_24h', 
                    'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'proposed_credit_limit', 'session_length_in_minutes']

    scaler = MinMaxScaler()
    print("============= Scaling Numeric features =============")
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    print("============= Saving MinMaxScaler as a pickle object =============")
    save_pkl(scaler, pathjoin(data_dir,"processed","min_max_scaler.pkl"))
    
    print("============= Running Backward stepwise Selection =============")
    DEFAULT_BSS_THRESHOLD = 0.05
    selected_features = backward_stepwise_selection(X_train, y_train, 
                                                    parsed_args.get("backward_stepwise_selection_threshold", 
                                                                    DEFAULT_BSS_THRESHOLD))
    
    X_train = X_train[selected_features]
    print("============= Saving processed X_train data before resampling to csv =============")
    DEAFULT_XTRAIN_NO_RESAMPLE_PATH = pathjoin(data_dir,"processed","X_train_preresampling.csv")
    X_train.to_csv(parsed_args.get("xtrain_no_resample_path",DEAFULT_XTRAIN_NO_RESAMPLE_PATH), index=False)
    print("============= Saving processed y_train data before resampling to csv =============")
    DEFAULT_YTRAIN_NO_RESAMPLE_PATH = pathjoin(data_dir,"processed","y_train_preresampling.csv")
    y_train.to_csv(parsed_args.get("ytrain_no_resample_path", DEFAULT_YTRAIN_NO_RESAMPLE_PATH), index=False)
    
    DEFAULT_SMOTE_SEED = 42
    DEFAULT_SMOTE_SAMPLING = 0.666 #ratio of minority:majority 40:60
    smote = SMOTE(random_state=parsed_args.get("smote_seed", DEFAULT_SMOTE_SEED), 
                  sampling_strategy = parsed_args.get("smote_sampling", DEFAULT_SMOTE_SAMPLING)) 
    Xt_resampled_SMOTE, yt_resampled_SMOTE = smote.fit_resample(X_train, y_train)
    ratio_SMOTE = yt_resampled_SMOTE.value_counts() / len(yt_resampled_SMOTE) * 100
    print(f'% of non-fraud class in resampled data: {round(ratio_SMOTE[0],3)}%\n% of fraud class in resampled data: {round(ratio_SMOTE[1],3)}%')
    
    print("============= Saving final processed X_train_resampled data to csv =============")
    DEFAULT_XTRAIN_RESAMPLED_PATH = pathjoin(data_dir,"processed","X_train_resampled.csv")
    Xt_resampled_SMOTE.to_csv(parsed_args.get("xtrain_resampled_path", DEFAULT_XTRAIN_RESAMPLED_PATH),
                              index=False)
    print("============= Saving final processed y_train_resampled data to csv =============")
    DEFAULT_YTRAIN_RESAMPLED_PATH = pathjoin(data_dir,"processed","y_train_resampled.csv")
    yt_resampled_SMOTE.to_csv(parsed_args.get("ytrain_resampled_path", DEFAULT_YTRAIN_RESAMPLED_PATH),
                              index=False)
    
    X_test = X_test[selected_features]
    print("============= Saving final processed X_test data to csv =============")
    DEFAULT_XTEST_PATH = pathjoin(data_dir,"processed","X_test.csv")
    X_test.to_csv(parsed_args.get("xtest_path", DEFAULT_XTEST_PATH), index=False)
    print("============= Saving final processed y_test data to csv =============")
    DEFAULT_YTEST_PATH = pathjoin(data_dir,"processed","y_test.csv")
    y_test.to_csv(parsed_args.get("ytest_path", DEFAULT_YTEST_PATH), index=False)

if __name__ == "__main__":
    parsed_args = parse_arguments()
    process_dataset()