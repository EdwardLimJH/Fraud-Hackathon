import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join as pathjoin
import warnings
from sklearn.feature_selection import chi2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",'--raw_csv_path', help='Input csv file path')
    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return args_dict

# Check balance of data
def fraud_distribution(df, save_path):
    count_of_0 = (df['fraud_bool'] == 0).sum()
    count_of_1 = (df['fraud_bool'] == 1).sum()

    # Plotting the pie chart
    labels = ['Non-Fraud', 'Fraud']
    sizes = [count_of_0, count_of_1]
    colors = ['blue', 'red']

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Fraud vs Non-Fraud Transactions')

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(save_path)
    plt.close()

# Bar plots of percentage of missing values for each feature for fraud and non-fraud cases
def missing_value_percentage(df, save_path):
    # Features with missing values represented by negative values according to documentation
    missing_features = ['prev_address_months_count', 'current_address_months_count', 'intended_balcon_amount', 
                        'bank_months_count', 'session_length_in_minutes', 'device_distinct_emails_8w']
    missing_vals_df = pd.DataFrame()

    # Replace negative values with NaN, then calculate the percentage of missing values for fraud and non-fraud cases
    for feature in missing_features:
        df[feature] = df[feature].apply(lambda x: x if x >= 0 else np.nan)
        missing_vals_df[feature] = df.groupby('fraud_bool')[feature].apply(lambda x: x.isnull().mean() * 100)

    missing_vals_df = missing_vals_df.reset_index().melt(id_vars='fraud_bool', var_name='feature', value_name='missing_percentage')

    pos = np.arange(len(missing_features))
    bar_width = 0.35
    mask = missing_vals_df['fraud_bool'] == 1

    plt.bar(pos, missing_vals_df[~mask]['missing_percentage'], bar_width, color='blue', label='Not Fraud')
    plt.bar(pos + bar_width, missing_vals_df[mask]['missing_percentage'], bar_width, color='red', label='Fraud')

    plt.xticks(pos + bar_width / 2, missing_features, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Missing Values %")
    plt.title("Percentage of Missing Values of Features by Fraud Status")
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(save_path)
    plt.close()

# Create a new df 
def new_df(df):
    features_to_drop = ['prev_address_months_count', 'intended_balcon_amount', 'bank_months_count']
    no_missing_df = df.drop(features_to_drop, axis=1)
    # Drop rows with missing values
    no_missing_df.dropna(inplace=True)
    # I am counting 'income' and 'customer_age' as numeric features
    numeric_features = [feature for feature in no_missing_df.columns if no_missing_df[feature].nunique() >= 9]

    return no_missing_df, numeric_features
    

# Box plot for each numeric feature
def box_plt_numeric_feature(no_missing_df,numeric_features , save_path):
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))

    # Box plot for each numeric feature
    fig.suptitle("Box Plot of Numeric Features by Fraud Status")
    for i, feature in enumerate(numeric_features):
        ax = axes[i // 3, i % 3]
        sns.boxplot(x='fraud_bool', y=feature, data=no_missing_df, ax=ax, palette=['blue', 'red'])
        ax.set_xlabel("Fraud Status")
        ax.set_ylabel(feature)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# KDE plot for each numeric feature
def KDE_plot_numeric(no_missing_df, numeric_features, save_path):
    warnings.filterwarnings("ignore", "use_inf_as_na")

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))

    # KDE plot for each numeric feature
    fig.suptitle("KDE Plot of Numeric Features by Fraud Status")
    for i, feature in enumerate(numeric_features):
        ax = axes[i // 3, i % 3]
        sns.kdeplot(no_missing_df[no_missing_df['fraud_bool'] == 0][feature], ax=ax, color='blue', fill=True, label='Not Fraud', legend=True)
        sns.kdeplot(no_missing_df[no_missing_df['fraud_bool'] == 1][feature], ax=ax, color='red', fill=True, label='Fraud', legend=True)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Create new chi2 df
def chi2_dataframe(no_missing_df, numeric_features):
    encoded_features = [feature for feature in no_missing_df.columns if no_missing_df[feature].dtype == 'object']
    encoded_df = pd.get_dummies(no_missing_df, columns=encoded_features, drop_first=True)
    categorical_df = encoded_df.drop(columns=numeric_features)
    # # Temporarily drop missing values in 'device_distinct_emails_8w' for chi-squared test
    # categorical_df = categorical_df[categorical_df['device_distinct_emails_8w'].notnull()]
    X_categorical_df = categorical_df.drop(columns=['fraud_bool'])
    y_categorical_df = categorical_df['fraud_bool']
    chi2_results = chi2(X_categorical_df, y_categorical_df)
    chi2_df = pd.DataFrame(chi2_results, columns=X_categorical_df.columns, index=['chi2-statistic', 'p-value']).transpose()
    chi2_df = chi2_df.sort_values(by='chi2-statistic', ascending=False)
    return chi2_df

# Plot chi-square statistics
def chi_square_stats(chi2_df, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    chi2_df['chi2-statistic'].plot(kind='bar', ax=ax)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Chi-square Statistic")
    ax.set_title("Chi-square Statistics of Categorical Features")
    plt.savefig(save_path)
    plt.close()

# Plot p-values
def p_values_plot(chi2_df, save_path):
    PVALUE_THRESHOLD = 0.05
    fig, ax = plt.subplots(figsize=(8, 6))
    chi2_df['p-value'].plot(kind='bar', ax=ax)
    ax.set_xlabel("Feature")
    ax.set_ylabel("P-value")
    ax.set_title("P-values of Categorical Features")
    plt.axhline(y=PVALUE_THRESHOLD, color='r', linestyle='--')
    plt.savefig(save_path)
    plt.close()

    
def create_EDA():
    data_dir = pathjoin("..","..","data")
    save_dir = pathjoin("..","..","reports")
    print("============= Parsing Arguments =============")
    DEAFULT_RAW_CSV_PATH = pathjoin(data_dir,"raw","base.csv")
    raw_csv_path = parsed_args.get("raw_csv_path", DEAFULT_RAW_CSV_PATH)
    df = pd.read_csv(raw_csv_path, usecols=lambda x: x != 'device_fraud_count')
    print("============= Plotting and saving fraud distribution =============")
    DEFAULT_FRAUD_DISTRIBUTION_PATH = pathjoin(save_dir,"figures","fraud_distribution.png")
    fraud_distribution(df, DEFAULT_FRAUD_DISTRIBUTION_PATH)
    print("============= Plotting and saving missing value percentage =============")
    DEFAULT_MISSING_VALUE_PERCENTAGE_PATH = pathjoin(save_dir,"figures","missing_value_percentage.png")
    missing_value_percentage(df, DEFAULT_MISSING_VALUE_PERCENTAGE_PATH)
    print("============= Creating no missing values df =============")
    no_missing_df, numeric_features = new_df(df)
    print("============= Plotting and saving box plots for numeric features =============")
    DEFAULT_NUMERIC_FEATURE_BOXPLOT_PATH = pathjoin(save_dir,"figures","numeric_feature_boxplot.png")
    box_plt_numeric_feature(no_missing_df, numeric_features, DEFAULT_NUMERIC_FEATURE_BOXPLOT_PATH)
    print("============= Plotting and saving KDE plots for numeric features (Might take a while) =============")
    DEFAULT_NUMERIC_FEATURE_KDE_PATH = pathjoin(save_dir,"figures","numeric_feature_KDE.png")
    KDE_plot_numeric(no_missing_df, numeric_features, DEFAULT_NUMERIC_FEATURE_KDE_PATH)
    print("============= Creating chi2 df =============")
    chi2_df = chi2_dataframe(no_missing_df, numeric_features)
    print("============= Plotting and saving chi sqaure statistics =============")
    DEFAULT_CHI_SQUARE_STATS_PATH = pathjoin(save_dir,"figures","chi_square_stats.png")
    chi_square_stats(chi2_df, DEFAULT_CHI_SQUARE_STATS_PATH)
    print("============= Plotting and saving p value plot =============")
    DEFAULT_P_VALUE_PLOT_PATH = pathjoin(save_dir,"figures","p_value_plot.png")
    p_values_plot(chi2_df, DEFAULT_P_VALUE_PLOT_PATH)


if __name__ == "__main__":
    parsed_args = parse_arguments()
    create_EDA()

