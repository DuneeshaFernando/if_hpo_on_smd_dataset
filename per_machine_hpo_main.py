import pandas as pd
from sklearn import preprocessing
import src.constants as const
from os.path import join
import numpy as np
from config import config as conf
from sklearn.ensemble import IsolationForest
from src.evaluation import Evaluation

import argparse
import random
import time
import optuna
import pymysql

pymysql.install_as_MySQLdb()

def combine_plotly_figs_to_html(plotly_figs, html_fname, include_plotlyjs="cdn"):
    with open(html_fname, "w") as f:
        f.write(plotly_figs[0].to_html(include_plotlyjs=include_plotlyjs))
        for fig in plotly_figs[1:]:
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))

def plot_optuna_default_graphs(optuna_study):
    history_plot = optuna.visualization.plot_optimization_history(optuna_study)
    parallel_plot = optuna.visualization.plot_parallel_coordinate(optuna_study)
    slice_plot = optuna.visualization.plot_slice(optuna_study)
    plot_list = [history_plot, parallel_plot, slice_plot]
    return plot_list

def average(lst):
    return sum(lst) / len(lst)

def kpoint(df, k):
    kpoint_column_titles = list(df.columns.values)
    for column_name in kpoint_column_titles:
        new_column_values = []
        for j in range(len(df[column_name])):
            range_start = max(0, j-k)
            range_end = max(0, j-1)
            if range_start == range_end == 0:
                value_range = df.loc[0,column_name]
                new_column_values.append(value_range)
            else:
                value_range = df.loc[range_start:range_end,column_name]
                new_column_values.append(average(value_range))
        df[column_name+"_kpoint"] = new_column_values
    return df

def main(config, file_name="machine-1-2"):
    # Pre-requisites
    min_max_scaler = preprocessing.MinMaxScaler()

    # setting seed for reproducibility
    np.random.seed(conf.SEED)

    dataset_path = const.SMD_DATASET_LOCATION

    # Read normal data
    normal_path = join(dataset_path,'train/')
    normal_data_file = join(normal_path, file_name+".csv")
    normal_df = pd.read_csv(normal_data_file)
    normal_df = normal_df.astype(float)

    # Read anomaly data
    anomaly_path = join(dataset_path,'test_with_labels/')
    anomaly_data_file = join(anomaly_path, file_name+".csv")
    anomaly_df = pd.read_csv(anomaly_data_file)
    # Separate out the anomaly labels before normalisation/standardization
    anomaly_df_labels = anomaly_df['Normal/Attack']
    anomaly_df = anomaly_df.drop(['Normal/Attack'], axis=1)
    anomaly_df = anomaly_df.astype(float)

    # Normalise/ standardize the normal and anomaly dataframe
    full_df = pd.concat([normal_df, anomaly_df])
    min_max_scaler.fit(full_df)

    # Add a new column called Normal/Attack with the value 0 to the normal_df, and add the anomaly_df_labels back to anomaly_df
    normal_df['Normal/Attack'] = 0
    anomaly_df['Normal/Attack'] = anomaly_df_labels

    # Create the train set and test set from the anomaly_df and normal_df
    # Split the anomaly_df into 2 parts
    anomaly_df_split_point = int(anomaly_df.shape[0] / 2)
    anomaly_df_part_1 = anomaly_df.iloc[:anomaly_df_split_point, :]
    anomaly_df_part_2 = anomaly_df.iloc[anomaly_df_split_point:, :]

    # Split the normal_df also such that anomaly_df_part_1 could replace a part of normal_df
    normal_df_split_point = normal_df.shape[0] - anomaly_df_split_point
    normal_df_part_1 = normal_df.iloc[:normal_df_split_point, :]
    normal_df_part_2 = normal_df.iloc[normal_df_split_point:, :]

    # Merge the split dataframes to form the train set and test set accordingly
    train_frames = [normal_df_part_1, anomaly_df_part_1]
    test_frames = [normal_df_part_2, anomaly_df_part_2]
    X_train = pd.concat(train_frames)
    X_test = pd.concat(test_frames)

    # Separate out the 'Normal/Attack' labels before normalisation/standardization
    y_train = X_train['Normal/Attack']
    X_train = X_train.drop(['Normal/Attack'], axis=1)
    y_test = X_test['Normal/Attack']
    X_test = X_test.drop(['Normal/Attack'], axis=1)

    # Normalise/ standardize the merged dataframe
    X_train_values = X_train.values
    X_train_values_scaled = min_max_scaler.transform(X_train_values)
    X_train_scaled = pd.DataFrame(X_train_values_scaled,
                                  columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col22', 'col23', 'col24', 'col25', 'col26', 'col27', 'col28', 'col29', 'col30', 'col31', 'col32', 'col33', 'col34', 'col35', 'col36', 'col37', 'col38'])
    X_test_values = X_test.values
    X_test_values_scaled = min_max_scaler.transform(X_test_values)
    X_test_scaled = pd.DataFrame(X_test_values_scaled,
                                 columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col22', 'col23', 'col24', 'col25', 'col26', 'col27', 'col28', 'col29', 'col30', 'col31', 'col32', 'col33', 'col34', 'col35', 'col36', 'col37', 'col38'])

    # Add k-point moving average new feature
    X_train_scaled = kpoint(X_train_scaled, config["k"])
    X_test_scaled = kpoint(X_test_scaled, config["k"])

    # Initialise the Isolation Forest model with the best hyper-parameters and train it using the train set
    if_model = IsolationForest(n_estimators=config["n_estimators"],
                               max_features=config["max_features"],
                               max_samples=config["max_samples"]).fit(X_train_scaled)

    X_test_scaled['y_pred'] = if_model.score_samples(X_test_scaled)
    thresholding_percentile = ((y_test.tolist().count(1.0)) / (len(y_test))) * 100
    threshold = np.percentile(X_test_scaled['y_pred'], [thresholding_percentile])[0]
    X_test_scaled['Normal/Attack'] = X_test_scaled['y_pred'] < threshold
    test_eval = Evaluation(y_test, X_test_scaled['Normal/Attack'])
    test_eval.print()

    return test_eval.auc

def objective(trial):
    params = dict()
    params["n_estimators"] = trial.suggest_int("n_estimators", 20, 500)
    params["max_features"] = trial.suggest_float("max_features", 0.1, 1.0, step=0.1)
    params["max_samples"] = trial.suggest_float("max_samples", 0.05, 1.0, step=0.05)
    params["k"] = trial.suggest_int("k", 20, 500)

    print(f"Initiating Run {trial.number} with params : {trial.params}")

    loss = main(params)
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna-db", type=str, help="Path to the Optuna Database file",
                        default="sqlite:///optuna.db")
    parser.add_argument("-n", "--optuna-study-name", type=str, help="Name of the optuna study",
                        default="duneesha_isolation_forest_run_1")
    args = parser.parse_args()

    # wait for some time to avoid overlapping run ids when running parallel
    wait_time = random.randint(0, 10) * 3
    print(f"Waiting for {wait_time} seconds before starting")
    time.sleep(wait_time)

    study = optuna.create_study(direction="maximize",
                                study_name=args.optuna_study_name,
                                storage=args.optuna_db,
                                load_if_exists=True,
                                )
    study.optimize(objective, n_trials=10) # When running locally, set n_trials as the no.of trial required

    # print best study
    best_trial = study.best_trial
    print(best_trial.params)

    plots = plot_optuna_default_graphs(study)

    combine_plotly_figs_to_html(plotly_figs=plots, html_fname="optimization_trial_plots/iforest_hpo.html")