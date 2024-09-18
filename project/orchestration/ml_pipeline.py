import os, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Prefect
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.infrastructure import Process
# tasks
from preprocess_data import run_data_prep
from split import split_data
from hpo import run_optimization, hpo_xgboost, mlflow_environment
from register_model import run_register_model



TRACKING_URI = "http://127.0.0.1:5000"

@task(
    name="read data", 
    tags=["data"], 
    retries=3, 
    retry_delay_seconds=60
)
def read_data(filename='Womens Clothing E-Commerce Reviews.csv'):
    data = pd.read_csv(filename)
    print("Data loaded.\n\n")
    return data


@flow(
    #name="Sentiment-Analysis-Flow",
    description="A flow to run the pipeline for the customer sentiment analysis",
    task_runner=SequentialTaskRunner()
)
def flight_delays_flow(raw_dataset_path: str, data_path:str, raw_dataset_year: int, raw_dataset_start_month:int, raw_dataset_end_month:int,
                            max_features:int, test_size: float, val_size: float, 
                            remote_server_uri: str, hpo_experiment_name:str, best_experiment_name:str, 
                            num_trials:int, top_n:int):

    # Load data
    #raw_dataset_path = "/workspaces/mlops-zoomcamp/project/raw/IMDB Dataset.csv"
    #df = read_data(raw_dataset_path)

    # Preprocessing
    #max_features = 5000
    #tfidfv = TfidfVectorizer(max_features=max_features)
    #X, y = preprocess(df, tfidfv)
    raw_datasets = []
    for month in range(raw_dataset_start_month, raw_dataset_end_month + 1):
        dataset_file = os.path.join(raw_dataset_path, f"raw_flightdata_{raw_dataset_year}-{month:02}.parquet")
        print(f"dataset_file: {dataset_file}")
        raw_datasets.append(dataset_file)
    #run_data_prep(raw_datasets, data_path, test_size)

    # Split data
    #test_size = 0.2
    #val_size = 0.1
    #random_states = 42
    #X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size, val_size, random_states)

    # HPO
    hpo_experiment_id = mlflow_environment(hpo_experiment_name, remote_server_uri)
    run_name = "Test sent-ana hpo"
    #num_trials = 10
    #run_optimization(data_path, hpo_experiment_id, run_name, num_trials)

    # Regster model
    best_experiment_id = mlflow_environment(best_experiment_name, remote_server_uri)
    run_name = "Test register"
    #run_register_model(X_train, y_train, X_val, y_val, X_test, y_test, hpo_experiment_id, best_experiment_id, top_n)
    run_register_model(data_path, hpo_experiment_id, best_experiment_id, top_n, remote_server_uri)


if __name__ == '__main__':
    flight_delays_flow.serve(name="flight-delays-deployment")