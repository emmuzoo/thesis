import os
import re
import click
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import pickle
from prefect import flow, task 



def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


#def split_data(X, y, test_size: float, random_state: float=42):
@task(
    name="split data", 
    tags=["data"], 
    retries=3, 
    retry_delay_seconds=60
)
def split_data(X, y, test_size: float,  val_size: float, random_state: float=42):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    #return X_train, X_test, y_train, y_test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

@click.command()
@click.option(
    '--data_path', 
    type=click.Path(exists=True),
    help="Location where the raw idbm data was saved")
@click.option(
    '--dest_path', 
    type=click.Path(), 
    help="Location where the resulting files will be saved")
@click.option('--test_size', default=0.2, help='Proporción del conjunto de prueba')
@click.option('--val_size', default=0.1, help='Proporción del conjunto de validación')
def run_split_data(data_path: str, dest_path: str, test_size: float, val_size: float):
    # Load parquet files
    X, y = load_pickle(data_path)

    # Fit the DictVectorizer and preprocess data
    X_train, X_temp, y_train, y_temp = split_data(X, y, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = split_data(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)
    
    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

if __name__ == '__main__':
    run_split_data()
