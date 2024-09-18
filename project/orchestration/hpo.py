import os
import click
# Load and Save
import pickle
import joblib
import pandas as pd
# xgboost
import xgboost as xgb
# sk-learn
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score
# mlflow, optuna
import mlflow
import optuna
from mlflow.models import infer_signature
from optuna.integration.mlflow import MLflowCallback
# Prefect
from prefect import flow, task 


@task(
    name="Setting up Mlflow",
    tags=["model"], 
    retries=3, 
    retry_delay_seconds=60
)
def mlflow_environment(experiment_name, remote_server_uri:str="http://127.0.0.1:5000"):
    mlflow.set_tracking_uri(remote_server_uri) #  connects to a tracking URI.
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    return experiment.experiment_id

def hpo_xgboost(X_train, y_train, X_valid, y_valid, experiment_id, run_name: str = "First hpo", num_trials:int=50):
    def objective(trial):
        with mlflow.start_run(nested=True):
            # Define hyperparameters
            params = {
                'objective': 'binary:logistic',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'eval_metric': 'logloss'
            }

            # Train XGBoost model
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
            evals = [(dtrain, 'train'), (dvalid, 'eval')]
            bst = xgb.train(params,
                            dtrain, evals=evals,
                            num_boost_round=100,
                            early_stopping_rounds=10,
                            verbose_eval=False)

            # Log to MLflow
            y_pred_valid_probs  = bst.predict(dvalid)
            y_pred_valid_labels = (y_pred_valid_probs > 0.5).astype(int)
            accuracy = accuracy_score(y_valid, y_pred_valid_labels)
            recall = recall_score(y_valid, y_pred_valid_labels)

            # Log to MLflow
            signature = infer_signature(X_valid[:5],
                                        y_pred_valid_probs[:5])
            mlflow.xgboost.log_model(bst, "model", signature=signature)
            mlflow.log_params(params)
            mlflow.log_metric("logloss", bst.best_score)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)

        return accuracy

    #experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"experiment_id: {experiment_id}, run_name: {run_name}, num_trials: {num_trials}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        study = optuna.create_study(study_name="sent_ana_prediction", direction='maximize')

        # Execute the hyperparameter optimization trials.
        study.optimize(objective, n_trials=num_trials)

        best_params = study.best_params
        best_accuracy = study.best_value

        # Registrar el estudio en MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric('best_accuracy', best_accuracy)

    return best_params, best_accuracy



@task(
    name="Train xgboost",
    tags=["model"], 
    retries=3, 
    retry_delay_seconds=60
)
def run_optimization(data_path: str, experiment_id, run_name: str, num_trials: int):

    # Load data
    train_path = os.path.join(data_path, "train.parquet")
    valid_path = os.path.join(data_path, "val.parquet")
    print(f"tain_path: {train_path}")
    print(f"val_path : {valid_path}")
    df_train = pd.read_parquet(train_path)
    df_valid = pd.read_parquet(valid_path)

    
    num_features = ['TIME', 'DISTANCE', 'DEPARTURE_DELAY']
    cat_features = ['ROUTE', 'T_DEPARTURE', 'T_ARRIVAL',  'CANCELLED', 'DIVERTED']
    stratify_colname = 'DELAY'
    X_train = df_train[num_features + cat_features]
    y_train = df_train[stratify_colname]
    X_valid = df_valid[num_features + cat_features]
    y_valid = df_valid[stratify_colname]

    hpo_xgboost(X_train, y_train, X_valid, y_valid, experiment_id, run_name, num_trials)