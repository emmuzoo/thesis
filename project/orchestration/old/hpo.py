import os
import click
# Load and Save
import pickle
import joblib
# xgboost
import xgboost as xgb
# sk-learn
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score
# mlflow, optuna
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
# Prefect
from prefect import flow, task 


def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

HPO_EXPERIMENT_NAME = "sent-analisis-xgboost-hyperopt"
TRACKING_URI = "http://127.0.0.1:5000"
#mlflow.set_tracking_uri(TRACKING_URI)
#experiment_id = get_or_create_experiment(experiment_name)
#mlflow.set_experiment(EXPERIMENT_NAME)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


#def objective(trial, X_train, y_train, X_val, y_val):

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



def hpo_xgboost(X_train, y_train, X_val, y_val, experiment_id, run_name: str = "First hpo", num_trials:int=50):
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
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dvalid, 'eval')]
            bst = xgb.train(params, dtrain, evals=evals, num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)

            # Log to MLflow
            y_pred_val = bst.predict(dvalid)
            y_pred_val_labels = (y_pred_val > 0.5).astype(int)
            accuracy = accuracy_score(y_val, y_pred_val_labels)
            recall = recall_score(y_val, y_pred_val_labels)

            # Log to MLflow
            mlflow.xgboost.log_model(bst, "model")
            mlflow.log_params(params)
            mlflow.log_metric("logloss", bst.best_score)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)

        return accuracy
    
    #experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"experiment_id: {experiment_id}, run_name: {run_name}, num_trials: {num_trials}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
    #with mlflow.start_run(run_name = "Fisrt attempt", nested=True):
        # Initialize the Optuna study
        study = optuna.create_study(study_name="sent_ana_prediction", direction='maximize')
        
        # Execute the hyperparameter optimization trials.s
        #study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=num_trials)
        #study.optimize(objective, n_trials=num_trials, callbacks=[champion_callback])
        study.optimize(objective, n_trials=num_trials)

        best_params = study.best_params
        best_accuracy = study.best_value

        # Registrar el estudio en MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric('best_accuracy', best_accuracy)

    return best_params, best_accuracy


'''
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
@click.option(
    "--run_name",
    default="First hpo",
    help="The number of parameter evaluations for the optimizer to explore"
)
'''
@task(
    name="Train xgboost",
    tags=["model"], 
    retries=3, 
    retry_delay_seconds=60
)
def run_optimization(data_path: str, experiment_id, run_name: str, num_trials: int):

    X_train, y_train = joblib.load(os.path.join(data_path, "train.pkl"))
    X_val, y_val = joblib.load(os.path.join(data_path, "val.pkl"))


    #experiment.experiment_id = mlflow_environment(HPO_EXPERIMENT_NAME, TRACKING_URI)
    #hpo_xgboost(X_train, y_train, X_val, y_val, experiment.experiment_id, run_name, num_trials)
    hpo_xgboost(X_train, y_train, X_val, y_val, experiment_id, run_name, num_trials)


if __name__ == '__main__':
    run_optimization()
