import os
import click
# Load and Save
import pickle
import joblib
# Xgboost
import xgboost as xgb
# Mlflow
import mlflow
from mlflow.models import infer_signature
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
# sk-learn
from sklearn.metrics import accuracy_score, recall_score
# Prefect
from prefect import flow, task 


HPO_EXPERIMENT_NAME = "setana-xgboost-hyperopt"
EXPERIMENT_NAME = "sentana-xgboost-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#mlflow.set_experiment(EXPERIMENT_NAME)
#mlflow.sklearn.autolog()

'''
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
'''

@task
def get_best_model(experiment_id, tracking_uri="http://127.0.0.1:5000"):
    client = MlflowClient(tracking_uri=tracking_uri)
    run = MlflowClient().search_runs(
        experiment_ids=experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.test_accuracy DESC"]
    )[0]
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/mlruns"
    model_src = RunsArtifactRepository.get_underlying_uri(model_uri)

    filter_string = "run_id='{}'".format(run_id)
    results = client.search_model_versions(filter_string)
    model_version=results[0].version

    return model_version, model_uri 

#def train_and_log_model(X_train, y_train, X_val, y_val, X_test, y_test, params, experiment_id):
#def train_and_log_model(data_path, params, experiment_id, run_name):
def train_and_log_model(data_path, params, experiment_id):
    try:
        print(f"[train] data_path: {data_path}, experiment_id: {experiment_id}")
        #X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        #X_val,   y_val   = load_pickle(os.path.join(data_path, "val.pkl"))
        #X_test,  y_test  = load_pickle(os.path.join(data_path, "test.pkl"))
        X_train, y_train = joblib.load(os.path.join(data_path, "train.pkl"))
        X_val,   y_val   = joblib.load(os.path.join(data_path, "val.pkl"))
        X_test,  y_test  = joblib.load(os.path.join(data_path, "test.pkl"))

        #mlflow.xgboost.autolog()
        #print(params)
        #with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        with mlflow.start_run(experiment_id=experiment_id):
            #for param in RF_PARAMS:
            #    params[param] = int(params[param])
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_val, label=y_val)
            dtest  = xgb.DMatrix(X_test, label=y_test)
            evals = [(dtrain, 'train'), (dvalid, 'eval')]
            bst = xgb.train(params, dtrain, evals=evals, num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)

            # Evaluate val
            y_pred_val = bst.predict(dvalid)
            y_pred_val_labels = (y_pred_val > 0.5).astype(int)
            val_accuracy = accuracy_score(y_val, y_pred_val_labels)
            val_recall = recall_score(y_val, y_pred_val_labels)

            # Evaluate test
            y_pred_test = bst.predict(dtest)
            y_pred_test_labels = (y_pred_test > 0.5).astype(int)
            test_accuracy = accuracy_score(y_test, y_pred_test_labels)
            test_recall = recall_score(y_test, y_pred_test_labels)

            # Log to MLflow
            signature = infer_signature(X_train, y_pred_val)
            mlflow.xgboost.log_model(bst, "model", signature=signature)
            mlflow.log_params(params)
            mlflow.log_metric("logloss", bst.best_score)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("val_recall", val_recall)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_recall", test_recall)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


#def run_register_model(X_train, y_train, X_val, y_val, X_test, y_test, 
@task(
    name="Register model", 
    tags=["model"], 
    #retries=3, 
    log_prints=True,
    #retry_delay_seconds=60
)
def run_register_model(data_path, 
                       hpo_experiment_id, experiment_id, run_name, top_n: int, 
                       remote_server_uri:str="http://127.0.0.1:5000"):
    try:
        print(f"hpo_experiment_id: {hpo_experiment_id},  experiment_id: {experiment_id}, run_name: {run_name}. top_n: {top_n}")

        mlflow.set_tracking_uri(remote_server_uri)
        #mlflow.set_experiment(experiment_name)
        #best_experiment = client.get_experiment_by_name(experiment_name)
        client = MlflowClient()

        # Retrieve the top_n model runs and log the models
        # hpo_experiment = client.get_experiment_by_name(hpo_experiment_name)
        runs = client.search_runs(
            experiment_ids=hpo_experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=top_n,
            order_by=["metrics.accuracy DESC"]
        )
        for run in runs:
            #train_and_log_model(X_train, y_train, X_val, y_val, X_test, y_test, 
            #                    params=run.data.params, experiment_id=experiment_id)
            #train_and_log_model(data_path, params=run.data.params, experiment_id=experiment_id, run_name=run_name)
            train_and_log_model(data_path, params=run.data.params, experiment_id=experiment_id)

        # Select the model with the lowest test RMSE
        
        print(f"Registering model")
        # best_run = client.search_runs( ...  )[0]
        best_run = client.search_runs(
            experiment_ids=experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=top_n,
            order_by=["metrics.test_accuracy DESC"]
        )[0]
        
        # Register the best model
        # mlflow.register_model( ... )
        mlflow.register_model(
            model_uri=f"runs:/{best_run.info.run_id}/model",
            name="xgboost-model"
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == '__main__':
    run_register_model()
