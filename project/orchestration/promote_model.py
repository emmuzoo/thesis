
import os
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

@task
def promote_best_model(model_version, model_name, tracking_uri="http://127.0.0.1:5000"):
    new_stage = "Production"
    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=new_stage,
            archive_existing_versions=False
    )