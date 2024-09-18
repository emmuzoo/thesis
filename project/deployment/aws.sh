export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com

export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_DEFAULT_REGION=

export MLFLOW_ARTIFACTS_BUCKET=mflow-artifacts-975050210764


export S3_BUCKET_NAME=${MLFLOW_ARTIFACTS_BUCKET}
export ARTIFACT_KEY=artifacts
export EXPERIMENT_ID=2
export RUN_ID=
export VECTORIZER_KEY=output/tfidfv.pkl


curl -X POST -H 'Content-Type: application/json' -d "{\"review\": \"I am sad\"}"  http://localhost:8000/predict


# Exports
export MLFLOW_TRACKING_URI=http://localhost:5000

export-experiment --experiment sent-analisis-xgboost-hyperopt \
                  --output-dir /workspaces/mlops-zoomcamp/project/tmp/export/sent-analisis-xgboost-hyperopt

export-experiment --experiment sent-analisis-xgboost-best-models \
                  --output-dir /workspaces/mlops-zoomcamp/project/tmp/export/sent-analisis-xgboost-best-models

Import experiment

export MLFLOW_TRACKING_URI=http://localhost:5001

import-experiment \
  --experiment-name sent-analisis-xgboost-hyperopt \
  --input-dir /workspaces/mlops-zoomcamp/project/tmp/export/sent-analisis-xgboost-hyperopt

import-experiment \
  --experiment-name sent-analisis-xgboost-best-models \
  --input-dir /workspaces/mlops-zoomcamp/project/tmp/export/sent-analisis-xgboost-best-models




aws s3 sync /workspaces/mlops-zoomcamp/project/output s3://${MLFLOW_ARTIFACTS_BUCKET}/output



    sentiment-classifier:
      image: sentiment-classifier:latest
      container_name: sentiment-classifier
      environment:
         ARTIFACT_ROOT_PATH: ${ARTIFACT_ROOT_PATH}
         EXPERIMENT_ID: ${EXPERIMENT_ID}
         RUN_ID: ${RUN_ID}
      ports:
        - "8080:8080"
      depends_on:
        - mlflow
        