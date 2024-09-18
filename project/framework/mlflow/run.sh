
#mlflow server --backend-store-uri sqlite:///mlflow.db \
#              --default-artifact-root artifacts

export MLFLOW_PORT=5000
export MLFLOW_DIR=./data
export MLFLOW_BACKEND_STORE_URI=sqlite:///data/mlflow.db

nohup mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
                    --default-artifact-root $MLFLOW_DIR/artifacts \
                    --host 0.0.0.0 --port 5000 > $MLFLOW_DIR/output.log 2>&1 &
