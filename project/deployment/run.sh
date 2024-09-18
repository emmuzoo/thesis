#!/bin/bash

docker run -d --name sentiment-classifier-container \
    -p 8000:8000 \
    -e MODEL_URI="models:/sentiment-analysis/1" \
    -e EXPERIMENT_ID="2" \
    -e RUN_ID="25ab5596ca7249f3bd4211d2198d94ef" \
    -e BASE_PATH="/workspaces/mlops-zoomcamp/project/framework/mlflow/artifacts" \
    sentiment-classifier