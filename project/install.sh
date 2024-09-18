pip install mlflow
pip install boto3
pip install click
pip install nltk
pip install xgboost==1.7.6
pip install optuna
pip install optuna-integration


pip install mlflow-export-import
pip install awscli

# Mlflow 
pip install -r requirements. txt

mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root artifacts

curl -X POST -H 'Content-Type: application/json' -d "{\"review\": \"I am sad\"}"  http://localhost:8000/predict


# Prefect
pip install -U prefect
prefect server start

prefect deployment build -n sentiment-analysis-deployment ml_pipeline.py:sentiment_analysis_flow -o sentiment_analysis_deployment.yaml

prefect deployment apply sentiment_analysis_deployment.yaml


prefect deployment run 'Sentiment-Analysis-Flow/my-first-deployment'




#prefect agent start --pool default-agent-pool --work-queue credit_risk_model_queue
prefect deployment run sentiment_analysis_flow/sentiment-analysis-deployment
prefect deployment run sentiment-analysis-flow/sentiment-analysis-deployment
prefect agent start -q "default"


# Mage
git clone https://github.com/mage-ai/mlops.git
cd mlops
./scripts/start.sh
docker compose stop



