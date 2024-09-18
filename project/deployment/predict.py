import os
# Flask
from flask import Flask, request, jsonify
# Store
import joblib
# Xgboost
import xgboost as xgb
import numpy as np
# Mlflow
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
# Parser
from bs4 import BeautifulSoup
import re
import boto3
# nltk
import nltk
from nltk.corpus import stopwords


# Descargar stopwords si es necesario
nltk.download('stopwords')
nltk.download('punkt')

# Environments
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID")
RUN_ID = os.environ.get("RUN_ID")
ARTIFACT_KEY = os.environ.get("ARTIFACT_KEY")

VECTORIZER_KEY = os.environ.get("VECTORIZER_KEY")  # URI del vectorizador en S3

#S3_BUCKET = os.environ.get("S3_BUCKET")
#MODEL_URI = os.environ.get("MODEL_URI")  # URI del modelo registrado en MLflow
#VECTORIZER_URI = os.environ.get("VECTORIZER_URI")  # URI del vectorizador en S3

#logged_model = f"s3://{S3_BUCKET_NAME}/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo y el vectorizador TF-IDF
#EXPERIMENT_NAME = "sentana-xgboost-best-models"
#MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)



# Recuperar el mejor modelo del experimento "setana-xgboost-best-models"
def load_best_model(experiment_id, run_id):
    
    #model_uri = f"runs:/{best_run_id}/model"
    #model_uri = f"s3://{S3_BUCKET_NAME}/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"

    return mlflow.pyfunc.load_model(model_uri)

# Cargar el modelo al iniciar el servidor
#model = load_best_model(EXPERIMENT_ID, RUN_ID)

#model_uri = f"{ARTIFACT_ROOT_PATH}/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"
model_uri = f"s3://{S3_BUCKET_NAME}/{ARTIFACT_KEY}/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"
#model_uri = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(model_uri)
print(f"LOADED: {model_uri}")

#tfidf_path = '/workspaces/mlops-zoomcamp/project/output/tfidfv.pkl'
s3 = boto3.client('s3')
vectorizer_path = '/tmp/tfidf_vectorizer.pkl'
s3.download_file(S3_BUCKET_NAME, VECTORIZER_KEY, vectorizer_path)
# Cargar el TfidfVectorizer guardado durante el preprocesamiento
#tfidf_path = 'tfidfv.pkl'
tfidf_vectorizer = joblib.load(vectorizer_path)
print(f"LOADED: {vectorizer_path}")



# Función para preprocesar el texto
def preprocess_text(text: str):
    # Lowercase
    text = text.lower()

    # Remove html 
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove rare characters
    #text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Tokenize text to words
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text



# Definir el endpoint para predecir la sentencia
# Definir el endpoint de predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data['review']
        print(f"review: {review}")
        # Preprocesar el texto
        processed_text = preprocess_text(review)
        print(f"processed_text: {processed_text}")
        
        # Transformar el texto con TfidfVectorizer
        features = tfidf_vectorizer.transform([processed_text])
        print(f"features: {features.shape}")
        
        # Realizar la predicción
        prediction = model.predict(features)
        print(prediction)
        #prediction = int(prediction[0])
        sentiment = 'POSITIVE' if prediction[0] > 0.5 else 'NEGATIVE'
        return jsonify({'sentiment': sentiment, 'prediction': str(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Iniciar la aplicación (si se ejecuta este archivo directamente)
if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)
     app.run(debug=True, port=8000)
