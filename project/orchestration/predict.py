from flask import Flask, request, jsonify
import xgboost as xgb
import joblib
import numpy as np
import re
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords


# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo y el vectorizador TF-IDF
EXPERIMENT_NAME = "sentana-xgboost-best-models"
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

tfidf_path = '/workspaces/mlops-zoomcamp/project/output/tfidfv.pkl'
# Cargar el TfidfVectorizer guardado durante el preprocesamiento
tfidf_vectorizer = joblib.load(tfidf_path)

# Descargar stopwords si es necesario
import nltk
nltk.download('stopwords')

# Función para preprocesar el texto
def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'[^a-z\s]', '', text)  # Eliminar caracteres extraños
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Eliminar stop words
    return text


# Recuperar el mejor modelo del experimento "setana-xgboost-best-models"
def load_best_model(experiment_name: str):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise Exception("No se encontró el experimento 'setana-xgboost-best-models'.")

    # Recuperar la mejor ejecución basada en la métrica de accuracy en el test set
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              run_view_type=ViewType.ACTIVE_ONLY,
                              order_by=["metrics.accuracy DESC"],
                              max_results=1)
    if not runs:
        raise Exception("No se encontró ninguna ejecución en el experimento.")

    best_run_id = runs[0].info.run_id
    model_uri = f"runs:/{best_run_id}/model"
    return mlflow.pyfunc.load_model(model_uri)

# Cargar el modelo al iniciar el servidor
model = load_best_model(EXPERIMENT_NAME)

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
