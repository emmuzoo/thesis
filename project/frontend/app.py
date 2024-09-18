# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import joblib
import os

# Cargar el fichero CSV
df = pd.read_parquet('raw_ref_data_v3.parquet')
#data_path = './'
#df = joblib.load(os.path.join(data_path, "ref.pkl"))
COLUMNS = ['SCHEDULED_DEPARTURE', #'MONTH', 'DAY', 'DAY_OF_WEEK', 'HOUR', 
           'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
           'SCHEDULED_ARRIVAL', 'DISTANCE', 
           #'DELAY'
           ]
df = df[COLUMNS]
df = df.head(500)

st.title("Flight Delay Prediction")

# Mostrar la tabla de vuelos
st.write("Listado de vuelos:")

# Usando st.dataframe para mostrar la tabla completa
st.dataframe(df)  # Puedes usar st.table(df) si prefieres una tabla estática sin la opción de desplazamiento

# Añadir botones para cada fila
for i, row in df.iterrows():
    # Botón para enviar al backend
    if st.button(f"Predecir Retraso para vuelo {i+1}", key=i):
        # Construir los datos a enviar
        data = {
            "airline": row['AIRLINE'],
            "origin_airport": row['ORIGIN_AIRPORT'],
            "destination_airport": row['DESTINATION_AIRPORT'],
            "scheduled_time": row['SCHEDULED_TIME'],
            "scheduled_arrival": row['SCHEDULED_ARRIVAL']
        }
        response = requests.post("http://localhost:5000/predict_delay", json=data)
        
        if response.status_code == 200:
            prediction = response.json().get('delay')
            if prediction:
                st.success(f"El vuelo {i+1} probablemente tenga retraso.")
            else:
                st.success(f"El vuelo {i+1} probablemente no tenga retraso.")
        else:
            st.error("Error al predecir el retraso.")