
import os
import re
import numpy as np
# Store
import joblib
import pickle
# Pandas
import pandas as pd
# Mlflows
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
# nltk
import time
#import nltk
#from nltk.corpus import stopwords
# Prefecr
from prefect import flow, task 
from prefect.filesystems import S3


def categorize_departure_time(hour):
    """
    Categoriza la hora de salida en diferentes franjas horarias:
    - Madrugada (00:00 - 06:00)
    - Mañana (06:00 - 12:00)
    - Tarde (12:00 - 18:00)
    - Noche (18:00 - 24:00)
    """
    #early morning, morning, afternoon, evening
    if 0 <= hour < 6:
        return 'Early morning'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

def preprocess(data_path, output_path):
  # Load data
  start_time = time.time()
  if type(data_path) == list:
    print(f"List paquet files")
    df = pd.concat([pd.read_parquet(f) for f in data_path])
  else:
    print(f"One parquet file")
    df = pd.read_parquet(data_path)
  end_time = time.time()
  print(f"Tiempo de ejecución de 'combine_date_time': {end_time - start_time:.4f} segundos")

  start_time = time.time()

  # Impute cancelaton
  df['CANCELLATION_REASON'].fillna('Unknown', inplace=True)

  # Conversión de tipos de datos
  df['TIME'] = (df['SCHEDULED_ARRIVAL'] - df['SCHEDULED_DEPARTURE']).dt.total_seconds() / 60 /60  # En minutos
  df['ACTUAL_FLIGHT_TIME'] = (df['ARRIVAL_TIME'] - df['DEPARTURE_TIME']).dt.total_seconds() / 60 /60  # En minutos

  # Crear columna categórica para franja horaria de salida
  #df['DEPARTURE_PERIOD'] = df['SCHEDULED_DEPARTURE'].dt.hour.apply(categorize_departure_time)
  df['HOUR'] = df['SCHEDULED_DEPARTURE'].dt.hour

  end_time = time.time()
  print(f"Tiempo de ejecución de 'combine_date_time': {end_time - start_time:.4f} segundos")

  df['T_DEPARTURE'] = df['SCHEDULED_DEPARTURE'].apply(lambda x: f"{x.day_of_week:02}{x.hour:02}")
  df['T_ARRIVAL']   = df['SCHEDULED_ARRIVAL'].apply(lambda x: f"{x.day_of_week:02}{x.hour:02}")
  df['ROUTE'] = df['ORIGIN_AIRPORT'].str.cat(df['DESTINATION_AIRPORT'], sep='-')

  df['CANCELLED'] = df['CANCELLED'].astype(str)
  df['DIVERTED'] = df['DIVERTED'].astype(str)

  # Inicializar diccionario para guardar los LabelEncoders
  label_encoders = {}

  # 4. Codificación de variables categóricas con LabelEncoder
  categorical_columns = ['ROUTE', 'T_DEPARTURE', 'T_ARRIVAL', 'CANCELLED',	'DIVERTED'
                         #'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'CANCELLATION_REASON'
                         ]
  '''
  for col in categorical_columns:
    print(f"col: {col}")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    df[col] = df[col].astype('category')
    label_encoders[col] = le
  '''
  for col in categorical_columns:
    print(f"col: {col}")
    df[col] = df[col].astype('category')

  # Target
  df['ARRIVAL_DELAY'].fillna(-1, inplace=True)
  df['DELAY'] = np.where(df['ARRIVAL_DELAY'] > 0, 1, 0)
  df.drop(df[df['ARRIVAL_DELAY'] == -1].index, inplace=True)

  # Remove variables
  # Remover na
  #df.dropna(inplace = True)
  print(f"df.shape: {df.shape}")
  
  # Remove variables
  #cat_columns = ['T_ARRIVAL', 'ORIGIN_DESTINATION_AIRPORT', 'CANCELLED', 'DIVERTED']
  #num_columns = ['DEPARTURE_DELAY', 'TIME', 'DISTANCE']
  #stratify_colname = ['DELAY']
  columns = [
      'SCHEDULED_DEPARTURE', 'ROUTE',
      'T_DEPARTURE', 'T_ARRIVAL', 'DEPARTURE_DELAY', 'CANCELLED', 'DIVERTED',
      'TIME', 'DISTANCE', 'DELAY'
  ]
  df = df[columns]
  print(f"df.dtypes: {df.dtypes}")

  start_time = time.time()
  # Guardar el DataFrame preprocesado en formato Parquet
  print(f"columns: {df.columns.values}")
  parquet_file_path = os.path.join(output_path, "processed_data.parquet")
  df.to_parquet(parquet_file_path, index=False, engine='pyarrow')

  # Guardar el diccionario de LabelEncoders usando pickle
  label_encoders_file_path = os.path.join(output_path, "label_encoders.pkl")
  joblib.dump(label_encoders, label_encoders_file_path)

  end_time = time.time()
  print(f"Tiempo de ejecución de 'combine_date_time': {end_time - start_time:.4f} segundos")

  print(f"Datos procesados guardados en: {parquet_file_path}")
  print(f"Diccionario de LabelEncoders guardado en: {label_encoders_file_path}")

  #return df, label_encoders


@task(
    name="preprocess data", 
    tags=["data"], 
    #retries=3, 
    #retry_delay_seconds=60
)
def run_data_prep(raw_data_path: str, dest_path: str, test_size: float):
    # Fit the DictVectorizer and preprocess data
    preprocess(raw_data_path, dest_path)

    # Split
    parquet_file_path = os.path.join(dest_path, "processed_data.parquet")
    print(f"parquet_file_path: {parquet_file_path}")
    data = pd.read_parquet(parquet_file_path)
    print(f"df.shape: {data.shape}, columns: {data.columns.values}")
    random_state = 42
    stratify_colname = 'DELAY'
    X = data # Contains all columns.
    y = data[[stratify_colname]]

    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=(1.0 - test_size),
                                                        random_state=random_state)
    #val_size_adjusted = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                    y_temp,
                                                    stratify=y_temp,
                                                    test_size=0.5,
                                                    random_state=random_state)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save data
    X_train.to_parquet(os.path.join(dest_path, "train.parquet"), index=False)
    X_val.to_parquet(os.path.join(dest_path, "val.parquet"), index=False)
    X_test.to_parquet(os.path.join(dest_path, "test.parquet"), index=False)