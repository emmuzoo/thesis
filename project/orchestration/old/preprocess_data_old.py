import os
import re
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
#import nltk
#from nltk.corpus import stopwords
# Prefecr
from prefect import flow, task 
from prefect.filesystems import S3

# Descargar stopwords si es necesario
#nltk.download('stopwords')

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


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


# Función para preprocesar el dataset
def preprocess(df: pd.DataFrame, tfidf: TfidfVectorizer):
    #df = pd.read_csv(data_path)
    df = df[['review', 'sentiment']]
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
    df['review'] = df['review'].apply(preprocess_text)

    X = df['review']
    y = df['sentiment']

    X_tfidf = tfidf.fit_transform(X)

    return X_tfidf, y


@task(
    name="preprocess data", 
    tags=["data"], 
    retries=3, 
    retry_delay_seconds=60
)
def run_data_prep(raw_data_path: str, dest_path: str, max_features, test_size: float, val_size: float):
    # Download stopwords
    nltk.download('stopwords')
    nltk.download('punkt')

    # Load parquet files
    df_train = pd.read_csv(
        raw_data_path
    )

    # Fit the DictVectorizer and preprocess data
    max_features = 500
    tfidfv = TfidfVectorizer(max_features=max_features)
    X, y = preprocess(df_train, tfidfv)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)
    
    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    #dump_pickle(tfidfv, os.path.join(dest_path, "tfidfv.pkl"))
    joblib.dump(tfidfv, os.path.join(dest_path, "tfidfv.pkl"))
    #dump_pickle((X, y), os.path.join(dest_path, "clean_data.pkl"))
    #dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    #dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    #dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
    joblib.dump((X, y), os.path.join(dest_path, "clean_data.pkl"))
    joblib.dump((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    joblib.dump((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    joblib.dump((X_test, y_test), os.path.join(dest_path, "test.pkl"))

if __name__ == '__main__':
    run_data_prep()
