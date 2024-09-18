# Usa una imagen base con Python
FROM python:3.10.13-slim

# Establece el directorio de trabajo en la imagen Docker
WORKDIR /app

# Copia los archivos necesarios al contenedor
#COPY Pipfile Pipfile.lock ./
COPY requirements.txt ./
#COPY predict.py ./

# Instala las dependencias utilizando Pipenv
#RUN pip install pipenv && pipenv install --system --deploy
RUN pip3 install -r requirements.txt

COPY predict.py ./
COPY tfidfv.pkl ./

# Expone el puerto en el que correrá Flask
EXPOSE 8000

# Define la variable de entorno para Flask
ENV FLASK_APP=predict.py

# Comando para correr la aplicación Flask
CMD ["flask", "run", "--host=0.0.0.0"]