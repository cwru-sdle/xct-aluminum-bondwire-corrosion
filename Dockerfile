FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt