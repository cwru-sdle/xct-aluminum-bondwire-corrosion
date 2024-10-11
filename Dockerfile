FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# deploy Luigi pipeline on container launch
# EXPOSE 8082
# CMD ["python", "luigi_pipeline.py"]