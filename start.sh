#!/bin/bash

# Variables for MinIO
MINIO_ENDPOINT=${MINIO_ENDPOINT:-"minio:9000"}
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-"minioadmin"}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-"minioadmin"}
BUCKET_NAME=${BUCKET_NAME:-"model"}
MODEL_FILENAME=${MODEL_FILENAME:-"finbert_individual2_sentiment_model.tar.gz"}

# Wait for MinIO to be available
echo "Waiting for MinIO to be available at $MINIO_ENDPOINT..."
until curl -u "$MINIO_ACCESS_KEY:$MINIO_SECRET_KEY" -f "http://$MINIO_ENDPOINT/$BUCKET_NAME/$MODEL_FILENAME"; do
    echo "MinIO is not available yet. Retrying in 5 seconds..."
    sleep 5
done

# Download and extract the model
echo "Downloading the model from MinIO..."
curl -u "$MINIO_ACCESS_KEY:$MINIO_SECRET_KEY" \
    "http://$MINIO_ENDPOINT/$BUCKET_NAME/$MODEL_FILENAME" \
    -o /app/model.tar.gz

echo "Extracting the model..."
mkdir -p /app/model
tar -xzvf /app/model.tar.gz -C /app/model
rm /app/model.tar.gz

# Start the application
echo "Starting the application..."
gunicorn -w 4 -b 0.0.0.0:5001 functions:app
