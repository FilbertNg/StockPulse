#!/bin/bash

# Path to the preloaded model directory
MODEL_DIR=${MODEL_DIR:-"/opt/models/finbert_individual2_sentiment_model"}

# Check if the model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory not found at $MODEL_DIR."
    echo "Please ensure the model directory is available before starting the application."
    exit 1
fi

# Ensure the application model path is set up
APP_MODEL_PATH="/app/model"
echo "Setting up model directory for the application..."
mkdir -p "$APP_MODEL_PATH"
cp -r "$MODEL_DIR"/* "$APP_MODEL_PATH"

# Start the application
echo "Starting the application..."
gunicorn -w 4 -b 0.0.0.0:5001 functions:app
