#!/bin/bash

# CVAT Queue Manager startup script
# Usage: ./run.sh [dev|prod]

MODE="${1:-dev}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create data directory if it doesn't exist
mkdir -p data backups

echo ""
echo "========================================"
echo "CVAT Queue Manager"
echo "Mode: $MODE"
echo "========================================"

if [ "$MODE" = "prod" ] || [ "$MODE" = "production" ]; then
    echo "Starting in PRODUCTION mode with Gunicorn..."
    gunicorn main:app -c gunicorn.conf.py
else
    echo "Starting in DEVELOPMENT mode with Uvicorn..."
    echo "(Hot reload enabled)"
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
fi
