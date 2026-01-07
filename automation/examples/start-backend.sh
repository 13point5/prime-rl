#!/bin/bash
# Script to start the metrics backend

set -e

cd "$(dirname "$0")/../backend"

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo "Starting metrics backend on http://localhost:8080..."
python -m backend.main
