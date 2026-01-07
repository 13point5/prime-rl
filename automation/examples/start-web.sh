#!/bin/bash
# Script to start the web UI

set -e

cd "$(dirname "$0")/../web"

if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo "Starting web UI on http://localhost:3000..."
npm run dev
