#!/bin/bash
# Example script to run training locally with metrics tracking

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Prime-RL Training with Metrics Tracking${NC}"

# Check if backend is running
if ! curl -s http://localhost:8080/ > /dev/null; then
    echo "Error: Metrics backend is not running at http://localhost:8080"
    echo "Please start the backend first:"
    echo "  cd automation/backend && python -m backend.main"
    exit 1
fi

# Generate unique run ID
RUN_ID="reverse-text-$(date +%s)"
echo -e "${GREEN}Run ID: ${RUN_ID}${NC}"

# Export environment variables
export RUN_ID

# Run training
echo -e "${BLUE}Starting training...${NC}"
uv run rl \
  --trainer @ automation/examples/reverse-text-with-metrics.toml \
  --orchestrator @ automation/examples/reverse-text-with-metrics.toml \
  --inference @ automation/examples/reverse-text-with-metrics.toml

echo -e "${GREEN}Training complete!${NC}"
echo -e "${BLUE}View metrics at: http://localhost:3000/runs/${RUN_ID}${NC}"
