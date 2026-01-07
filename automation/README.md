# Prime-RL GPU Automation & Metrics Dashboard

This directory contains tools for automating GPU instance management and visualizing training metrics.

## Components

### 1. CLI Tool (`cli/`)
Automates the process of spinning up Prime Intellect GPUs, running training, and monitoring progress.

**Features:**
- Create and manage Prime Intellect GPU instances via API
- SSH into instances and set up environment
- Launch training runs with custom configurations
- Monitor training progress
- Clean up instances when done

**Usage:**
```bash
python -m automation.cli.main \
  --gpu-config gpu_config.yaml \
  --training-config configs/reverse-text.toml \
  --env-id reverse-text
```

### 2. Metrics Backend (`backend/`)
FastAPI service that receives and stores training metrics for visualization.

**Features:**
- REST API for receiving metrics from training runs
- TimescaleDB storage for time-series data
- WebSocket support for real-time updates
- Query API for web UI

**Endpoints:**
- `POST /api/v1/runs` - Create new run
- `POST /api/v1/metrics` - Submit metrics
- `GET /api/v1/runs/{run_id}` - Get run details
- `GET /api/v1/runs/{run_id}/metrics` - Get run metrics
- `WS /ws/runs/{run_id}` - Real-time metric updates

### 3. Web UI (`web/`)
React-based dashboard for visualizing training metrics (W&B-style).

**Features:**
- Real-time training metric visualization
- Multi-run comparison
- Reward distribution histograms
- Training progress tracking
- Log viewing

## Setup

### Backend Setup
```bash
cd automation/backend
pip install -r requirements.txt
python -m backend.main
```

### Web UI Setup
```bash
cd automation/web
npm install
npm run dev
```

### CLI Setup
```bash
cd automation/cli
pip install -r requirements.txt
```

## Architecture

```
┌─────────────┐
│  Prime RL   │  (Training)
│  Instance   │
└──────┬──────┘
       │ Metrics
       ▼
┌─────────────┐     WebSocket     ┌─────────────┐
│   Backend   │ ◄────────────────►│   Web UI    │
│  (FastAPI)  │                   │   (React)   │
└──────┬──────┘                   └─────────────┘
       │
       ▼
┌─────────────┐
│ TimescaleDB │
└─────────────┘

┌─────────────┐
│     CLI     │  (Orchestration)
│    Tool     │
└──────┬──────┘
       │ SSH/API
       ▼
┌─────────────┐
│Prime Intell.│
│ GPU Instance│
└─────────────┘
```

## Environment Variables

### CLI
- `PRIME_API_KEY` - Prime Intellect API key
- `PRIME_API_URL` - Prime Intellect API base URL (default: https://api.primeintellect.ai)

### Backend
- `DATABASE_URL` - PostgreSQL/TimescaleDB connection string
- `BACKEND_API_KEY` - API key for securing the backend

### Prime-RL Training
- `RUN_ID` - Unique run identifier
- `METRICS_BACKEND_URL` - URL of the metrics backend
- `METRICS_API_KEY` - API key for metrics backend
