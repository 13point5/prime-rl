# Getting Started with Prime-RL GPU Automation

This guide will walk you through setting up and using the GPU automation and metrics visualization system for Prime-RL.

## Overview

The automation system consists of three main components:

1. **CLI Tool** - Automates GPU instance creation and training orchestration
2. **Metrics Backend** - Collects and stores training metrics
3. **Web UI** - Visualizes training metrics in real-time

## Quick Start

### 1. Set Up the Metrics Backend

First, start the metrics backend server:

```bash
cd automation/backend
pip install -r requirements.txt
python -m backend.main
```

The backend will start on `http://localhost:8080` by default.

### 2. Set Up the Web UI

In a new terminal, start the web dashboard:

```bash
cd automation/web
npm install
npm run dev
```

The web UI will be available at `http://localhost:3000`.

### 3. Run Training with Metrics Tracking

Update your training configuration to enable metrics tracking. Create or edit your TOML config:

```toml
# configs/my-training.toml

max_steps = 1000

[model]
name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

[metrics_backend]
backend_url = "http://localhost:8080"
log_samples = false
log_distributions = true

[[orchestrator.env]]
id = "reverse-text"
```

Run training:

```bash
export RUN_ID="my-training-run-$(date +%s)"
uv run rl \
  --trainer @ configs/my-training.toml \
  --orchestrator @ configs/my-training.toml \
  --inference @ configs/my-training.toml
```

### 4. View Metrics in Web UI

Open `http://localhost:3000` in your browser. You should see your training run appear in the list. Click on it to view real-time metrics!

## Using the CLI Tool for GPU Automation

### Setup

Install CLI dependencies:

```bash
cd automation/cli
pip install -r requirements.txt
```

### Configure GPU Settings

Create a GPU configuration file `gpu_config.yaml`:

```yaml
gpu_type: "H100"
gpu_count: 2
region: "us-east"
image: "ubuntu-22.04-cuda-12.1"
disk_size_gb: 200
```

### Set Environment Variables

```bash
export PRIME_API_KEY="your-prime-intellect-api-key"
export METRICS_API_KEY="your-metrics-backend-api-key"  # Optional
```

### Launch Training on Remote GPUs

```bash
python -m automation.cli.main \
  --gpu-config gpu_config.yaml \
  --training-config configs/reverse-text.toml \
  --env-id reverse-text \
  --metrics-backend-url http://your-backend-url:8080 \
  --monitor
```

This will:
1. Create a Prime Intellect GPU instance
2. SSH into the instance
3. Set up the environment (clone repo, install dependencies)
4. Upload your training config
5. Start training
6. Monitor progress (optional)
7. Clean up the instance when done

### CLI Options

**Instance Management:**
- `--gpu-config` - Path to GPU configuration YAML file
- `--instance-name` - Custom name for the instance
- `--instance-id` - Use existing instance instead of creating new one

**Training Configuration:**
- `--training-config` - Path to training configuration TOML file
- `--env-id` - Environment ID to train on
- `--run-id` - Custom run ID (auto-generated if not provided)
- `--metrics-backend-url` - URL of metrics backend service

**Execution Options:**
- `--monitor` - Monitor training progress in terminal
- `--no-cleanup` - Don't terminate instance after training
- `--download-artifacts` - Download training artifacts before cleanup
- `--artifacts-path` - Local path to save artifacts (default: `./artifacts`)

**Instance Queries:**
- `--list-instances` - List all instances and exit
- `--terminate-instance INSTANCE_ID` - Terminate specific instance

**Testing:**
- `--mock` - Use mock API client for testing without actual GPU instances
- `--local` - Run training locally without creating instances

## Example Workflows

### 1. Simple Training Run

```bash
python -m automation.cli.main \
  --gpu-config gpu_config.yaml \
  --training-config configs/reverse-text.toml \
  --env-id reverse-text
```

### 2. Training with Monitoring

```bash
python -m automation.cli.main \
  --gpu-config gpu_config.yaml \
  --training-config configs/gsm8k.toml \
  --env-id gsm8k \
  --metrics-backend-url http://localhost:8080 \
  --monitor
```

### 3. Resume Training on Existing Instance

```bash
python -m automation.cli.main \
  --instance-id existing-instance-id \
  --training-config configs/my-training.toml \
  --env-id my-env
```

### 4. List Running Instances

```bash
python -m automation.cli.main --list-instances
```

### 5. Clean Up Instance

```bash
python -m automation.cli.main --terminate-instance INSTANCE_ID
```

## Metrics Backend API

The metrics backend provides a REST API for submitting and querying metrics.

### Create Run

```bash
curl -X POST http://localhost:8080/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "my-run-123",
    "env_id": "reverse-text",
    "name": "Test Run"
  }'
```

### Submit Metrics

```bash
curl -X POST http://localhost:8080/api/v1/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "my-run-123",
    "step": 10,
    "metrics": {
      "loss/mean": 0.5,
      "reward/mean": 0.8
    }
  }'
```

### Get Metrics

```bash
curl http://localhost:8080/api/v1/runs/my-run-123/metrics
```

## Configuration Reference

### Prime-RL Training Config

Add metrics backend configuration to your TOML file:

```toml
[metrics_backend]
backend_url = "http://localhost:8080"
api_key_var = "METRICS_API_KEY"  # Optional
timeout = 30
log_samples = false
log_distributions = true
```

### Backend Configuration

Create `.env` file in `automation/backend/`:

```env
DATABASE_URL=sqlite:///metrics.db
BACKEND_API_KEY=your-secret-key  # Optional
HOST=0.0.0.0
PORT=8080
```

For production, use PostgreSQL:

```env
DATABASE_URL=postgresql://user:password@localhost/metrics
```

## Troubleshooting

### Backend not receiving metrics

1. Check that the backend is running: `curl http://localhost:8080/`
2. Verify `RUN_ID` environment variable is set
3. Check backend logs for errors

### Web UI not showing runs

1. Verify backend is accessible from browser
2. Check browser console for API errors
3. Ensure backend URL is correct in web UI

### CLI cannot create instances

1. Verify `PRIME_API_KEY` is set correctly
2. Check Prime Intellect API documentation for endpoint changes
3. Use `--mock` flag to test without actual API calls

### SSH connection fails

1. Ensure instance has completed startup
2. Check firewall rules allow SSH (port 22)
3. Verify SSH key has correct permissions: `chmod 600 ~/.ssh/id_rsa`

## Advanced Usage

### Using PostgreSQL/TimescaleDB

For production deployments, use PostgreSQL or TimescaleDB:

```bash
# Install PostgreSQL
sudo apt-get install postgresql

# Create database
sudo -u postgres createdb metrics

# Update DATABASE_URL
export DATABASE_URL="postgresql://postgres:password@localhost/metrics"

# Start backend
python -m backend.main
```

### Securing the Backend

Set an API key for the backend:

```env
BACKEND_API_KEY=your-secret-key
```

Update training config:

```toml
[metrics_backend]
backend_url = "https://your-backend.com"
api_key_var = "METRICS_API_KEY"
```

Set environment variable:

```bash
export METRICS_API_KEY="your-secret-key"
```

### Deploying the Web UI

Build for production:

```bash
cd automation/web
npm run build
```

Serve with nginx or any static file server:

```bash
npx serve -s dist -p 3000
```

## Next Steps

- Explore different environments from the [Environments Hub](https://app.primeintellect.ai/dashboard/environments)
- Customize the web UI to show additional metrics
- Set up PostgreSQL for production deployments
- Configure automatic backup of training artifacts
- Integrate with your CI/CD pipeline

## Support

- [Prime-RL Documentation](https://github.com/PrimeIntellect-ai/prime-rl)
- [Prime Intellect Platform](https://app.primeintellect.ai)
- [Report Issues](https://github.com/PrimeIntellect-ai/prime-rl/issues)
