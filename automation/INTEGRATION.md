# Integration Architecture

This document explains how the CLI, Backend, and Prime-RL are integrated.

## Integration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI Tool                                     │
│  (automation/cli/main.py)                                           │
│                                                                       │
│  1. Creates GPU instance via Prime Intellect API                    │
│  2. SSHs into instance                                              │
│  3. Uploads training config TOML (with [metrics_backend] section)   │
│  4. Sets environment variables:                                      │
│     - export RUN_ID="run-12345"                                     │
│     - export METRICS_BACKEND_URL="http://backend:8080" (optional)   │
│  5. Runs: uv run rl --trainer @ config.toml ...                    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Prime-RL Training                              │
│  (src/prime_rl/trainer/rl/train.py)                                │
│                                                                       │
│  1. Reads config.toml including [metrics_backend] section           │
│  2. Initializes MetricsBackendMonitor:                              │
│     - Reads RUN_ID from environment                                 │
│     - Connects to backend_url from config                           │
│     - Creates run in backend                                        │
│  3. During training loop:                                            │
│     - monitor.log(metrics, step=X) called automatically             │
│     - Sends POST /api/v1/metrics to backend                         │
│     - Sends distributions, samples (if enabled)                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Metrics Backend API                              │
│  (automation/backend/main.py)                                       │
│                                                                       │
│  1. Receives metrics via POST /api/v1/metrics                       │
│  2. Stores in database (SQLite/PostgreSQL)                          │
│  3. Serves data via GET /api/v1/runs/{id}/metrics                  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Web UI                                       │
│  (automation/web/)                                                   │
│                                                                       │
│  1. Polls backend for runs and metrics                              │
│  2. Displays charts and progress                                    │
│  3. Auto-refreshes for near real-time updates                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Integration Points in Detail

### 1. Configuration Integration (TOML)

Prime-RL reads the `[metrics_backend]` section from your training config:

```toml
# Any training config file (trainer.toml, orchestrator.toml, etc.)

[metrics_backend]
backend_url = "http://localhost:8080"
api_key_var = "METRICS_API_KEY"  # Optional
timeout = 30
log_samples = false
log_distributions = true
```

**Code Location:**
- Config definition: `src/prime_rl/trainer/rl/config.py:156-157`
- Also in orchestrator: `src/prime_rl/orchestrator/config.py:592-593`

### 2. Monitor Initialization

The trainer and orchestrator initialize the MetricsBackendMonitor during setup:

```python
# src/prime_rl/trainer/rl/train.py:77-84
monitor = setup_monitor(
    wandb_config=config.wandb,
    metrics_backend_config=config.metrics_backend,  # ← Integration point
    output_dir=config.output_dir,
    run_config=config,
)
```

**Code Location:**
- Trainer: `src/prime_rl/trainer/rl/train.py:77-84`
- Orchestrator: `src/prime_rl/orchestrator/orchestrator.py:116-125`

### 3. Automatic Metric Logging

Prime-RL automatically calls `monitor.log()` during training, which sends data to backend:

```python
# src/prime_rl/utils/monitor/metrics_backend.py:128-145
def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics."""
    # Automatically called by prime-rl during training
    payload = {
        "run_id": self.run_id,  # From RUN_ID env var
        "step": step or metrics.get("step", 0),
        "metrics": metrics,
    }

    response = self.client.post(
        f"{self.backend_url}/api/v1/metrics",
        json=payload,
    )
```

**Code Location:** `src/prime_rl/utils/monitor/metrics_backend.py:128-145`

### 4. CLI Orchestration

The CLI ties everything together by:

1. **Uploading Config**:
   ```python
   # automation/cli/ssh_orchestrator.py:107-130
   orchestrator.upload_config(local_config_path, remote_path)
   ```

2. **Setting Environment Variables**:
   ```python
   # automation/cli/ssh_orchestrator.py:173-179
   env_vars = [
       f"export RUN_ID={run_id}",
   ]
   if metrics_backend_url:
       env_vars.append(f"export METRICS_BACKEND_URL={metrics_backend_url}")
   ```

3. **Running Prime-RL**:
   ```python
   # automation/cli/ssh_orchestrator.py:182
   train_cmd = f"cd prime-rl && uv run rl --trainer @ {config_path} ..."
   ```

**Code Location:** `automation/cli/ssh_orchestrator.py:143-195`

## Environment Variables

The integration uses these environment variables:

| Variable | Set By | Used By | Purpose |
|----------|--------|---------|---------|
| `RUN_ID` | CLI or manual | MetricsBackendMonitor | Unique identifier for the training run |
| `METRICS_BACKEND_URL` | CLI or manual | Not used yet* | Could override backend_url from config |
| `METRICS_API_KEY` | Manual | MetricsBackendMonitor | Optional API key for securing backend |

*Note: `METRICS_BACKEND_URL` is set by CLI but not currently used. You could extend MetricsBackendMonitor to check this env var and override the config value.

## Data Flow Example

Here's what happens when you run training:

1. **Trainer starts** → Reads `[metrics_backend]` from TOML
2. **Monitor initialized** → Reads `RUN_ID` from environment
3. **Monitor creates run** → `POST /api/v1/runs` with run_id, env_id, config
4. **Training step 1** → `monitor.log({"loss/mean": 0.5, ...}, step=1)`
5. **Metrics sent** → `POST /api/v1/metrics` with run_id, step, metrics
6. **Backend stores** → Saves to database
7. **Web UI polls** → `GET /api/v1/runs/{run_id}/metrics`
8. **Charts update** → User sees real-time metrics

## Files Modified for Integration

1. **Monitor System** (`src/prime_rl/utils/monitor/`)
   - `__init__.py`: Added MetricsBackendMonitor import and setup
   - `metrics_backend.py`: NEW - Monitor implementation

2. **Trainer** (`src/prime_rl/trainer/rl/`)
   - `config.py`: Added metrics_backend config field
   - `train.py`: Pass metrics_backend_config to setup_monitor()

3. **Orchestrator** (`src/prime_rl/orchestrator/`)
   - `config.py`: Added metrics_backend config field
   - `orchestrator.py`: Pass metrics_backend_config to setup_monitor()

## Testing the Integration

### Without CLI (Manual):

```bash
# 1. Start backend
cd automation/backend
python -m backend.main &

# 2. Create config with metrics_backend section
cat > test-config.toml << 'EOF'
max_steps = 10
[model]
name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
[metrics_backend]
backend_url = "http://localhost:8080"
[[orchestrator.env]]
id = "reverse-text"
EOF

# 3. Run training with RUN_ID
export RUN_ID="test-run-$(date +%s)"
uv run rl --trainer @ test-config.toml --orchestrator @ test-config.toml --inference @ test-config.toml

# 4. Check metrics
curl http://localhost:8080/api/v1/runs/$RUN_ID/metrics
```

### With CLI (Full Automation):

```bash
# 1. Start backend
cd automation/backend && python -m backend.main &

# 2. Use CLI to orchestrate everything
python -m automation.cli.main \
  --gpu-config automation/cli/gpu_config.example.yaml \
  --training-config automation/examples/reverse-text-with-metrics.toml \
  --env-id reverse-text \
  --metrics-backend-url http://localhost:8080 \
  --mock  # Use mock mode for testing without real GPU instances
```

## Summary

The integration is **automatic and transparent**:

1. ✅ Add `[metrics_backend]` to your TOML config
2. ✅ Set `RUN_ID` environment variable
3. ✅ Run training normally with `uv run rl ...`
4. ✅ Metrics automatically flow to backend
5. ✅ View in web UI at http://localhost:3000

No code changes needed in your training scripts - just configuration!
