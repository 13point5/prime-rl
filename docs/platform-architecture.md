# Prime Intellect Platform Architecture

A guide to understanding the tech stack behind Prime Intellect's hosted RL training platform.

## Table of Contents

- [Overview](#overview)
- [The Three Core Components](#the-three-core-components)
- [Infrastructure Layer](#infrastructure-layer)
- [Communication & Data Flow](#communication--data-flow)
- [Monitoring & Observability](#monitoring--observability)
- [How It All Fits Together](#how-it-all-fits-together)

---

## Overview

Prime Intellect's platform lets users train RL models at scale without managing infrastructure. You select an environment, configure training, and the platform handles GPU provisioning, scaling, and monitoring.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PRIME INTELLECT PLATFORM                         │
│                                                                      │
│   User: "Train on math environment with 8 GPUs"                     │
│                           ↓                                          │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Web UI / API                              │   │
│   │         (app.primeintellect.ai)                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                           ↓                                          │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              Kubernetes Cluster                              │   │
│   │    ┌───────────┐  ┌───────────┐  ┌───────────────────┐     │   │
│   │    │Orchestrator│  │ Inference │  │     Trainer       │     │   │
│   │    │  (CPU)    │  │  (GPUs)   │  │    (GPUs)         │     │   │
│   │    └───────────┘  └───────────┘  └───────────────────┘     │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                           ↓                                          │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              W&B-style Dashboard                             │   │
│   │      (metrics, samples, distributions, logs)                │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Three Core Components

The training system is split into three independently scalable components.

### 1. Orchestrator

**What it is:** A CPU-based coordinator that manages the training workflow.

**What it does:**
- Loads environments from the [Environment Hub](https://app.primeintellect.ai/dashboard/environments)
- Spawns subprocess workers to generate rollouts
- Calls inference servers to get model completions
- Verifies outputs and computes rewards using the `verifiers` library
- Packages rollouts into training batches
- Sends batches to the trainer
- Triggers weight updates on inference servers when trainer produces new weights

**Tech used:**
- Python async/await for concurrent I/O
- `verifiers` library for environment evaluation
- Subprocess isolation (each env worker runs in its own process)
- ZMQ or filesystem for sending batches

**Key files:**
- `src/prime_rl/orchestrator/orchestrator.py`
- `src/prime_rl/orchestrator/scheduler.py`
- `src/prime_rl/orchestrator/env_worker.py`

---

### 2. Inference

**What it is:** GPU servers running vLLM that generate text completions.

**What it does:**
- Serves an OpenAI-compatible API (`/v1/chat/completions`)
- Generates rollouts when the orchestrator requests completions
- Loads updated model weights when the trainer produces them
- Supports LoRA adapters for multi-run training

**Tech used:**
- **vLLM** - High-performance LLM inference engine with PagedAttention
- OpenAI-compatible REST API
- Custom endpoints for weight management:
  - `POST /update_weights` - Load new weights from path
  - `POST /load_lora_adapter` - Load a LoRA adapter
  - `POST /reload_weights` - Reset to base model

**Why vLLM?**
- Optimized for throughput with continuous batching
- PagedAttention reduces memory waste
- Native tensor parallelism for multi-GPU inference
- OpenAI API compatibility makes it easy to integrate

**Key files:**
- `src/prime_rl/inference/vllm/server.py`
- `src/prime_rl/inference/vllm/serving_chat_with_tokens.py`

---

### 3. Trainer

**What it is:** Distributed PyTorch training across multiple GPUs/nodes.

**What it does:**
- Receives training batches from orchestrator
- Runs forward/backward passes with gradient accumulation
- Updates model weights using various RL objectives (GRPO, RLOO, etc.)
- Saves checkpoints to shared storage
- Broadcasts new weights for inference to pick up

**Tech used:**
- **PyTorch** - Deep learning framework
- **FSDP2** (Fully Sharded Data Parallel) - Distributes model across GPUs
- **torchrun** - Launches distributed training
- **LoRA** - Low-rank adaptation for efficient fine-tuning

**Parallelism strategies:**

| Strategy | What it does |
|----------|--------------|
| Data Parallel (DP) | Same model on each GPU, different data |
| Tensor Parallel (TP) | Split layers across GPUs |
| Pipeline Parallel (PP) | Split model stages across GPUs |
| Context Parallel (CP) | Split long sequences across GPUs |
| Expert Parallel (EP) | Split MoE experts across GPUs |

**Key files:**
- `src/prime_rl/trainer/rl/train.py`
- `src/prime_rl/trainer/model.py`
- `src/prime_rl/trainer/optim.py`

---

## Infrastructure Layer

### Kubernetes (K8s)

**What it is:** Container orchestration platform that manages where and how containers run.

**What it does for Prime RL:**
- Schedules pods onto nodes with available GPUs
- Restarts crashed containers automatically
- Provides stable DNS names for distributed training
- Manages shared storage across all components

**Key concepts:**

| Term | Description |
|------|-------------|
| **Pod** | Smallest deployable unit - one or more containers |
| **Node** | A physical or virtual machine in the cluster |
| **StatefulSet** | Manages pods with stable names (trainer-0, trainer-1, etc.) |
| **Service** | Stable network endpoint to reach pods |
| **PVC** | Persistent Volume Claim - storage that survives restarts |

**Why StatefulSets?**

Distributed training requires:
- Stable hostnames (PyTorch needs to know peer addresses)
- Predictable pod names for rank assignment
- Ordered startup/shutdown

Regular Deployments give random names like `trainer-7f8b9c-x2k4p` which breaks distributed training coordination.

---

### Helm

**What it is:** Package manager for Kubernetes.

**What it does:**
- Bundles all Kubernetes YAML files into a single deployable "chart"
- Allows configuration via values (replicas, GPU count, etc.)
- Enables one-command deployment and upgrades

**Example:**
```bash
# Deploy with custom settings
helm install my-experiment ./k8s/prime-rl \
  --set trainer.replicas=8 \
  --set inference.replicas=4 \
  --set trainer.gpu.count=8
```

**Chart structure:**
```
k8s/prime-rl/
├── Chart.yaml          # Chart metadata (name, version)
├── values.yaml         # Default configuration
├── templates/
│   ├── orchestrator.yaml
│   ├── trainer.yaml
│   ├── inference.yaml
│   ├── pvc.yaml        # Shared storage
│   └── services.yaml   # Network configuration
└── examples/
    └── reverse-text.yaml
```

---

### Shared Storage (NFS/Ceph)

**What it is:** Network filesystem accessible by all pods.

**What it does:**
- Stores model checkpoints
- Holds training rollouts and batches
- Shares updated weights between trainer and inference
- Persists data across pod restarts

**Mount structure:**
```
/data/
├── outputs/
│   ├── run_abc123/
│   │   ├── checkpoints/
│   │   │   └── step_1000/
│   │   ├── rollouts/
│   │   └── broadcast/      # Weights for inference
│   │       └── step_1000/
│   └── run_def456/
└── models/                  # Base model weights
```

**Why shared storage is critical:**
1. Trainer saves weights to `/data/outputs/broadcast/step_N/`
2. Orchestrator detects new weights
3. Orchestrator tells inference servers to load from that path
4. All components read/write the same filesystem

---

## Communication & Data Flow

### Transport Options

The system supports two transport backends for sending training batches:

#### 1. Filesystem Transport (Default)

```
Orchestrator                          Trainer
     │                                    │
     │  Write batch to /data/rollouts/    │
     │ ─────────────────────────────────► │
     │                                    │  Poll for new files
     │                                    │  Read and process
```

- Simple and reliable
- Works with any shared filesystem
- Higher latency due to filesystem polling

#### 2. ZMQ Transport

```
Orchestrator                          Trainer
     │                                    │
     │         ZMQ PUSH socket            │
     │ ─────────────────────────────────► │  ZMQ PULL socket
     │        (direct TCP)                │
```

- Lower latency (direct socket communication)
- Better for multi-run scenarios
- Requires network connectivity between pods

---

### Elastic Inference Pool

**What it is:** DNS-based auto-discovery of inference servers.

**How it works:**
1. Orchestrator is configured with a DNS hostname (e.g., `inference.default.svc.cluster.local`)
2. DNS resolves to multiple IP addresses (one per inference pod)
3. Pool periodically re-resolves DNS to discover new/removed servers
4. Health checks verify each server is responding
5. Adapter sync ensures all servers have the latest LoRA weights

```python
# Automatic server discovery
pool = ElasticInferencePool(
    hostname="inference.default.svc.cluster.local",
    port=8000,
    sync_interval=5.0  # Re-check DNS every 5 seconds
)

# Pool automatically:
# - Discovers new inference pods
# - Removes unhealthy pods
# - Syncs LoRA adapters to all pods
```

**Key file:** `src/prime_rl/utils/elastic.py`

---

### Weight Update Flow

```
┌──────────┐                ┌─────────────┐               ┌───────────┐
│ Trainer  │                │Orchestrator │               │ Inference │
└────┬─────┘                └──────┬──────┘               └─────┬─────┘
     │                             │                             │
     │  Save weights to            │                             │
     │  /data/broadcast/step_N/    │                             │
     │ ───────────────────────────►│                             │
     │                             │                             │
     │                             │  POST /update_weights       │
     │                             │  {path: "/data/.../step_N"} │
     │                             │────────────────────────────►│
     │                             │                             │
     │                             │         200 OK              │
     │                             │◄────────────────────────────│
     │                             │                             │
     │                             │  (repeat for all servers)   │
     │                             │                             │
```

---

## Monitoring & Observability

### Weights & Biases (W&B)

**What it is:** Third-party ML experiment tracking platform.

**What it logs:**
- Training metrics (loss, throughput, learning rate)
- Sample rollouts (prompt, completion, reward)
- Distribution histograms (reward, advantage, entropy)

**Configuration:**
```bash
uv run rl ... --wandb.project my-project --wandb.name my-run
```

Creates two W&B runs:
- `my-run-trainer` - Training metrics
- `my-run-orchestrator` - Sampling and environment metrics

---

### Prime Monitor

**What it is:** Prime Intellect's own monitoring backend (powers their web UI).

**API endpoints:**
```
POST https://api.primeintellect.ai/api/internal/rft/metrics
POST https://api.primeintellect.ai/api/internal/rft/samples
POST https://api.primeintellect.ai/api/internal/rft/distributions
POST https://api.primeintellect.ai/api/internal/rft/finalize
```

**What it sends:**
- Real-time training metrics
- Sample rollouts with full trajectory data
- Reward/advantage distributions
- Final summary on completion

**Authentication:**
- Requires `PRIME_API_KEY` environment variable
- Requires `RUN_ID` to identify the experiment

**Key file:** `src/prime_rl/utils/monitor/prime.py`

---

### Prometheus Metrics Server

**What it is:** Standard metrics format for Kubernetes monitoring.

**Exposed metrics:**
```
trainer_step                          # Current training step
trainer_loss                          # Current loss value
trainer_throughput_tokens_per_sec     # Training throughput
trainer_mfu_percent                   # Model FLOPS utilization
trainer_runs_discovered               # Multi-run: runs found
trainer_runs_active                   # Multi-run: runs training
trainer_run_step{run="abc123"}        # Per-run step counter
```

**Endpoint:** `GET /metrics` on trainer pods

**Key file:** `src/prime_rl/utils/metrics_server.py`

---

## How It All Fits Together

### Complete Training Flow

```
1. USER STARTS EXPERIMENT
   └─► Platform receives config (model, environment, hyperparameters)

2. KUBERNETES DEPLOYMENT
   └─► Helm deploys orchestrator, trainer, inference pods
   └─► Shared PVC mounted at /data on all pods
   └─► DNS names registered for service discovery

3. INITIALIZATION
   └─► Inference loads base model into GPU memory
   └─► Trainer initializes FSDP2 distributed training
   └─► Orchestrator discovers inference servers via DNS
   └─► Environment workers spawned as subprocesses

4. TRAINING LOOP
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   │  Orchestrator                                                │
   │       │                                                      │
   │       ├─► Sample prompts from environment                    │
   │       ├─► Send to inference for completion                   │
   │       ├─► Verify outputs, compute rewards                    │
   │       ├─► Package into training batch                        │
   │       └─► Send batch to trainer (ZMQ/filesystem)             │
   │                                                              │
   │  Trainer                                                     │
   │       │                                                      │
   │       ├─► Receive batch                                      │
   │       ├─► Forward pass (distributed across GPUs)             │
   │       ├─► Compute RL loss (GRPO/RLOO/etc.)                   │
   │       ├─► Backward pass + optimizer step                     │
   │       └─► Save weights to /data/broadcast/step_N/            │
   │                                                              │
   │  Weight Sync                                                 │
   │       │                                                      │
   │       ├─► Orchestrator detects new weights                   │
   │       └─► Triggers /update_weights on all inference servers  │
   │                                                              │
   │  (repeat)                                                    │
   └──────────────────────────────────────────────────────────────┘

5. MONITORING
   └─► Metrics sent to Prime Monitor API
   └─► Optionally logged to W&B
   └─► Prometheus metrics exposed for cluster monitoring

6. COMPLETION
   └─► Final checkpoint saved
   └─► Summary uploaded to Prime Monitor
   └─► Resources released
```

---

### Multi-Run Architecture

The platform can run multiple experiments on shared trainer infrastructure:

```
┌─────────────────────────────────────────────────────────────────┐
│                      SHARED TRAINER                              │
│                                                                  │
│   Base Model (frozen)                                           │
│        │                                                         │
│        ├─► LoRA Adapter (run_abc123) ─► Optimizer A             │
│        ├─► LoRA Adapter (run_def456) ─► Optimizer B             │
│        └─► LoRA Adapter (run_ghi789) ─► Optimizer C             │
│                                                                  │
│   MultiRunManager:                                               │
│   - Discovers runs from filesystem (run_*/control/orch.toml)   │
│   - Allocates LoRA slots                                        │
│   - Routes batches to correct adapter                           │
│   - Syncs state across distributed ranks                        │
└─────────────────────────────────────────────────────────────────┘
```

Each run gets:
- Its own LoRA adapter weights
- Separate optimizer state
- Independent learning rate schedule
- Isolated metrics and checkpoints

---

## Technology Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Container Orchestration** | Kubernetes | Manage pods, scaling, restarts |
| **Deployment** | Helm | Package and configure deployments |
| **Storage** | NFS/Ceph | Shared filesystem for all components |
| **Inference** | vLLM | Fast LLM inference with OpenAI API |
| **Training** | PyTorch + FSDP2 | Distributed model training |
| **Efficient Training** | LoRA | Low-rank adapters for multi-run |
| **Transport** | ZMQ / Filesystem | Send batches between components |
| **Service Discovery** | DNS | Find inference servers dynamically |
| **Environments** | verifiers | RL environment evaluation |
| **Monitoring** | Prometheus | Metrics collection |
| **Experiment Tracking** | W&B / Prime Monitor | Dashboards and logging |

---

## Further Reading

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Prime Intellect Platform](https://app.primeintellect.ai)
