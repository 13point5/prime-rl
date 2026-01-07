"""
Metrics Backend API

FastAPI application for receiving and serving training metrics.
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio

from .database import Database, Run, Metric, Distribution, Sample


# Pydantic models for API
class RunCreate(BaseModel):
    """Create run request."""
    run_id: str
    env_id: str
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    instance_id: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None


class RunResponse(BaseModel):
    """Run response."""
    id: str
    name: str
    env_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    instance_id: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    config: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class MetricSubmit(BaseModel):
    """Submit metrics request."""
    run_id: str
    step: Optional[int] = None
    metrics: Dict[str, Any]


class MetricResponse(BaseModel):
    """Metric response."""
    step: int
    timestamp: datetime
    throughput: Optional[float] = None
    throughput_per_gpu: Optional[float] = None
    mfu: Optional[float] = None
    peak_memory: Optional[float] = None
    lr: Optional[float] = None
    grad_norm: Optional[float] = None
    loss_mean: Optional[float] = None
    entropy_mean: Optional[float] = None
    kl_mean: Optional[float] = None
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    reward_min: Optional[float] = None
    reward_max: Optional[float] = None
    extras: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class DistributionSubmit(BaseModel):
    """Submit distributions request."""
    run_id: str
    step: int
    distributions: Dict[str, List[float]]


class SampleSubmit(BaseModel):
    """Submit samples request."""
    run_id: str
    step: int
    samples: List[Dict[str, Any]]


class StatusUpdate(BaseModel):
    """Update run status."""
    status: str
    completed_at: Optional[datetime] = None


# Create FastAPI app
app = FastAPI(
    title="Prime-RL Metrics API",
    description="API for collecting and serving RL training metrics",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database instance
db = Database(
    database_url=os.getenv("DATABASE_URL", "sqlite:///metrics.db")
)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, run_id: str):
        """Connect a WebSocket client."""
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)

    def disconnect(self, websocket: WebSocket, run_id: str):
        """Disconnect a WebSocket client."""
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def broadcast(self, run_id: str, message: dict):
        """Broadcast message to all clients watching a run."""
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass  # Client disconnected


manager = ConnectionManager()


# API key validation (optional)
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if configured."""
    expected_key = os.getenv("BACKEND_API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


# Routes
@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "prime-rl-metrics"}


@app.post("/api/v1/runs", response_model=RunResponse)
async def create_run(
    run_create: RunCreate,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Create a new training run."""
    try:
        run = db.create_run(
            run_id=run_create.run_id,
            env_id=run_create.env_id,
            name=run_create.name,
            config=run_create.config,
            instance_id=run_create.instance_id,
            gpu_type=run_create.gpu_type,
            gpu_count=run_create.gpu_count,
        )
        return RunResponse.from_orm(run)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/runs", response_model=List[RunResponse])
async def list_runs(limit: int = 100):
    """List all training runs."""
    runs = db.list_runs(limit=limit)
    return [RunResponse.from_orm(run) for run in runs]


@app.get("/api/v1/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str):
    """Get a specific training run."""
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse.from_orm(run)


@app.patch("/api/v1/runs/{run_id}/status")
async def update_run_status(
    run_id: str,
    status_update: StatusUpdate,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Update run status."""
    db.update_run_status(
        run_id=run_id,
        status=status_update.status,
        completed_at=status_update.completed_at,
    )

    # Broadcast status update
    await manager.broadcast(run_id, {
        "type": "status_update",
        "status": status_update.status,
    })

    return {"status": "ok"}


@app.post("/api/v1/metrics")
async def submit_metrics(
    metric_submit: MetricSubmit,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Submit training metrics."""
    # Get step from metrics if not provided
    step = metric_submit.step
    if step is None:
        step = metric_submit.metrics.get("step", 0)

    db.add_metric(
        run_id=metric_submit.run_id,
        step=step,
        metrics=metric_submit.metrics,
    )

    # Broadcast metrics update
    await manager.broadcast(metric_submit.run_id, {
        "type": "metrics",
        "step": step,
        "metrics": metric_submit.metrics,
    })

    return {"status": "ok", "step": step}


@app.get("/api/v1/runs/{run_id}/metrics", response_model=List[MetricResponse])
async def get_metrics(
    run_id: str,
    limit: int = 1000,
    offset: int = 0,
):
    """Get metrics for a training run."""
    metrics = db.get_metrics(run_id=run_id, limit=limit, offset=offset)
    return [MetricResponse.from_orm(m) for m in metrics]


@app.post("/api/v1/distributions")
async def submit_distributions(
    dist_submit: DistributionSubmit,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Submit distribution data."""
    db.add_distribution(
        run_id=dist_submit.run_id,
        step=dist_submit.step,
        distributions=dist_submit.distributions,
    )

    # Broadcast distribution update
    await manager.broadcast(dist_submit.run_id, {
        "type": "distributions",
        "step": dist_submit.step,
        "distributions": dist_submit.distributions,
    })

    return {"status": "ok"}


@app.get("/api/v1/runs/{run_id}/distributions")
async def get_distributions(run_id: str, step: Optional[int] = None):
    """Get distributions for a training run."""
    distributions = db.get_distributions(run_id=run_id, step=step)
    return [
        {
            "step": d.step,
            "name": d.name,
            "values": d.values,
            "timestamp": d.timestamp,
        }
        for d in distributions
    ]


@app.post("/api/v1/samples")
async def submit_samples(
    sample_submit: SampleSubmit,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Submit training samples."""
    for sample in sample_submit.samples:
        db.add_sample(
            run_id=sample_submit.run_id,
            step=sample_submit.step,
            sample=sample,
        )

    return {"status": "ok", "count": len(sample_submit.samples)}


@app.get("/api/v1/runs/{run_id}/samples")
async def get_samples(
    run_id: str,
    step: Optional[int] = None,
    limit: int = 100,
):
    """Get samples for a training run."""
    samples = db.get_samples(run_id=run_id, step=step, limit=limit)
    return [
        {
            "step": s.step,
            "example_id": s.example_id,
            "prompt": s.prompt,
            "completion": s.completion,
            "answer": s.answer,
            "task": s.task,
            "reward": s.reward,
            "advantage": s.advantage,
            "trajectory": s.trajectory,
            "timestamp": s.timestamp,
        }
        for s in samples
    ]


@app.websocket("/ws/runs/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, run_id)
    try:
        while True:
            # Keep connection alive and wait for messages
            data = await websocket.receive_text()
            # Echo back (can be used for ping/pong)
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
    )
