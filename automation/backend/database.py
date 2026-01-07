"""
Database models and connection for metrics storage.
"""

from datetime import datetime
from typing import Optional, List
import json

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    JSON,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

Base = declarative_base()


class Run(Base):
    """Training run."""
    __tablename__ = "runs"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    env_id = Column(String, nullable=False)
    status = Column(String, default="running")  # running, completed, failed, cancelled
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Metadata
    instance_id = Column(String, nullable=True)
    gpu_type = Column(String, nullable=True)
    gpu_count = Column(Integer, nullable=True)


class Metric(Base):
    """Metric data point."""
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Performance metrics
    throughput = Column(Float, nullable=True)
    throughput_per_gpu = Column(Float, nullable=True)
    mfu = Column(Float, nullable=True)  # Model FLOPS utilization
    peak_memory = Column(Float, nullable=True)

    # Optimizer metrics
    lr = Column(Float, nullable=True)  # Learning rate
    grad_norm = Column(Float, nullable=True)

    # Loss metrics
    loss_mean = Column(Float, nullable=True)
    entropy_mean = Column(Float, nullable=True)
    kl_mean = Column(Float, nullable=True)

    # Reward metrics (RL-specific)
    reward_mean = Column(Float, nullable=True)
    reward_std = Column(Float, nullable=True)
    reward_min = Column(Float, nullable=True)
    reward_max = Column(Float, nullable=True)

    # Custom metrics (stored as JSON)
    extras = Column(JSON, nullable=True)

    # Indexes for efficient queries
    __table_args__ = (
        Index('ix_metrics_run_step', 'run_id', 'step'),
        Index('ix_metrics_run_timestamp', 'run_id', 'timestamp'),
    )


class Distribution(Base):
    """Distribution data (e.g., reward distributions)."""
    __tablename__ = "distributions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    step = Column(Integer, nullable=False)
    name = Column(String, nullable=False)  # e.g., "reward", "lcs_reward_func"
    values = Column(JSON, nullable=False)  # List of values
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_distributions_run_step', 'run_id', 'step'),
    )


class Sample(Base):
    """Training sample/rollout data."""
    __tablename__ = "samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    step = Column(Integer, nullable=False)
    example_id = Column(String, nullable=True)

    # Prompt and completion
    prompt = Column(JSON, nullable=True)  # List of messages
    completion = Column(JSON, nullable=True)  # List of messages
    answer = Column(Text, nullable=True)
    task = Column(Text, nullable=True)

    # Metrics for this sample
    reward = Column(Float, nullable=True)
    advantage = Column(Float, nullable=True)

    # Full trajectory data
    trajectory = Column(JSON, nullable=True)

    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_samples_run_step', 'run_id', 'step'),
    )


class Database:
    """Database manager."""

    def __init__(self, database_url: str = "sqlite:///metrics.db"):
        """
        Initialize database connection.

        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def create_run(
        self,
        run_id: str,
        env_id: str,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        instance_id: Optional[str] = None,
        gpu_type: Optional[str] = None,
        gpu_count: Optional[int] = None,
    ) -> Run:
        """Create a new training run."""
        session = self.get_session()
        try:
            run = Run(
                id=run_id,
                name=name or run_id,
                env_id=env_id,
                config=config,
                instance_id=instance_id,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run
        finally:
            session.close()

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a training run by ID."""
        session = self.get_session()
        try:
            return session.query(Run).filter(Run.id == run_id).first()
        finally:
            session.close()

    def list_runs(self, limit: int = 100) -> List[Run]:
        """List all training runs."""
        session = self.get_session()
        try:
            return (
                session.query(Run)
                .order_by(Run.created_at.desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def update_run_status(
        self,
        run_id: str,
        status: str,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """Update run status."""
        session = self.get_session()
        try:
            run = session.query(Run).filter(Run.id == run_id).first()
            if run:
                run.status = status
                run.updated_at = datetime.utcnow()
                if completed_at:
                    run.completed_at = completed_at
                session.commit()
        finally:
            session.close()

    def add_metric(self, run_id: str, step: int, metrics: dict) -> None:
        """Add a metric data point."""
        session = self.get_session()
        try:
            # Extract known metrics
            metric = Metric(
                run_id=run_id,
                step=step,
                throughput=metrics.get("perf/throughput"),
                throughput_per_gpu=metrics.get("perf/throughput_per_gpu"),
                mfu=metrics.get("perf/mfu"),
                peak_memory=metrics.get("perf/peak_memory"),
                lr=metrics.get("optim/lr"),
                grad_norm=metrics.get("optim/grad_norm"),
                loss_mean=metrics.get("loss/mean"),
                entropy_mean=metrics.get("entropy/mean"),
                kl_mean=metrics.get("mismatch_kl/mean") or metrics.get("kl/mean"),
                reward_mean=metrics.get("reward/mean"),
                reward_std=metrics.get("reward/std"),
                reward_min=metrics.get("reward/min"),
                reward_max=metrics.get("reward/max"),
                extras=metrics,  # Store all metrics as extras too
            )
            session.add(metric)
            session.commit()
        finally:
            session.close()

    def get_metrics(
        self,
        run_id: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[Metric]:
        """Get metrics for a run."""
        session = self.get_session()
        try:
            return (
                session.query(Metric)
                .filter(Metric.run_id == run_id)
                .order_by(Metric.step)
                .limit(limit)
                .offset(offset)
                .all()
            )
        finally:
            session.close()

    def add_distribution(
        self,
        run_id: str,
        step: int,
        distributions: dict[str, list],
    ) -> None:
        """Add distribution data."""
        session = self.get_session()
        try:
            for name, values in distributions.items():
                dist = Distribution(
                    run_id=run_id,
                    step=step,
                    name=name,
                    values=values,
                )
                session.add(dist)
            session.commit()
        finally:
            session.close()

    def get_distributions(
        self,
        run_id: str,
        step: Optional[int] = None,
    ) -> List[Distribution]:
        """Get distributions for a run."""
        session = self.get_session()
        try:
            query = session.query(Distribution).filter(
                Distribution.run_id == run_id
            )
            if step is not None:
                query = query.filter(Distribution.step == step)
            return query.order_by(Distribution.step).all()
        finally:
            session.close()

    def add_sample(
        self,
        run_id: str,
        step: int,
        sample: dict,
    ) -> None:
        """Add a training sample."""
        session = self.get_session()
        try:
            s = Sample(
                run_id=run_id,
                step=step,
                example_id=sample.get("example_id"),
                prompt=sample.get("prompt"),
                completion=sample.get("completion"),
                answer=sample.get("answer"),
                task=sample.get("task"),
                reward=sample.get("reward"),
                advantage=sample.get("advantage"),
                trajectory=sample.get("trajectory"),
            )
            session.add(s)
            session.commit()
        finally:
            session.close()

    def get_samples(
        self,
        run_id: str,
        step: Optional[int] = None,
        limit: int = 100,
    ) -> List[Sample]:
        """Get samples for a run."""
        session = self.get_session()
        try:
            query = session.query(Sample).filter(Sample.run_id == run_id)
            if step is not None:
                query = query.filter(Sample.step == step)
            return query.order_by(Sample.step).limit(limit).all()
        finally:
            session.close()
