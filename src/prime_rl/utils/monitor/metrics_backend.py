"""
Metrics Backend Monitor

Logs training metrics to the custom metrics backend for visualization.
"""

import os
import time
from pathlib import Path
from typing import Any

import httpx
import verifiers as vf
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor
from prime_rl.utils.pydantic_config import BaseSettings


class MetricsBackendConfig(BaseConfig):
    """Configuration for Metrics Backend monitor."""

    backend_url: str = "http://localhost:8080"
    api_key_var: str = "METRICS_API_KEY"
    timeout: int = 30
    log_samples: bool = False
    log_distributions: bool = False


class MetricsBackendMonitor(Monitor):
    """Logs to custom metrics backend for visualization."""

    def __init__(
        self,
        config: MetricsBackendConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseSettings | None = None,
    ):
        self.config = config
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0

        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(
                    f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})"
                )
            return

        assert config is not None
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")

        # Get API key from environment variable (optional)
        self.api_key = os.getenv(config.api_key_var)

        self.backend_url = config.backend_url.rstrip("/")

        # Get run_id from environment variable
        self.run_id = os.getenv("RUN_ID")
        if not self.run_id:
            self.logger.warning(
                "RUN_ID environment variable not set. MetricsBackendMonitor will not be able to upload data."
            )
            self.enabled = False
            return

        # Set up HTTP client
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        self.client = httpx.Client(
            timeout=config.timeout,
            headers=headers,
        )

        # Create run in backend
        try:
            self._create_run(run_config)
        except Exception as e:
            self.logger.warning(f"Failed to create run in backend: {e}")
            self.enabled = False

    def _create_run(self, run_config: BaseSettings | None) -> None:
        """Create run in backend."""
        if not self.enabled:
            return

        # Extract environment ID from config if available
        env_id = "unknown"
        if run_config and hasattr(run_config, "orchestrator"):
            if hasattr(run_config.orchestrator, "env"):
                envs = run_config.orchestrator.env
                if envs and len(envs) > 0:
                    env_id = envs[0].get("id", "unknown")

        # Serialize config
        config_dict = None
        if run_config:
            try:
                config_dict = run_config.model_dump()
            except Exception:
                pass

        payload = {
            "run_id": self.run_id,
            "env_id": env_id,
            "name": self.run_id,
            "config": config_dict,
            "instance_id": os.getenv("INSTANCE_ID"),
            "gpu_type": os.getenv("GPU_TYPE"),
        }

        response = self.client.post(
            f"{self.backend_url}/api/v1/runs",
            json=payload,
        )
        response.raise_for_status()
        self.logger.info(f"Created run in metrics backend: {self.run_id}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics."""
        self.history.append(metrics)

        if not self.is_master or not self.enabled:
            return

        try:
            payload = {
                "run_id": self.run_id,
                "step": step or metrics.get("step", 0),
                "metrics": metrics,
            }

            response = self.client.post(
                f"{self.backend_url}/api/v1/metrics",
                json=payload,
            )
            response.raise_for_status()
        except Exception as e:
            self.logger.debug(f"Failed to log metrics to backend: {e}")

    def log_samples(self, rollouts: list[vf.State], step: int) -> None:
        """Log training samples."""
        if not self.is_master or not self.enabled:
            return

        if not self.config or not self.config.log_samples:
            return

        try:
            # Prepare samples
            samples = []
            for rollout in rollouts:
                trajectory = rollout["trajectory"]
                if not trajectory:
                    continue

                last_step = trajectory[-1]

                # Serialize trajectory
                trajectory_data = []
                for traj_step in rollout["trajectory"]:
                    trajectory_data.append(
                        {
                            "prompt": traj_step["prompt"],
                            "completion": traj_step["completion"],
                            "reward": traj_step.get("reward"),
                            "advantage": traj_step.get("advantage"),
                            "extras": traj_step.get("extras", {}),
                        }
                    )

                sample = {
                    "example_id": rollout.get("example_id"),
                    "prompt": last_step["prompt"],
                    "completion": last_step["completion"],
                    "trajectory": trajectory_data,
                    "reward": rollout.get("reward"),
                    "advantage": rollout.get("advantage"),
                    "answer": rollout.get("answer"),
                    "task": rollout.get("task"),
                }
                samples.append(sample)

            payload = {
                "run_id": self.run_id,
                "step": step,
                "samples": samples,
            }

            response = self.client.post(
                f"{self.backend_url}/api/v1/samples",
                json=payload,
            )
            response.raise_for_status()
        except Exception as e:
            self.logger.debug(f"Failed to log samples to backend: {e}")

    def log_distributions(
        self,
        distributions: dict[str, list[float]],
        step: int,
    ) -> None:
        """Log metric distributions."""
        if not self.is_master or not self.enabled:
            return

        if not self.config or not self.config.log_distributions:
            return

        try:
            payload = {
                "run_id": self.run_id,
                "step": step,
                "distributions": distributions,
            }

            response = self.client.post(
                f"{self.backend_url}/api/v1/distributions",
                json=payload,
            )
            response.raise_for_status()
        except Exception as e:
            self.logger.debug(f"Failed to log distributions to backend: {e}")

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary."""
        if not self.is_master or not self.enabled:
            return

        try:
            payload = {
                "status": "completed",
            }

            response = self.client.patch(
                f"{self.backend_url}/api/v1/runs/{self.run_id}/status",
                json=payload,
            )
            response.raise_for_status()
        except Exception as e:
            self.logger.debug(f"Failed to update run status: {e}")

    def close(self) -> None:
        """Close the HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.close()
