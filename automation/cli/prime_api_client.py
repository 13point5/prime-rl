"""
Prime Intellect API Client

This client handles interactions with the Prime Intellect GPU management API.
Note: API endpoints are based on common cloud GPU provider patterns.
Update these based on actual Prime Intellect API documentation.
"""

import os
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import httpx


class InstanceStatus(str, Enum):
    """GPU instance status."""
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class GPUInstance:
    """Represents a GPU instance."""
    id: str
    name: str
    status: InstanceStatus
    gpu_type: str
    gpu_count: int
    ip_address: Optional[str] = None
    ssh_port: int = 22
    ssh_user: str = "root"
    created_at: Optional[str] = None

    @property
    def ssh_address(self) -> Optional[str]:
        """Get SSH connection string."""
        if self.ip_address:
            return f"{self.ssh_user}@{self.ip_address}"
        return None


@dataclass
class GPUConfig:
    """GPU instance configuration."""
    gpu_type: str  # e.g., "H100", "A100", "RTX4090"
    gpu_count: int = 1
    region: str = "us-east"
    image: str = "ubuntu-22.04-cuda-12.1"
    disk_size_gb: int = 100


class PrimeIntellectClient:
    """
    Client for Prime Intellect GPU Management API.

    Environment Variables:
        PRIME_API_KEY: API key for authentication
        PRIME_API_URL: Base URL for API (default: https://api.primeintellect.ai)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.getenv("PRIME_API_KEY")
        if not self.api_key:
            raise ValueError("PRIME_API_KEY environment variable not set")

        self.base_url = (base_url or os.getenv("PRIME_API_URL") or
                        "https://api.primeintellect.ai/api/v1")

        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def create_instance(
        self,
        config: GPUConfig,
        name: Optional[str] = None,
        ssh_key: Optional[str] = None,
    ) -> GPUInstance:
        """
        Create a new GPU instance.

        Args:
            config: GPU configuration
            name: Optional instance name
            ssh_key: Optional SSH public key for access

        Returns:
            Created GPU instance
        """
        if not name:
            name = f"prime-rl-{int(time.time())}"

        payload = {
            "name": name,
            "gpu_type": config.gpu_type,
            "gpu_count": config.gpu_count,
            "region": config.region,
            "image": config.image,
            "disk_size_gb": config.disk_size_gb,
        }

        if ssh_key:
            payload["ssh_key"] = ssh_key

        # TODO: Update endpoint based on actual API docs
        response = self.client.post(f"{self.base_url}/instances", json=payload)
        response.raise_for_status()

        data = response.json()
        return self._parse_instance(data)

    def get_instance(self, instance_id: str) -> GPUInstance:
        """Get instance details."""
        response = self.client.get(f"{self.base_url}/instances/{instance_id}")
        response.raise_for_status()
        return self._parse_instance(response.json())

    def list_instances(self) -> List[GPUInstance]:
        """List all instances."""
        response = self.client.get(f"{self.base_url}/instances")
        response.raise_for_status()
        data = response.json()
        instances = data.get("instances", data)  # Handle both formats
        return [self._parse_instance(inst) for inst in instances]

    def stop_instance(self, instance_id: str) -> GPUInstance:
        """Stop a running instance."""
        response = self.client.post(
            f"{self.base_url}/instances/{instance_id}/stop"
        )
        response.raise_for_status()
        return self._parse_instance(response.json())

    def start_instance(self, instance_id: str) -> GPUInstance:
        """Start a stopped instance."""
        response = self.client.post(
            f"{self.base_url}/instances/{instance_id}/start"
        )
        response.raise_for_status()
        return self._parse_instance(response.json())

    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance (permanent)."""
        response = self.client.delete(
            f"{self.base_url}/instances/{instance_id}"
        )
        response.raise_for_status()
        return True

    def wait_for_instance(
        self,
        instance_id: str,
        target_status: InstanceStatus = InstanceStatus.RUNNING,
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> GPUInstance:
        """
        Wait for instance to reach target status.

        Args:
            instance_id: Instance ID
            target_status: Desired status
            timeout: Maximum wait time in seconds
            poll_interval: Time between status checks

        Returns:
            Instance when it reaches target status

        Raises:
            TimeoutError: If instance doesn't reach status in time
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            instance = self.get_instance(instance_id)

            if instance.status == target_status:
                return instance

            if instance.status == InstanceStatus.ERROR:
                raise RuntimeError(f"Instance {instance_id} entered error state")

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Instance {instance_id} did not reach {target_status} "
            f"within {timeout} seconds"
        )

    def _parse_instance(self, data: Dict[str, Any]) -> GPUInstance:
        """Parse instance data from API response."""
        return GPUInstance(
            id=data["id"],
            name=data["name"],
            status=InstanceStatus(data["status"]),
            gpu_type=data["gpu_type"],
            gpu_count=data["gpu_count"],
            ip_address=data.get("ip_address") or data.get("public_ip"),
            ssh_port=data.get("ssh_port", 22),
            ssh_user=data.get("ssh_user", "root"),
            created_at=data.get("created_at"),
        )

    def get_available_gpu_types(self) -> List[Dict[str, Any]]:
        """Get list of available GPU types and their specs."""
        response = self.client.get(f"{self.base_url}/gpu-types")
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Mock client for testing when API is not available
class MockPrimeIntellectClient(PrimeIntellectClient):
    """Mock client for testing without actual API access."""

    def __init__(self, *args, **kwargs):
        # Don't call parent __init__ to avoid requiring API key
        self.instances: Dict[str, GPUInstance] = {}
        self.instance_counter = 0

    def create_instance(
        self,
        config: GPUConfig,
        name: Optional[str] = None,
        ssh_key: Optional[str] = None,
    ) -> GPUInstance:
        """Mock instance creation."""
        self.instance_counter += 1
        instance_id = f"mock-{self.instance_counter}"

        if not name:
            name = f"prime-rl-{int(time.time())}"

        instance = GPUInstance(
            id=instance_id,
            name=name,
            status=InstanceStatus.PENDING,
            gpu_type=config.gpu_type,
            gpu_count=config.gpu_count,
            ip_address=f"192.168.1.{self.instance_counter}",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        self.instances[instance_id] = instance
        return instance

    def get_instance(self, instance_id: str) -> GPUInstance:
        """Mock get instance."""
        if instance_id not in self.instances:
            raise ValueError(f"Instance {instance_id} not found")
        return self.instances[instance_id]

    def list_instances(self) -> List[GPUInstance]:
        """Mock list instances."""
        return list(self.instances.values())

    def wait_for_instance(
        self,
        instance_id: str,
        target_status: InstanceStatus = InstanceStatus.RUNNING,
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> GPUInstance:
        """Mock wait - immediately set to running."""
        instance = self.get_instance(instance_id)
        instance.status = target_status
        self.instances[instance_id] = instance
        return instance

    def terminate_instance(self, instance_id: str) -> bool:
        """Mock terminate."""
        if instance_id in self.instances:
            del self.instances[instance_id]
        return True

    def close(self):
        """Mock close."""
        pass
