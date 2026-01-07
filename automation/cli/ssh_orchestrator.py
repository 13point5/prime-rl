"""
SSH Orchestration for Prime-RL Training

Handles SSH connections to GPU instances and orchestrates training setup/execution.
"""

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
import tempfile

from .prime_api_client import GPUInstance


class SSHOrchestrator:
    """
    Orchestrates training setup and execution on remote GPU instances via SSH.
    """

    def __init__(
        self,
        instance: GPUInstance,
        ssh_key_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize SSH orchestrator.

        Args:
            instance: GPU instance to connect to
            ssh_key_path: Path to SSH private key
            verbose: Whether to print verbose output
        """
        self.instance = instance
        self.ssh_key_path = ssh_key_path or os.path.expanduser("~/.ssh/id_rsa")
        self.verbose = verbose

    def wait_for_ssh(self, timeout: int = 300, poll_interval: int = 5) -> bool:
        """
        Wait for SSH to become available.

        Args:
            timeout: Maximum wait time in seconds
            poll_interval: Time between connection attempts

        Returns:
            True if SSH becomes available, False otherwise
        """
        if self.verbose:
            print(f"Waiting for SSH to become available on {self.instance.ssh_address}...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                result = self._run_ssh_command(
                    "echo 'SSH ready'",
                    check=False,
                    capture_output=True
                )
                if result.returncode == 0:
                    if self.verbose:
                        print("SSH is ready!")
                    return True
            except Exception as e:
                if self.verbose:
                    print(f"SSH not ready yet: {e}")

            time.sleep(poll_interval)

        return False

    def setup_environment(self, repo_url: str = "https://github.com/PrimeIntellect-ai/prime-rl.git") -> bool:
        """
        Set up the training environment on the instance.

        Args:
            repo_url: Git repository URL to clone

        Returns:
            True if setup successful
        """
        if self.verbose:
            print("Setting up environment...")

        commands = [
            # Update system
            "sudo apt-get update",

            # Install basic dependencies
            "sudo apt-get install -y git curl build-essential",

            # Clone repository
            f"git clone {repo_url} prime-rl || (cd prime-rl && git pull)",

            # Install uv
            "curl -LsSf https://astral.sh/uv/install.sh | sh",

            # Source uv environment
            "source $HOME/.local/bin/env",

            # Install dependencies
            "cd prime-rl && $HOME/.local/bin/uv sync --all-extras",
        ]

        for cmd in commands:
            if self.verbose:
                print(f"Running: {cmd}")

            result = self._run_ssh_command(cmd, check=False)
            if result.returncode != 0 and "already exists" not in result.stderr:
                print(f"Warning: Command failed: {cmd}")
                # Continue anyway for some commands that might fail

        return True

    def upload_config(self, local_config_path: str, remote_path: str = "~/prime-rl/config.toml") -> bool:
        """
        Upload training configuration file to instance.

        Args:
            local_config_path: Path to local config file
            remote_path: Destination path on remote instance

        Returns:
            True if upload successful
        """
        if self.verbose:
            print(f"Uploading config from {local_config_path} to {remote_path}...")

        scp_cmd = self._build_scp_command(local_config_path, remote_path)

        result = subprocess.run(scp_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error uploading config: {result.stderr}")
            return False

        return True

    def start_training(
        self,
        config_path: str,
        env_id: str,
        run_id: Optional[str] = None,
        metrics_backend_url: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        detached: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Start training on the remote instance.

        Args:
            config_path: Path to config file on remote instance
            env_id: Environment ID to train on
            run_id: Unique run identifier
            metrics_backend_url: URL of metrics backend
            extra_args: Additional arguments for training
            detached: Whether to run in detached mode (tmux/screen)

        Returns:
            Subprocess result
        """
        if self.verbose:
            print("Starting training...")

        # Generate run ID if not provided
        if not run_id:
            run_id = f"run-{int(time.time())}"

        # Build environment variables
        env_vars = [
            f"export RUN_ID={run_id}",
        ]

        if metrics_backend_url:
            env_vars.append(f"export METRICS_BACKEND_URL={metrics_backend_url}")

        # Build training command
        train_cmd = f"cd prime-rl && source $HOME/.local/bin/env && uv run rl --trainer @ {config_path} --orchestrator @ {config_path} --inference @ {config_path}"

        # Add extra arguments
        if extra_args:
            for key, value in extra_args.items():
                train_cmd += f" --{key} {value}"

        # Combine environment setup and training command
        full_cmd = " && ".join(env_vars + [train_cmd])

        # Wrap in tmux if detached
        if detached:
            session_name = f"training-{run_id}"
            full_cmd = f"tmux new-session -d -s {session_name} '{full_cmd}'"

        return self._run_ssh_command(full_cmd)

    def get_training_logs(self, run_id: str, tail: int = 100) -> str:
        """
        Get training logs from tmux session.

        Args:
            run_id: Run identifier
            tail: Number of lines to retrieve

        Returns:
            Log output
        """
        session_name = f"training-{run_id}"
        cmd = f"tmux capture-pane -p -t {session_name} -S -{tail}"

        result = self._run_ssh_command(cmd, check=False, capture_output=True)

        if result.returncode != 0:
            return f"Error getting logs: {result.stderr}"

        return result.stdout

    def stop_training(self, run_id: str) -> bool:
        """
        Stop a training run.

        Args:
            run_id: Run identifier

        Returns:
            True if stopped successfully
        """
        session_name = f"training-{run_id}"
        cmd = f"tmux kill-session -t {session_name}"

        result = self._run_ssh_command(cmd, check=False)
        return result.returncode == 0

    def download_artifacts(
        self,
        remote_path: str,
        local_path: str,
        recursive: bool = True,
    ) -> bool:
        """
        Download training artifacts from instance.

        Args:
            remote_path: Path on remote instance
            local_path: Local destination path
            recursive: Whether to download recursively

        Returns:
            True if download successful
        """
        if self.verbose:
            print(f"Downloading artifacts from {remote_path} to {local_path}...")

        scp_cmd = self._build_scp_command(
            remote_path,
            local_path,
            download=True,
            recursive=recursive
        )

        result = subprocess.run(scp_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error downloading artifacts: {result.stderr}")
            return False

        return True

    def _run_ssh_command(
        self,
        command: str,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess:
        """
        Run a command via SSH.

        Args:
            command: Command to run
            check: Whether to raise on non-zero exit
            capture_output: Whether to capture stdout/stderr

        Returns:
            Subprocess result
        """
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
        ]

        if self.ssh_key_path:
            ssh_cmd.extend(["-i", self.ssh_key_path])

        ssh_cmd.extend([
            f"{self.instance.ssh_user}@{self.instance.ip_address}",
            command
        ])

        kwargs = {}
        if capture_output:
            kwargs["capture_output"] = True
            kwargs["text"] = True

        result = subprocess.run(ssh_cmd, check=False, **kwargs)

        if check and result.returncode != 0:
            error_msg = f"SSH command failed: {command}"
            if capture_output:
                error_msg += f"\nStderr: {result.stderr}"
            raise RuntimeError(error_msg)

        return result

    def _build_scp_command(
        self,
        source: str,
        dest: str,
        download: bool = False,
        recursive: bool = False,
    ) -> List[str]:
        """
        Build SCP command.

        Args:
            source: Source path
            dest: Destination path
            download: Whether this is a download (vs upload)
            recursive: Whether to copy recursively

        Returns:
            SCP command as list
        """
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
        ]

        if recursive:
            scp_cmd.append("-r")

        if self.ssh_key_path:
            scp_cmd.extend(["-i", self.ssh_key_path])

        if download:
            # Download: remote -> local
            scp_cmd.extend([
                f"{self.instance.ssh_user}@{self.instance.ip_address}:{source}",
                dest
            ])
        else:
            # Upload: local -> remote
            scp_cmd.extend([
                source,
                f"{self.instance.ssh_user}@{self.instance.ip_address}:{dest}"
            ])

        return scp_cmd


class LocalOrchestrator:
    """
    Orchestrator for local execution (no SSH, for testing).
    """

    def __init__(self, work_dir: Optional[str] = None, verbose: bool = True):
        self.work_dir = work_dir or os.getcwd()
        self.verbose = verbose

    def wait_for_ssh(self, *args, **kwargs) -> bool:
        """Mock wait for SSH."""
        return True

    def setup_environment(self, *args, **kwargs) -> bool:
        """Mock setup."""
        if self.verbose:
            print("Running locally - skipping environment setup")
        return True

    def upload_config(self, local_config_path: str, *args, **kwargs) -> bool:
        """Mock upload - config already local."""
        if self.verbose:
            print(f"Using local config: {local_config_path}")
        return True

    def start_training(
        self,
        config_path: str,
        env_id: str,
        run_id: Optional[str] = None,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """Start training locally."""
        if not run_id:
            run_id = f"run-{int(time.time())}"

        if self.verbose:
            print(f"Starting local training with run_id={run_id}")

        # Build command
        cmd = f"RUN_ID={run_id} uv run rl --trainer @ {config_path} --orchestrator @ {config_path} --inference @ {config_path}"

        return subprocess.run(cmd, shell=True, cwd=self.work_dir)

    def get_training_logs(self, *args, **kwargs) -> str:
        """Mock get logs."""
        return "Local training - logs written to stdout"

    def stop_training(self, *args, **kwargs) -> bool:
        """Mock stop."""
        return True
