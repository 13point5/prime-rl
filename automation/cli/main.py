#!/usr/bin/env python3
"""
Prime-RL GPU Automation CLI

Automates the process of spinning up Prime Intellect GPUs and running RL training.
"""

import argparse
import sys
import time
import yaml
import os
from pathlib import Path
from typing import Optional

from .prime_api_client import (
    PrimeIntellectClient,
    MockPrimeIntellectClient,
    GPUConfig,
    GPUInstance,
    InstanceStatus,
)
from .ssh_orchestrator import SSHOrchestrator, LocalOrchestrator


def load_gpu_config(config_path: str) -> GPUConfig:
    """Load GPU configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    return GPUConfig(
        gpu_type=config_data['gpu_type'],
        gpu_count=config_data.get('gpu_count', 1),
        region=config_data.get('region', 'us-east'),
        image=config_data.get('image', 'ubuntu-22.04-cuda-12.1'),
        disk_size_gb=config_data.get('disk_size_gb', 100),
    )


def create_and_setup_instance(
    client: PrimeIntellectClient,
    gpu_config: GPUConfig,
    instance_name: Optional[str] = None,
    ssh_key_path: Optional[str] = None,
) -> tuple[GPUInstance, SSHOrchestrator]:
    """
    Create GPU instance and set up environment.

    Returns:
        Tuple of (instance, orchestrator)
    """
    print(f"\n{'='*60}")
    print("CREATING GPU INSTANCE")
    print(f"{'='*60}")

    # Create instance
    print(f"\nCreating {gpu_config.gpu_count}x {gpu_config.gpu_type} instance...")
    instance = client.create_instance(
        config=gpu_config,
        name=instance_name,
    )
    print(f"✓ Instance created: {instance.id}")

    # Wait for instance to be running
    print(f"\nWaiting for instance to be running...")
    instance = client.wait_for_instance(
        instance.id,
        target_status=InstanceStatus.RUNNING,
        timeout=600,
    )
    print(f"✓ Instance is running")
    print(f"  IP Address: {instance.ip_address}")
    print(f"  SSH: {instance.ssh_address}")

    # Set up SSH orchestrator
    orchestrator = SSHOrchestrator(
        instance=instance,
        ssh_key_path=ssh_key_path,
        verbose=True,
    )

    # Wait for SSH
    print(f"\nWaiting for SSH to become available...")
    if not orchestrator.wait_for_ssh(timeout=300):
        raise RuntimeError("SSH did not become available in time")
    print("✓ SSH is ready")

    # Set up environment
    print(f"\n{'='*60}")
    print("SETTING UP ENVIRONMENT")
    print(f"{'='*60}\n")

    orchestrator.setup_environment()
    print("\n✓ Environment setup complete")

    return instance, orchestrator


def run_training(
    orchestrator: SSHOrchestrator,
    training_config_path: str,
    env_id: str,
    run_id: Optional[str] = None,
    metrics_backend_url: Optional[str] = None,
    monitor: bool = False,
) -> str:
    """
    Run training on the instance.

    Returns:
        Run ID
    """
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")

    # Generate run ID
    if not run_id:
        run_id = f"run-{env_id}-{int(time.time())}"

    print(f"Run ID: {run_id}")
    print(f"Environment: {env_id}")
    print(f"Config: {training_config_path}")

    # Upload config
    remote_config_path = "~/prime-rl/run_config.toml"
    orchestrator.upload_config(training_config_path, remote_config_path)
    print(f"✓ Config uploaded")

    # Start training
    orchestrator.start_training(
        config_path=remote_config_path,
        env_id=env_id,
        run_id=run_id,
        metrics_backend_url=metrics_backend_url,
        detached=True,
    )
    print(f"✓ Training started in tmux session: training-{run_id}")

    # Monitor if requested
    if monitor:
        print(f"\n{'='*60}")
        print("MONITORING TRAINING")
        print(f"{'='*60}\n")
        print("Press Ctrl+C to stop monitoring (training will continue)\n")

        try:
            while True:
                time.sleep(10)
                logs = orchestrator.get_training_logs(run_id, tail=20)
                print(logs)
                print("-" * 60)
        except KeyboardInterrupt:
            print("\nStopped monitoring (training continues in background)")

    return run_id


def cleanup_instance(
    client: PrimeIntellectClient,
    instance: GPUInstance,
    download_artifacts: bool = False,
    artifacts_path: Optional[str] = None,
) -> None:
    """Clean up GPU instance."""
    print(f"\n{'='*60}")
    print("CLEANING UP")
    print(f"{'='*60}\n")

    if download_artifacts and artifacts_path:
        print("Downloading artifacts...")
        orchestrator = SSHOrchestrator(instance)
        orchestrator.download_artifacts(
            remote_path="~/prime-rl/outputs",
            local_path=artifacts_path,
            recursive=True,
        )
        print(f"✓ Artifacts downloaded to {artifacts_path}")

    print(f"Terminating instance {instance.id}...")
    client.terminate_instance(instance.id)
    print("✓ Instance terminated")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prime-RL GPU Automation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create instance and run training
  python -m automation.cli.main \\
    --gpu-config gpu_config.yaml \\
    --training-config configs/reverse-text.toml \\
    --env-id reverse-text \\
    --monitor

  # List running instances
  python -m automation.cli.main --list-instances

  # Terminate an instance
  python -m automation.cli.main --terminate-instance INSTANCE_ID

  # Monitor existing training
  python -m automation.cli.main \\
    --instance-id INSTANCE_ID \\
    --run-id RUN_ID \\
    --monitor-only
        """,
    )

    # Instance management
    parser.add_argument(
        "--gpu-config",
        type=str,
        help="Path to GPU configuration YAML file",
    )
    parser.add_argument(
        "--instance-name",
        type=str,
        help="Custom name for the instance",
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        help="Existing instance ID to use",
    )

    # Training configuration
    parser.add_argument(
        "--training-config",
        type=str,
        help="Path to training configuration TOML file",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        help="Environment ID to train on",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Custom run ID (generated if not provided)",
    )

    # Metrics backend
    parser.add_argument(
        "--metrics-backend-url",
        type=str,
        help="URL of metrics backend service",
    )

    # Execution options
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor training progress",
    )
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Only monitor existing training (don't start new)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't terminate instance after training",
    )
    parser.add_argument(
        "--download-artifacts",
        action="store_true",
        help="Download artifacts before cleanup",
    )
    parser.add_argument(
        "--artifacts-path",
        type=str,
        default="./artifacts",
        help="Local path to download artifacts",
    )

    # Instance queries
    parser.add_argument(
        "--list-instances",
        action="store_true",
        help="List all instances and exit",
    )
    parser.add_argument(
        "--terminate-instance",
        type=str,
        metavar="INSTANCE_ID",
        help="Terminate specific instance and exit",
    )

    # SSH options
    parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key",
    )

    # Testing
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock API client for testing",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally without creating instance",
    )

    args = parser.parse_args()

    # Initialize API client
    if args.local:
        print("Running in local mode (no instance creation)")
        client = None
    elif args.mock:
        print("Using mock API client")
        client = MockPrimeIntellectClient()
    else:
        client = PrimeIntellectClient()

    try:
        # Handle instance queries
        if args.list_instances:
            instances = client.list_instances()
            print(f"\nFound {len(instances)} instance(s):\n")
            for inst in instances:
                print(f"  {inst.id}")
                print(f"    Name: {inst.name}")
                print(f"    Status: {inst.status.value}")
                print(f"    GPU: {inst.gpu_count}x {inst.gpu_type}")
                print(f"    IP: {inst.ip_address}")
                print()
            return

        if args.terminate_instance:
            print(f"Terminating instance {args.terminate_instance}...")
            client.terminate_instance(args.terminate_instance)
            print("✓ Instance terminated")
            return

        # Handle monitoring-only mode
        if args.monitor_only:
            if not args.instance_id or not args.run_id:
                print("Error: --instance-id and --run-id required for --monitor-only")
                sys.exit(1)

            instance = client.get_instance(args.instance_id)
            orchestrator = SSHOrchestrator(instance, ssh_key_path=args.ssh_key)

            print(f"Monitoring run {args.run_id} on instance {instance.id}...")
            try:
                while True:
                    time.sleep(10)
                    logs = orchestrator.get_training_logs(args.run_id, tail=20)
                    print(logs)
                    print("-" * 60)
            except KeyboardInterrupt:
                print("\nStopped monitoring")
            return

        # Main workflow: create instance and run training
        if not args.training_config or not args.env_id:
            print("Error: --training-config and --env-id are required")
            parser.print_help()
            sys.exit(1)

        if args.local:
            # Local execution
            orchestrator = LocalOrchestrator()
            run_id = run_training(
                orchestrator=orchestrator,
                training_config_path=args.training_config,
                env_id=args.env_id,
                run_id=args.run_id,
                metrics_backend_url=args.metrics_backend_url,
                monitor=args.monitor,
            )
            print(f"\n✓ Training complete! Run ID: {run_id}")

        else:
            # Remote execution
            if args.instance_id:
                # Use existing instance
                instance = client.get_instance(args.instance_id)
                orchestrator = SSHOrchestrator(
                    instance,
                    ssh_key_path=args.ssh_key,
                )
            else:
                # Create new instance
                if not args.gpu_config:
                    print("Error: --gpu-config required when creating new instance")
                    sys.exit(1)

                gpu_config = load_gpu_config(args.gpu_config)
                instance, orchestrator = create_and_setup_instance(
                    client=client,
                    gpu_config=gpu_config,
                    instance_name=args.instance_name,
                    ssh_key_path=args.ssh_key,
                )

            # Run training
            run_id = run_training(
                orchestrator=orchestrator,
                training_config_path=args.training_config,
                env_id=args.env_id,
                run_id=args.run_id,
                metrics_backend_url=args.metrics_backend_url,
                monitor=args.monitor,
            )

            print(f"\n{'='*60}")
            print("TRAINING COMPLETE")
            print(f"{'='*60}\n")
            print(f"Instance ID: {instance.id}")
            print(f"Run ID: {run_id}")
            print(f"SSH: ssh {instance.ssh_address}")
            print(f"View logs: tmux attach -t training-{run_id}")

            # Cleanup
            if not args.no_cleanup:
                cleanup_instance(
                    client=client,
                    instance=instance,
                    download_artifacts=args.download_artifacts,
                    artifacts_path=args.artifacts_path,
                )
            else:
                print(f"\nInstance {instance.id} left running (use --terminate-instance to clean up)")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if client and not isinstance(client, MockPrimeIntellectClient):
            client.close()


if __name__ == "__main__":
    main()
