"""Logging utilities for TTA experiments."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Setup basic logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


class WandBLogger:
    """Weights & Biases logger for experiment tracking."""

    def __init__(
        self,
        config: DictConfig,
        enabled: bool = True,
    ) -> None:
        """Initialize W&B logger.

        Args:
            config: Hydra configuration
            enabled: Whether to enable W&B logging
        """
        self.enabled = enabled
        self._run = None

        if not enabled:
            logging.info("W&B logging disabled")
            return

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            logging.warning("wandb not installed, disabling W&B logging")
            self.enabled = False
            return

        # Extract experiment info
        experiment_name = config.get("experiment", {}).get("name", "sd3_tta")
        experiment_id = config.get("experiment", {}).get("id", 1)
        project = config.get("logging", {}).get("wandb", {}).get("project", "sd3-tta-seg")
        entity = config.get("logging", {}).get("wandb", {}).get("entity", None)

        # Create unique run name with experiment ID
        run_name = f"{experiment_name}_{experiment_id}"

        # Initialize W&B
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=OmegaConf.to_container(config, resolve=True),
            reinit=True,
        )

        logging.info(f"W&B initialized: {run_name}")

    @property
    def run(self):
        """Get the W&B run object."""
        return self._run

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step number
            prefix: Prefix to add to metric names (e.g., "train/", "eval/")
        """
        if not self.enabled or self._run is None:
            return

        # Add prefix to metric names
        if prefix:
            prefix = prefix.rstrip("/") + "/"
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        self._wandb.log(metrics, step=step)

    def log_task_metrics(
        self,
        metrics: Dict[str, Any],
        task_name: str,
        step: int,
    ) -> None:
        """Log metrics for a specific task/corruption.

        Args:
            metrics: Dictionary of metric names and values
            task_name: Name of the task/corruption
            step: Global step number
        """
        if not self.enabled or self._run is None:
            return

        # Add task-specific prefix
        task_metrics = {f"{task_name}/{k}": v for k, v in metrics.items()}
        self._wandb.log(task_metrics, step=step)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log summary metrics at the end of experiment.

        Args:
            summary: Dictionary of summary metrics
        """
        if not self.enabled or self._run is None:
            return

        for key, value in summary.items():
            self._run.summary[key] = value

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled and self._run is not None:
            self._run.finish()
            logging.info("W&B run finished")
