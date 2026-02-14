"""SD3-TTA Main Entry Point.

Run with:
    python -m src.main

Or with custom config:
    python -m src.main experiment=baseline
    python -m src.main experiment.id=2 tta.forward_mode=discriminative_only
"""

import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_environment() -> None:
    """Setup environment variables and logging."""
    # Ensure CUDA is properly configured
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        logging.warning(
            "CUDA_VISIBLE_DEVICES not set. "
            "Consider setting it explicitly for multi-GPU runs."
        )

    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def print_config(config: DictConfig) -> None:
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("SD3-TTA Configuration")
    print("=" * 60)
    print(f"Experiment: {config.experiment.name} (ID: {config.experiment.id})")
    print(f"Forward Mode: {config.tta.forward_mode}")
    print(f"Continual: {config.tta.continual}")
    print(f"Precision: {config.device.precision}")
    print(f"Data Root: {config.data.root}")
    dataset = config.data.get("dataset", "ADE20K-C")
    tasks = config.data.corruptions
    severity = config.data.get("severity", None)
    if severity is not None:
        print(f"Tasks: {len(tasks)} corruptions, severity {severity}")
    else:
        print(f"Tasks: {len(tasks)} conditions")
    print("=" * 60 + "\n")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(config: DictConfig) -> None:
    """Main entry point for SD3-TTA.

    Args:
        config: Hydra configuration
    """
    # Setup environment
    setup_environment()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Print configuration
    print_config(config)

    # Log full config
    logger.info("Full configuration:")
    logger.info(OmegaConf.to_yaml(config))

    # Import here to avoid circular imports
    from src.tta.runner import run_tta

    # Run TTA
    try:
        results = run_tta(config)

        # Print final results
        print("\n" + "=" * 60)
        print("Final Results")
        print("=" * 60)
        for metric, value in sorted(results.items()):
            print(f"  {metric}: {value:.4f}")
        print("=" * 60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during TTA: {e}")
        raise


if __name__ == "__main__":
    main()
