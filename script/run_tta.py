import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

# Ensure src is importable when launched from Hydra working dir
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.datamodule import ADE20KCorruptionDataModule
from src.tta.module import TTAModule


def build_logger(cfg: DictConfig, run_name: str):
    if not cfg.logging.get("enable_wandb", True):
        return False
    return WandbLogger(
        project=cfg.logging.get("project", "tta"),
        name=run_name,
        group=cfg.logging.get("experiment_name", "tta"),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def build_trainer(cfg: DictConfig, logger):
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    return pl.Trainer(logger=logger, **trainer_cfg)


def build_module(cfg: DictConfig, corruption: str, severity: int) -> TTAModule:
    mutable_cfg: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    mutable_cfg["data"]["corruption"] = corruption
    mutable_cfg["data"]["severity"] = severity
    return TTAModule(mutable_cfg)


def build_datamodule(cfg: DictConfig, corruption: str, severity: int) -> ADE20KCorruptionDataModule:
    data_root = cfg.data.get("data_root")
    return ADE20KCorruptionDataModule(
        data_root=data_root,
        corruption=corruption,
        severity=severity,
        split=cfg.data.get("split", "val"),
        short_edge=cfg.data.get("short_edge", 512),
        batch_size=cfg.data.get("batch_size", 1),
        num_workers=cfg.data.get("num_workers", 4),
        shuffle=True,
    )


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")
    corruptions = cfg.data.get("corruptions", [cfg.data.get("corruption", "none")])
    severities = cfg.data.get("severities", [cfg.data.get("severity", 1)])

    for idx, corruption in enumerate(corruptions):
        for severity in severities:
            run_suffix = f"{corruption}-s{severity}"
            run_name = f"{cfg.logging.get('experiment_name', 'tta')}-{run_suffix}"
            logger = build_logger(cfg, run_name)
            module = build_module(cfg, corruption, severity)
            datamodule = build_datamodule(cfg, corruption, severity)
            trainer = build_trainer(cfg, logger)
            trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
