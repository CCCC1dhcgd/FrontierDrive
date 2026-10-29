#!/usr/bin/env python
"""Train DiffusionDrive with a TBAT manifest sampler."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Tuple


import hydra
import navsim
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module_diffusiondrive import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from frontierdrive.curriculum_sampler import build_curriculum_sampler


logger = logging.getLogger(__name__)

CONFIG_PATH = str(Path(navsim.__file__).resolve().parent / "planning/script/config/training")
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)
    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Global Seed set to %s", cfg.seed)
    logger.info("Path where all results are stored: %s", cfg.output_dir)

    manifest_path = cfg.get("tbat_manifest_path", None)
    if not manifest_path:
        raise ValueError("run_training_tbat.py requires +tbat_manifest_path=/path/to/train_manifest.csv")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    if agent._checkpoint_path is not None:
        agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(agent=agent)

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert not cfg.force_cache_computation, "force_cache_computation must be False when using cached data"
        assert cfg.cache_path is not None, "cache_path must be provided when using cached data"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
            split="train",
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
            split="val",
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building TBAT weighted dataloader")
    train_sampler = build_curriculum_sampler(
        train_data,
        manifest_path,
        rotation_fraction=float(cfg.get("tbat_rotation_fraction", 0.85)),
        epoch_size=(None if int(cfg.get("tbat_epoch_size", -1)) <= 0 else int(cfg.get("tbat_epoch_size"))),
        seed=int(cfg.get("tbat_sampler_seed", cfg.seed)),
    )
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, sampler=train_sampler, shuffle=False)
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num training samples: %d", len(train_data))
    logger.info("Num validation samples: %d", len(val_data))

    tf_logger = pl.loggers.TensorBoardLogger(
        save_dir=str(cfg.output_dir),
        name="tensorboard",
        version=0,
        default_hp_metric=False,
    )

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, logger=tf_logger, callbacks=agent.get_training_callbacks())

    ckpt_path = None
    if cfg.get("resume_ckpt_path", None):
        resume_ckpt_path = f"{os.environ.get('NAVSIM_EXP_ROOT')}/{cfg.resume_ckpt_path}"
        if not os.path.exists(resume_ckpt_path):
            raise FileNotFoundError(f"resume_ckpt_path does not exist: {resume_ckpt_path}")
        logger.info("Resuming from checkpoint %s", resume_ckpt_path)
        ckpt_path = resume_ckpt_path

    logger.info("Starting TBAT Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
