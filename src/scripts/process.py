from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import decord
import hydra
import numpy as np
import polars as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import utils
from src.models import AbstractDetector


@hydra.main(config_path="config", config_name="process", version_base=None)
def process(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    model: AbstractDetector = hydra.utils.instantiate(cfg.model)

    # Run detector for each batch and save detections to the polars dataframe
    dataframes = []
    decord.bridge.set_bridge("torch")
    video_reader = utils.BatchedVideoReader(
        cfg.path, cfg.device, cfg=utils.VideoConfig(**cfg.video_config)
    )
    logger.info(f"Total frames: {video_reader.frame_number()}")

    for indices, batch in tqdm(video_reader):
        for idx, dets in zip(indices, model.process(batch)):
            df = pl.DataFrame(
                {
                    "score": dets.scores,
                    "x1": dets.x1,
                    "y1": dets.y1,
                    "x2": dets.x2,
                    "y2": dets.y2,
                    "label": dets.labels,
                }
            )
            df = df.with_columns(pl.repeat(idx, pl.len()).alias("frame"))
            dataframes.append(df)
    df: pl.DataFrame = pl.concat(dataframes)

    # Save results to parquet
    Path(cfg.output).parent.mkdir(exist_ok=True, parents=True)
    df.write_parquet(cfg.output)


if "__main__" == __name__:
    process()
