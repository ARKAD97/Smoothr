from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import decord
import hydra
import numpy as np
import polars as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.models import AbstractDetector

decord.bridge.set_bridge("torch")


@dataclass
class VideoConfig:
    start: int = 0
    end: Optional[int] = None
    batch_size: int = 1
    skip: int = 0


def read_video(path: str, device: str, cfg: VideoConfig) -> Iterator[np.ndarray]:
    """
    Read video in batches for better performance.
    """

    # TODO: build from sources to enable gpu acceleratoion https://github.com/dmlc/decord
    video_reader = decord.VideoReader(path, ctx=decord.cpu(0))

    logger.info(f"Total frames: {len(video_reader)}")

    # if segment is specified, read till the end of the video
    end = len(video_reader)
    end = min(cfg.end, end) if cfg.end else end

    # Read batches for specified segment skiping specified number of frames
    frames_per_inference = cfg.skip + 1
    for start_idx in range(cfg.start, end, cfg.batch_size * frames_per_inference):
        end_idx = min(start_idx + cfg.batch_size * frames_per_inference, end)
        indices = list(range(start_idx, end_idx, frames_per_inference))

        batch = (
            video_reader.get_batch(indices).to(device).to(torch.float32)
        )  # (N, H, W, 3)
        batch /= 255
        batch = batch.permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
        yield indices, batch


@hydra.main(config_path="config", config_name="process", version_base=None)
def process(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    model: AbstractDetector = hydra.utils.instantiate(cfg.model)

    # Run detector for each batch and save detections to the polars dataframe
    dataframes = []
    for indices, batch in tqdm(
        read_video(cfg.path, cfg.device, cfg=VideoConfig(**cfg.video_config))
    ):
        for idx, detections in zip(indices, model.process(batch)):
            x1, y1, x2, y2 = [detections.boxes[:, i] for i in range(4)]
            df = pl.DataFrame(
                {
                    "score": detections.scores,
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "labels": detections.labels,
                }
            )
            df = df.with_columns(pl.repeat(idx, pl.len()).alias("frame"))
            dataframes.append(df)
    df: pl.DataFrame = pl.concat(dataframes)

    # Save results to parquet
    Path(cfg.output).parent.mkdir(exist_ok=True, parents=True)
    df.write_parquet(cfg.outputs)


if "__main__" == __name__:
    process()
