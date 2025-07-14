from typing import Any, Dict, List

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src import utils
from src.datatypes import Rect


def stats_per_event(
    dets_per_frame: Dict[int, pl.DataFrame],
    events: List[Dict[str, Any]],
    white_area: Rect,
) -> pl.DataFrame:
    frames = []
    ids = []
    num_dets = []
    for frame, event in events.items():
        frames.append(frame)
        ids.append(str(event["id"]))

        if frame in dets_per_frame:
            dets = dets_per_frame[frame]
            dets = dets.filter(
                (pl.col("x1") >= white_area.x1)
                & (pl.col("x2") <= white_area.x2)
                & (pl.col("y1") >= white_area.y1),
                (pl.col("y2") <= white_area.y2),
            )
            num_dets.append(len(dets))
        else:
            num_dets.append(0)
    return pl.DataFrame({"frames": frames, "dets": num_dets, "id": ids})


@hydra.main(config_path="config", config_name="stats", version_base=None)
def stats(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    events = utils.read_events(cfg.events)
    dets_per_frame = {
        group[0]: grouped_df
        for group, grouped_df in pl.read_parquet(cfg.detections).group_by("frame")
    }
    white_area = Rect(**cfg.white_area)

    stats = stats_per_event(dets_per_frame, events, white_area)
    stats.write_csv(cfg.dst)


if "__main__" == __name__:
    stats()
