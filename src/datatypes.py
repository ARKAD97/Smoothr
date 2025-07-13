from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Tuple

import numpy as np
import polars as pl


@dataclass
class Detections:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray

    @staticmethod
    def from_polars(df: pl.DataFrame, round: bool = False) -> Detections:
        box_cols = pl.col(["x1", "y1", "x2", "y2"])
        if round:
            box_cols = box_cols.round().cast(pl.Int32)
        return Detections(
            boxes=df.select(box_cols).to_numpy(),
            labels=df["label"].to_numpy(),
            scores=df["score"].to_numpy(),
        )

    def __len__(self) -> int:
        return self.scores.shape[0]

    @property
    def x1(self) -> np.ndarray:
        return self.boxes[:, 0]

    @property
    def y1(self) -> np.ndarray:
        return self.boxes[:, 1]

    @property
    def x2(self) -> np.ndarray:
        return self.boxes[:, 2]

    @property
    def y2(self) -> np.ndarray:
        return self.boxes[:, 3]

    @property
    def width(self) -> np.ndarray:
        return self.x2 - self.x1

    @property
    def height(self) -> np.ndarray:
        return self.y2 - self.y1


@dataclass
class Size2D:
    height: Number
    width: Number

    def as_hw_tuple(self) -> Tuple[Number, Number]:
        return (self.height, self.width)

    def as_wh_tuple(self) -> Tuple[Number, Number]:
        return (self.width, self.height)


@dataclass
class Rect:
    x1: Number
    y1: Number
    x2: Number
    y2: Number
