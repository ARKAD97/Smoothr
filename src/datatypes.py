from dataclasses import dataclass
from numbers import Number
from typing import Tuple

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


@dataclass
class Size2D:
    height: Number
    width: Number

    def as_tuple(self) -> Tuple[Number, Number]:
        return (self.height, self.width)
