from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import polars as pl
import torch
from loguru import logger

from src.datatypes import Detections, Size2D


class VideoReaderCV:
    def __init__(self, path: str) -> None:
        self.cap = cv2.VideoCapture(path)
        self.idx = -1
        if not self.cap.isOpened():
            logger.error(f"Cannot open video: {path}")
            return

    def __iter__(self) -> VideoReaderCV:
        return self

    def __next__(self) -> cv2.Mat:
        ret, frame = self.cap.read()
        self.idx += 1
        if not ret:
            raise StopIteration
        return self.idx, frame

    @property
    def fps(self) -> float:
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def frame_size(self) -> Size2D:
        return Size2D(
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )

    def __del__(self) -> None:
        self.cap.release()


class VideoWriterCV:
    def __init__(self, path: str, fps: float, frame_size: Size2D) -> None:
        self.writer = cv2.VideoWriter(
            path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(frame_size.width, frame_size.height),
        )

    def write(self, frame: cv2.Mat) -> None:
        self.writer.write(frame)

    def __del__(self) -> None:
        self.writer.release()


@dataclass
class VideoConfig:
    start: int = 0
    end: Optional[int] = None
    batch_size: int = 1
    skip: int = 0


class BatchedVideoReader:

    def __init__(self, path: str, device: str, cfg: VideoConfig) -> None:
        import decord

        # TODO: build from sources to enable gpu acceleratoion https://github.com/dmlc/decord
        self.video_reader = decord.VideoReader(path, ctx=decord.cpu(0))
        self.device = device
        self.cfg = cfg
        self.end = len(self.video_reader)
        self.end = min(cfg.end, self.end) if cfg.end else self.end

        # indices take into the account batch sizes and skips
        self.start_indices = list(
            range(self.cfg.start, self.end, self.cfg.batch_size * (self.cfg.skip + 1))
        )
        self.idx = -1

    def frame_number(self) -> int:
        return len(self.video_reader)

    def __iter__(self) -> BatchedVideoReader:
        self.idx = -1
        return self

    def __next__(self) -> Tuple[int, torch.Tensor]:
        self.idx += 1
        if self.idx >= len(self.start_indices):
            raise StopIteration

        # generate indices
        start_idx = self.start_indices[self.idx]
        frames_per_step = self.cfg.skip + 1
        end_idx = min(start_idx + self.cfg.batch_size * frames_per_step, self.end)
        indices = list(range(start_idx, end_idx, frames_per_step))

        return indices, self.video_reader.get_batch(indices).to(self.device).to(
            torch.float32
        )


def read_events(path: str) -> Dict[int, Dict]:
    events = Path(path).read_text().splitlines()
    events = list(map(ast.literal_eval, events))
    return {event["frame"]: event for event in events}
