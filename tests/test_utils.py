import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
import torch

from src.datatypes import Size2D
from src.utils import (
    BatchedVideoReader,
    VideoConfig,
    VideoReaderCV,
    VideoWriterCV,
    read_events,
)


class TestVideoReaderCV:

    @pytest.fixture
    def mock_video_capture(self) -> Mock:
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
        }.get(prop, 0)
        return mock_cap

    @patch("cv2.VideoCapture")
    def test_init_successful(self, mock_cv2_capture, mock_video_capture):
        mock_cv2_capture.return_value = mock_video_capture

        reader = VideoReaderCV("test_video.mp4")

        assert reader.idx == -1
        mock_cv2_capture.assert_called_once_with("test_video.mp4")

    @patch("cv2.VideoCapture")
    def test_init_failed(self, mock_cv2_capture):
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap

        with patch("loguru.logger.error") as mock_logger:
            VideoReaderCV("invalid_video.mp4")
            mock_logger.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_iterator(self, mock_cv2_capture, mock_video_capture):
        test_frames = [
            (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
            (True, np.ones((720, 1280, 3), dtype=np.uint8)),
            (False, None),
        ]
        mock_video_capture.read.side_effect = test_frames
        mock_cv2_capture.return_value = mock_video_capture

        reader = VideoReaderCV("test_video.mp4")
        frames = list(reader)

        assert len(frames) == 2
        assert frames[0][0] == 0
        assert frames[1][0] == 1
        assert np.array_equal(frames[0][1], test_frames[0][1])
        assert np.array_equal(frames[1][1], test_frames[1][1])

    @patch("cv2.VideoCapture")
    def test_fps_property(self, mock_cv2_capture, mock_video_capture):
        mock_cv2_capture.return_value = mock_video_capture

        reader = VideoReaderCV("test_video.mp4")

        assert reader.fps == 30

    @patch("cv2.VideoCapture")
    def test_frame_size_property(self, mock_cv2_capture, mock_video_capture):
        mock_cv2_capture.return_value = mock_video_capture

        reader = VideoReaderCV("test_video.mp4")
        frame_size = reader.frame_size

        assert isinstance(frame_size, Size2D)
        assert frame_size.height == 720
        assert frame_size.width == 1280

    @patch("cv2.VideoCapture")
    def test_destructor(self, mock_cv2_capture, mock_video_capture):
        mock_cv2_capture.return_value = mock_video_capture

        reader = VideoReaderCV("test_video.mp4")
        del reader

        mock_video_capture.release.assert_called_once()


class TestVideoWriterCV:

    @patch("cv2.VideoWriter")
    def test_init(self, mock_cv2_writer):
        mock_writer = Mock()
        mock_cv2_writer.return_value = mock_writer

        frame_size = Size2D(height=720, width=1280)
        VideoWriterCV("output.mp4", 30.0, frame_size)

        mock_cv2_writer.assert_called_once_with(
            "output.mp4",
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=30.0,
            frameSize=(1280, 720),
        )

    @patch("cv2.VideoWriter")
    def test_write(self, mock_cv2_writer):
        mock_writer = Mock()
        mock_cv2_writer.return_value = mock_writer

        frame_size = Size2D(height=720, width=1280)
        writer = VideoWriterCV("output.mp4", 30.0, frame_size)

        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        writer.write(test_frame)

        mock_writer.write.assert_called_once()

    @patch("cv2.VideoWriter")
    def test_destructor(self, mock_cv2_writer):
        mock_writer = Mock()
        mock_cv2_writer.return_value = mock_writer

        frame_size = Size2D(height=720, width=1280)
        writer = VideoWriterCV("output.mp4", 30.0, frame_size)
        del writer

        mock_writer.release.assert_called_once()


class TestVideoConfig:

    def test_default_values(self):
        config = VideoConfig()

        assert config.start == 0
        assert config.end is None
        assert config.batch_size == 1
        assert config.skip == 0

    def test_custom_values(self):
        config = VideoConfig(start=10, end=100, batch_size=4, skip=2)

        assert config.start == 10
        assert config.end == 100
        assert config.batch_size == 4
        assert config.skip == 2


class TestBatchedVideoReader:

    @pytest.fixture
    def mock_decord(self):
        with patch("src.utils.decord") as mock:
            mock_video_reader = Mock()
            type(mock_video_reader).__len__ = MagicMock(return_value=100)
            mock_video_reader.get_batch.return_value = Mock(
                to=lambda device: Mock(to=lambda dtype: torch.zeros((4, 720, 1280, 3)))
            )
            mock.VideoReader.return_value = mock_video_reader
            mock.cpu.return_value = Mock()
            yield mock, mock_video_reader

    def test_init_default_end(self, mock_decord):
        config = VideoConfig(start=0, batch_size=4, skip=1)
        reader = BatchedVideoReader("test_video.mp4", "cpu", config)

        assert reader.end == 100
        assert reader.start_indices == list(range(0, 100, 8))

    def test_init_custom_end(self, mock_decord):
        config = VideoConfig(start=0, end=50, batch_size=4, skip=1)
        reader = BatchedVideoReader("test_video.mp4", "cpu", config)

        assert reader.end == 50
        assert reader.start_indices == list(range(0, 50, 8))

    def test_frame_number(self, mock_decord):
        config = VideoConfig()
        reader = BatchedVideoReader("test_video.mp4", "cpu", config)

        assert reader.frame_number() == 100

    def test_iterator(self, mock_decord):
        config = VideoConfig(start=0, end=20, batch_size=4, skip=1)
        reader = BatchedVideoReader("test_video.mp4", "cpu", config)

        batches = list(reader)

        assert len(batches) == 3

        indices, frames = batches[0]
        assert indices == [0, 2, 4, 6]
        assert frames.shape == torch.Size([4, 720, 1280, 3])

        indices, frames = batches[-1]
        assert indices == [16, 18]


class TestReadEvents:

    def test_valid_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="txt") as f:
            f.write('{"frame": 10, "type": "event1", "data": "value1"}\n')
            f.write('{"frame": 20, "type": "event2", "data": "value2"}\n')
            f.write('{"frame": 30, "type": "event3", "data": "value3"}\n')
            temp_path = f.name

        try:
            events = read_events(temp_path)

            assert len(events) == 3
            assert 10 in events
            assert 20 in events
            assert 30 in events

            assert events[10]["type"] == "event1"
            assert events[20]["type"] == "event2"
            assert events[30]["type"] == "event3"
        finally:
            Path(temp_path).unlink()

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            events = read_events(temp_path)
            assert events == {}
        finally:
            Path(temp_path).unlink()

    def test_malformed_data(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write('{"frame": 10, "type": "event1"}\n')
            f.write("invalid data\n")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                read_events(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_events("non_existent_file.txt")
