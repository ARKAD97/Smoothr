from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.datatypes import Size2D
from src.models import DEIMDetector


class TestDEIMDetector:
    @pytest.fixture
    def mock_predictor(self) -> Mock:
        predictor = Mock()
        predictor.model = Mock()
        predictor.model.eval = Mock()
        predictor.model_config = Mock()
        predictor.model_config.mean = [0, 0, 0]
        predictor.model_config.std = [1, 1, 1]
        return predictor

    @patch("deimkit.load_model")
    def test_initialization(self, mock_load_model, mock_predictor):
        mock_load_model.return_value = mock_predictor

        detector = DEIMDetector(
            model_name="test_model",
            threshold=0.5,
            image_size=Size2D(640, 640),
            device="cuda",
        )

        assert detector.threshold == 0.5
        assert detector.image_size == Size2D(640, 640)
        assert detector.device == "cuda"
        mock_load_model.assert_called_once_with(
            model_name="test_model", image_size=(640, 640), device="cuda"
        )
        mock_predictor.model.eval.assert_called_once()

    @patch("deimkit.load_model")
    def test_process_batch_images(self, mock_load_model, mock_predictor):
        mock_load_model.return_value = mock_predictor

        batch_size = 2
        mock_predictor.model.return_value = (
            [torch.tensor([[0], [1]]), torch.tensor([[0], [1]])],  # labels
            [
                torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
                torch.tensor([[15, 25, 35, 45], [55, 65, 75, 85]]),
            ],  # boxes
            [torch.tensor([0.9, 0.4]), torch.tensor([0.8, 0.3])],  # scores
        )

        detector = DEIMDetector("test_model", threshold=0.5, device="cpu")

        images = torch.rand(batch_size, 480, 640, 3)

        detections_list = detector.process(images)

        assert len(detections_list) == batch_size

        assert len(detections_list[0].boxes) == 1
        assert len(detections_list[1].boxes) == 1

        assert detections_list[0].scores == np.array([0.9], dtype=np.float32)
        assert detections_list[1].scores == np.array([0.8], dtype=np.float32)

    @patch("deimkit.load_model")
    def test_process_single_image(self, mock_load_model, mock_predictor):
        mock_load_model.return_value = mock_predictor
        mock_predictor.model.return_value = (
            [torch.tensor([[0]])],  # labels
            [torch.tensor([[10, 20, 30, 40]])],  # boxes
            [torch.tensor([0.9])],  # scores
        )

        detector = DEIMDetector("test_model", threshold=0.5, device="cpu")
        images = torch.rand(1, 480, 640, 3)

        detections_list = detector.process(images)

        assert len(detections_list) == 1
        assert len(detections_list[0].boxes) == 1

    @patch("deimkit.load_model")
    def test_process_batch(self, mock_load_model, mock_predictor):
        mock_load_model.return_value = mock_predictor

        mock_predictor.model.return_value = (
            [
                torch.tensor([[0]]),
                torch.tensor([[1]]),
                torch.tensor([[2]]),
            ],
            [
                torch.tensor([[10, 20, 30, 40]]),
                torch.tensor([[50, 60, 70, 80]]),
                torch.tensor([[90, 100, 110, 120]]),
            ],
            [
                torch.tensor([0.9]),
                torch.tensor([0.8]),
                torch.tensor([0.7]),
            ],
        )

        detector = DEIMDetector("test_model", threshold=0.5, device="cpu")
        images = torch.rand(3, 480, 640, 3)

        detections_list = detector.process(images)

        assert len(detections_list) == 3
        assert len(detections_list[0].boxes) == 1
        assert len(detections_list[1].boxes) == 1
        assert len(detections_list[2].boxes) == 1
