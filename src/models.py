import abc
from typing import List

import torch
import torchvision.transforms.v2 as T

from src.datatypes import Detections, Size2D

# TODO: Try yolov9


class AbstractDetector(abc.ABC):

    @abc.abstractmethod
    def process(self, input: torch.Tensor) -> Detections:
        """abstract method to predict detections on batch of imags

        Args:
            input (torch.Tensor): batch of images (N, H, W, 3)

        Returns:
            Detections: Detection structure
        """
        ...


class DEIMDetector(AbstractDetector):
    """Implementation of AbstractDetector for DEIM:
    https://github.com/ShihuaHuang95/DEIM
    Some say it beats all YOLO models
    """

    def __init__(
        self,
        model_name: str,
        thershold: float = 0.25,
        image_size: Size2D = Size2D(640, 640),
        device: str = "cuda",
    ):
        from deimkit import load_model

        self.threshold = thershold
        self.device = device
        self.predictor = load_model(
            model_name=model_name, image_size=image_size.as_tuple(), device=device
        )
        import torchvision.transforms.v2 as T

        self.transforms = T.Compose(
            [
                T.Resize(image_size.as_tuple()),
                T.Normalize(
                    mean=self.predictor.model_config.mean,
                    std=self.predictor.model_config.std,
                ),
            ]
        )

    @torch.no_grad()
    def process(self, input: torch.Tensor) -> List[Detections]:
        batch_size, _, height, width = input.shape
        original_sizes = torch.Tensor([height, width])
        original_sizes = original_sizes.repeat(input.shape[0], 1).to(self.device)
        input = self.transforms(input)
        labels, boxes, scores = self.predictor.model(input, original_sizes)
        detections_list = []
        for batch in range(batch_size):

            mask = scores[batch] > self.threshold
            detections_list.append(
                Detections(
                    boxes=boxes[batch][mask].cpu().numpy(),
                    labels=labels[batch][mask].cpu().numpy(),
                    scores=scores[batch][mask].cpu().numpy(),
                )
            )
        return detections_list
