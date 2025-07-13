import abc
from typing import List

import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T

from src.datatypes import Detections, Size2D


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
        threshold: float,
        image_size: Size2D = Size2D(640, 640),
        device: str = "cuda",
    ):
        from deimkit import load_model

        self.image_size = image_size
        self.threshold = threshold
        self.device = device
        self.predictor = load_model(
            model_name=model_name, image_size=image_size.as_hw_tuple(), device=device
        )
        self.predictor.model.eval()
        self.mean = (
            torch.tensor(self.predictor.model_config.mean).view(3, 1, 1).to(device)
        )
        self.std = (
            torch.tensor(self.predictor.model_config.std).view(3, 1, 1).to(device)
        )

    def process(self, images: torch.Tensor) -> List[Detections]:
        batch_size = images.shape[0]

        # Create tensor with original images sizes (requied by DEIMKit package)
        original_h, original_w = images.shape[1], images.shape[2]
        original_sizes = torch.tensor(
            [[original_w, original_h]] * batch_size, device=self.device
        )

        # TODO: If I move this permutation to video_reader, it corrupts detections ???
        images = images.permute(0, 3, 1, 2)  # (B, H, W, C) to (B, C, H, W)
        images = images.float() / 255.0  # [0,255] -> [0,1]

        # preprocess
        images = F.resize(images, self.image_size.as_hw_tuple())
        F.normalize(images, self.mean, self.std, inplace=True)

        # run inference
        batch_size = int(images.shape[0])
        with torch.no_grad():
            labels, boxes, scores = self.predictor.model(images, original_sizes)

        # TODO: Figure out why labels return lists instead of singular values.
        # Currently just pick first item of the list
        detections_list = []
        for batch in range(batch_size):
            mask = scores[batch] > self.threshold
            detections_list.append(
                Detections(
                    boxes=boxes[batch][mask].cpu().numpy(),
                    labels=labels[batch][mask, 0].cpu().numpy(),
                    scores=scores[batch][mask].cpu().numpy(),
                )
            )
        return detections_list


class DEIMDetectorONNX(AbstractDetector):
    pass
