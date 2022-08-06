from typing import Any, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint

from .cached_property import cached_property


class AuthModel:
    MEAN: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    STD: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    CLASSES: Tuple[str, str, str, str] = ("fraud", "authentic", "wrappingfoil_authentic", "void")
    CLASS_MAPPING: Dict[int:int] = {3: 0, 2: 1}

    def __init__(self, weights_path: str, image_size: int = 224, device: str = "cpu") -> None:
        self.weights_path = weights_path
        self.image_size = image_size
        self.device = device

    @cached_property
    def model(self) -> Any:
        mmcls_model = build_classifier(
            dict(
                type="ImageClassifier",
                backbone=dict(
                    type="TIMMBackbone",
                    pretrained=True,
                    model_name="mobilenetv3_large_100_miil_in21k",
                    init_cfg=None,
                ),
                neck=dict(type="GlobalAveragePooling"),
                head=dict(
                    type="LinearClsHead",
                    num_classes=4,
                    in_channels=1280,
                    topk=(1,),
                    loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, mode="original"),
                ),
            )
        )
        load_checkpoint(mmcls_model, self.weights_path)
        mmcls_model.eval()
        mmcls_model.to(self.device)
        return mmcls_model

    @cached_property
    def transform(self) -> Any:  # noqa
        return A.Compose(
            transforms=[
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(mean=AuthModel.MEAN, std=AuthModel.STD),
                ToTensorV2(),
            ]
        )

    def preprocess(self, image: np.ndarray) -> Any:
        return self.transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))["image"]

    def predict(self, image: npt.NDArray[Any]) -> Any:
        """
        Parameters
        ----------
        image: np.ndarray, BGR image.

        Returns
        ----------
            class
        """
        with torch.no_grad():
            output = self.model(self.preprocess(image).to(self.device))
        class_id = np.argmax(output)
        class_id = AuthModel.CLASS_MAPPING.get(class_id, class_id)
        return AuthModel.CLASSES[class_id]
