from typing import Any, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from mmcls.models import build_classifier
from mmcls.datasets.builder import build_from_cfg, PIPELINES
from mmcls.datasets.pipelines.compose import Compose
from mmcv.runner import load_checkpoint
import mmcv
from functools import partial

from .cached_property import cached_property


class AuthModel:
    MEAN: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    STD: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    CLASSES: Tuple[str, str, str, str] = ("fraud", "authentic", "wrappingfoil_authentic", "void")
    CLASS_MAPPING: Dict[int, int] = {3: 0, 2: 1}

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
                    pretrained=False,
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
        load_checkpoint(mmcls_model, self.weights_path, strict=True, map_location="cpu")
        mmcls_model.eval()
        mmcls_model.forward = partial(mmcls_model.forward, img_metas={}, return_loss=False)
        mmcls_model.to(self.device)
        return mmcls_model

    @cached_property
    def transform(self) -> Any:  # noqa
        img_norm_cfg = dict(mean=AuthModel.MEAN, std=AuthModel.STD, to_rgb=True)
        albu_val_transforms = [
            dict(type="LongestMaxSize", max_size=self.image_size),
            dict(type="PadIfNeeded", min_height=self.image_size, min_width=self.image_size, border_mode=0, value=list(img_norm_cfg["mean"])[::-1]),
        ]
        test_pipeline = [
            dict(
                type="Albu",
                transforms=albu_val_transforms,
            ),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"])]
        return Compose([PIPELINES.build(p) for p in test_pipeline])

    def preprocess(self, image: np.ndarray) -> Any:
        return self.transform(dict(image=image))['img']

    def predict(self, image: np.ndarray) -> Any:
        """
        Parameters
        ----------
        image: np.ndarray, BGR image.

        Returns
        ----------
            class
        """
        image = self.preprocess(image)
        with torch.no_grad():
            output = self.model(image.unsqueeze(0).to(self.device))
        class_id = np.argmax(output)
        class_id = AuthModel.CLASS_MAPPING.get(class_id, class_id)
        return AuthModel.CLASSES[class_id]
