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
from mmcv import Config, DictAction

from .cached_property import cached_property


class AuthModel:
    CLASSES: Tuple[str, str, str, str] = ("fraud", "authentic", "wrappingfoil_authentic", "void", "misc")
    CLASS_MAPPING: Dict[int, int] = {3: 0, 2: 1, 4: 0}

    def __init__(self, config_path: str, weights_path: str, device: str = "cpu") -> None:
        self.weights_path = weights_path
        self.device = device
        self.cfg = Config.fromfile(config_path)

    @cached_property
    def model(self) -> Any:
        mmcls_model = build_classifier(
            self.cfg.model
        )
        load_checkpoint(mmcls_model, self.weights_path, strict=True, map_location="cpu")
        mmcls_model.eval()
        mmcls_model.forward = partial(mmcls_model.forward, img_metas={}, return_loss=False)
        mmcls_model.to(self.device)
        return mmcls_model

    @cached_property
    def transform(self) -> Any:  # noqa

        return Compose([PIPELINES.build(p) for p in self.cfg.test_pipeline])

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
