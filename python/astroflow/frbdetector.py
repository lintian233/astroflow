from typing import override
import torch
import numpy as np
import cv2
import seaborn

from abc import ABC, abstractmethod

from .model.centernet import centernet
from .model.centernetutils import get_res
from .dmtime import DmTime


class FrbDetector(ABC):

    @abstractmethod
    def detect(self, dmt: DmTime) -> bool:
        pass


class CenterNetFrbDetector(FrbDetector):
    def __init__(self, confidence=0.35):
        self.confidence = confidence
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        base_model = "resnet50"
        model = centernet(model_name=base_model)
        model.load_state_dict(torch.load("cent_{}.pth".format(base_model)))
        model.eval()
        return model

    def _preprocess_dmt(self, dmt):
        dmt = (dmt - np.min(dmt)) / (np.max(dmt) - np.min(dmt))
        dmt = (dmt - np.mean(dmt)) / np.std(dmt)
        dmt = cv2.resize(dmt, (512, 512))

        dmt = np.clip(dmt, *np.percentile(dmt, (0.1, 99.9)))
        dmt = (dmt - np.min(dmt)) / (np.max(dmt) - np.min(dmt))

        dmt = seaborn.color_palette("mako", as_cmap=True)(dmt)
        dmt = dmt[..., :3]

        dmt -= [0.485, 0.456, 0.406]
        dmt /= [0.229, 0.224, 0.225]

        return dmt

    @override
    def detect(self, dmt: DmTime):
        model = self.model
        device = self.device
        pdmt = self._preprocess_dmt(dmt.data)
        img = torch.from_numpy(pdmt).permute(2, 0, 1).float().unsqueeze(0)
        with torch.no_grad():
            hm, wh, offset = model(img)
            hm = hm.to(device)
            wh = wh.to(device)
            offset = offset.to(device)
            top_conf, top_boxes = get_res(hm, wh, offset, confidence=self.confidence)
            if top_boxes is not None:
                return True

        return False
