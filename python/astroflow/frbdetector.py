from typing import override
import torch
import numpy as np
import cv2
import seaborn

from abc import ABC, abstractmethod
from typing import List, Tuple

from .model.centernet import centernet
from .model.centernetutils import get_res
from .dmtime import DmTime


class FrbDetector(ABC):

    @abstractmethod
    def detect(self, dmt: DmTime) -> List[Tuple[float, float, float, float]]:
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
        dmt = dmt.astype(np.float32)
        mean_val, std_val = cv2.meanStdDev(dmt)
        mean_val = float(mean_val[0,0])
        std_val = float(std_val[0,0])
        if std_val:
            # MinMax normalize
            dmt = cv2.normalize(dmt, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        dmt = cv2.resize(dmt, (512, 512), interpolation=cv2.INTER_LINEAR)

        lo, hi = np.percentile(dmt, (0.1, 99.5))
        np.clip(dmt, lo, hi, out=dmt)

        dmt = cv2.normalize(dmt, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if not hasattr(self, "_mako_cmap"):
            self._mako_cmap = seaborn.color_palette("mako", as_cmap=True)

        dmt = self._mako_cmap(dmt)[..., :3]

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        dmt = dmt.astype(np.float32)
        dmt = cv2.subtract(dmt, mean)
        dmt = cv2.divide(dmt, std)
        return dmt

    @override
    def detect(self, dmt: DmTime):
        model = self.model
        device = self.device
        pdmt = self._preprocess_dmt(dmt.data)
        img = torch.from_numpy(pdmt).permute(2, 0, 1).float().unsqueeze(0)
        result = []
        with torch.no_grad():
            hm, wh, offset = model(img)
            hm = hm.to(device)
            wh = wh.to(device)
            offset = offset.to(device)
            top_conf, top_boxes = get_res(hm, wh, offset, confidence=self.confidence)
            if top_boxes is None:
                return result
            for box in top_boxes:  # box: [left, top, right, bottom] #type: ignore
                left, top, right, bottom = box.astype(int)
                print(f"left: {left}, top: {top}, right: {right}, bottom: {bottom}")
                t_len = dmt.tend - dmt.tstart
                dm = ((top + bottom) / 2) * ((dmt.dm_high - dmt.dm_low) / 512) + dmt.dm_low
                dm_flag = dm >= 15
                toa = ((left + right) / 2) * (t_len / 512) + dmt.tstart
                toa = np.round(toa, 3)
                dm = np.round(dm, 3)
                if dm_flag:
                    result.append((dm, toa, dmt.freq_start, dmt.freq_end))
        return result
