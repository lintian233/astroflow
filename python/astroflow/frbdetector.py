from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numba as nb
import numpy as np
import seaborn
import torch

# override
from typing_extensions import override

from .dmtime import DmTime
from .model.binnet import BinaryNet
from .model.centernet import centernet
from .model.centernetutils import get_res
from .spectrum import Spectrum


@nb.njit(nb.float32[:, :](nb.uint64[:, :]), parallel=True, cache=True)
def nb_convert(src):
    dst = np.empty(src.shape, dtype=np.float32)
    rows, cols = src.shape
    for i in nb.prange(rows):
        for j in range(cols):
            dst[i, j] = src[i, j]  # 隐式类型转换
    return dst


class FrbDetector(ABC):
    @abstractmethod
    def detect(self, dmt: DmTime):
        pass


class BinaryChecker(ABC):
    @abstractmethod
    def check(self, spec: Spectrum, t_sample) -> List[int]:
        pass


class ResNetBinaryChecker(BinaryChecker):
    def __init__(self, confidence=0.5):
        self.confidence = confidence
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        base_model = "resnet50"
        model = BinaryNet(base_model, num_classes=2).to(self.device)
        model.load_state_dict(
            torch.load(
                f"class_{base_model}.pth",
                map_location=self.device,
            )
        )
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, spec: np.ndarray, exp_cut=1):
        spec = np.ascontiguousarray(spec, dtype=np.float32)
        spec = cv2.resize(spec, (256, 256), interpolation=cv2.INTER_LINEAR)
        spec /= np.mean(spec, axis=0)
        vmin, max = np.percentile(spec, (exp_cut, 100 - exp_cut))
        spec = np.clip(spec, vmin, max)
        spec = cv2.normalize(spec, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # type: ignore
        # spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))

        return spec

    @override
    def check(self, spec: Spectrum, t_sample) -> List[int]:  # type: ignore
        # Clip the spectrum to the specified time sample
        spec_origin = spec.clip(t_sample)
        num_samples = len(spec_origin)

        # Preprocess all samples in one go
        spec_processed = np.zeros((num_samples, 256, 256), dtype=np.float32)
        for i in range(num_samples):
            spec_processed[i] = self._preprocess(spec_origin[i])

        # Batch processing
        batch_size = 120
        total_pred = []
        for i in range(0, num_samples, batch_size):
            # Prepare batch
            batch = spec_processed[i : i + batch_size]
            batch_tensor = (
                torch.from_numpy(batch[:, np.newaxis, :, :]).float().to(self.device)
            )

            # Predict
            with torch.no_grad():
                pred = self.model(batch_tensor)
                pred_probs = pred.softmax(dim=1)[:, 1]
                pred_probs = pred_probs.cpu().numpy()

                # Filter predictions above confidence threshold
                frb_indices = np.where(pred_probs > self.confidence)[0]
                if frb_indices.size > 0:
                    total_pred.extend((i + frb_indices).tolist())

        # Log and return results
        if total_pred:
            print(f"Found FRBs at indices: {total_pred}")
            return total_pred
        return []


class CenterNetFrbDetector(FrbDetector):
    def __init__(self, confidence=0.35):
        self.confidence = confidence
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} NAME: {torch.cuda.get_device_name(self.device)}")
        self.model = self._load_model()

    def _load_model(self):
        base_model = "resnet50"
        model = centernet(model_name=base_model)
        model.load_state_dict(
            torch.load("cent_{}.pth".format(base_model), map_location=self.device)
        )
        model.to(self.device)
        model.eval()

        kernel_size = 5
        kernel = cv2.getGaussianKernel(kernel_size, 0)
        self.kernel_2d = np.outer(kernel, kernel.transpose())
        return model

    def _filter(self, img):
        
        for _ in range(2):
            img = cv2.filter2D(img, -1, self.kernel_2d)
        # for _ in range(2):
        #     img = cv2.medianBlur(img.astype(np.float32), ksize=5)
        
        return img

    def _preprocess_dmt(self, dmt):
        dmt = np.ascontiguousarray(dmt, dtype=np.float32)
        dmt = self._filter(dmt)
        dmt = cv2.resize(dmt, (512, 512), interpolation=cv2.INTER_LINEAR)
        lo, hi = np.percentile(dmt, (5, 100))
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
        img = (
            torch.from_numpy(pdmt).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        )
        result = []
        position = []
        with torch.no_grad():
            hm, wh, offset = model(img)
            offset = offset.to(device)
            top_conf, top_boxes = get_res(hm, wh, offset, confidence=self.confidence)
            if top_boxes is None:
                return result
            for box in top_boxes:  # box: [left, top, right, bottom] #type: ignore
                left, top, right, bottom = box.astype(int)
                t_len = dmt.tend - dmt.tstart
                dm = ((top + bottom) / 2) * (
                    (dmt.dm_high - dmt.dm_low) / 512
                ) + dmt.dm_low
                dm_flag = (dm <= 57 and dm >= 56)
                toa = ((left + right) / 2) * (t_len / 512) + dmt.tstart
                toa = np.round(toa, 3)
                dm = np.round(dm, 3)
                if dm_flag:
                    print(f"Confidence: {np.min(top_conf):.3f}")
                    result.append((dm, toa, dmt.freq_start, dmt.freq_end))
        return result
