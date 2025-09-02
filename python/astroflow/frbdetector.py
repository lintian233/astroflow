from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numba as nb
import numpy as np
import seaborn
import torch
import time
# override
from typing_extensions import override
from ultralytics import YOLO

from .config.taskconfig import TaskConfig
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
    def __init__(self, dm_limt, preprocess, confidence=0.5):
        self.confidence = confidence
        self.dm_limt = dm_limt  # type: Tuple[float, float]
        self.preprocess = preprocess if preprocess is not None else []

    def check_dm(self, dm):
        if self.dm_limt is None:
            return True
        for item in self.dm_limt:
            dm_low = item["dm_low"]
            dm_high = item["dm_high"]
            dm_name = item["name"]
            if dm_low <= dm <= dm_high:
                return True
        return False

    @abstractmethod
    def mutidetect(self, dmt_list: List[DmTime]):
        pass

    @abstractmethod
    def detect(self, dmt: DmTime):
        pass


class BinaryChecker(ABC):
    @abstractmethod
    def check(self, spec: Spectrum, t_sample) -> List[int]:
        pass


class Yolo11nFrbDetector(FrbDetector):
    def __init__(self, dm_limt=None, preprocess=None, confidence=0.5):
        super().__init__(dm_limt, preprocess, confidence)
        detgpu = TaskConfig().detgpu
        self.device = torch.device(f"cuda:{detgpu}" if torch.cuda.is_available() else "cpu")
        print(
            f"Using device: {self.device} NAME: {torch.cuda.get_device_name(self.device)}"
        )
        self.model = self._load_model()
        self.kernel_2d = None
        self.batch_size = TaskConfig().batchsize

    def _load_model(self):
        model = YOLO(TaskConfig().modelpath)
        return model

    def filter(self, img):
        
        kernel_size = 5
        kernel = cv2.getGaussianKernel(kernel_size, 0)
        self.kernel_2d = np.outer(kernel, kernel.transpose())
        for i in range(1):
            img = cv2.filter2D(img, -1, self.kernel_2d)
        for _ in range(2):
            img = cv2.medianBlur(img.astype(np.float32), ksize=3)
        return img

    def _preprocess(self, img):
        img = np.ascontiguousarray(img, dtype=np.float32)
        min_percentile, max_percentile = np.percentile(img, (1, 99.9))
        np.clip(img, min_percentile, max_percentile, out=img) 
        return img
    
    @override
    def mutidetect(self, dmt_list: List[DmTime]):
        model = self.model
        npy_dmt_list = []
        for dmt in dmt_list:
            # img = self._preprocess(dmt.data)
            img = dmt.data
            npy_dmt_list.append(img)
        
        candidate = []
        total_samples = len(npy_dmt_list)
        
        if total_samples <= self.batch_size:
            # start_time = time.time()
            results = model(
                npy_dmt_list, conf=self.confidence, device=self.device, iou=0.45, stream=True, verbose=False
            )
            # end_time = time.time()
            # detect_time = (end_time - start_time) * 1000
            # print(f"[TIMER] Detect: {detect_time:.3f} ms")
            self._process_results(results, dmt_list, candidate, start_index=0)
        else:

            for i in range(0, total_samples, self.batch_size):
                end_idx = min(i + self.batch_size, total_samples)
                batch_imgs = npy_dmt_list[i:end_idx]
                batch_dmts = dmt_list[i:end_idx]
                # start_time = time.time()
                results = model(
                    batch_imgs, conf=self.confidence, device=self.device, iou=0.1, stream=True
                )
                # end_time = time.time()
                # detect_time = (end_time - start_time) * 1000
                # print(f"[TIMER] Detect batch {i}-{end_idx}: {detect_time:.3f} ms")
                self._process_results(results, batch_dmts, candidate, start_index=i)
                        
        return candidate
    
    def _process_results(self, results, dmt_list, candidate, start_index=0):
        for i, r in enumerate(results):
            xywh = r.boxes.xywh
            if xywh is not None and len(xywh) > 0:
                dmt = dmt_list[i]
                dmt_index = start_index + i  
                for box in xywh:
                    x, y, w, h = box
                    # 直接使用x, y坐标，无需计算left, top, right, bottom
                    x_norm = np.round(x.cpu() / 512, 2)
                    y_norm = np.round(y.cpu() / 512, 2)
                    w_norm = np.round(w.cpu() / 512, 2)
                    h_norm = np.round(h.cpu() / 512, 2)

                    t_len = dmt.tend - dmt.tstart

                    dm = y.cpu() * (dmt.dm_high - dmt.dm_low) / 512 + dmt.dm_low
                    dm_flag = self.check_dm(dm)

                    toa = x.cpu() * (t_len / 512) + dmt.tstart
                    toa = np.round(toa.item(), 3)
                    dm = np.round(dm.item(), 3)
                    # print(f"DM: {dm}, TOA: {toa}")
                    if dm_flag:
                        candidate.append([dm, toa, dmt.freq_start, dmt.freq_end, dmt_index, (x_norm, y_norm, w_norm, h_norm)])
    
    @override
    def detect(self, dmt: DmTime):
        model = self.model
        data = dmt.data
        img = self._preprocess(data)
        candidate = []
        results = model(img, conf=self.confidence, device=self.device, iou=0.45)

        for i, r in enumerate(results):
            xywh = r.boxes.xywh
            if xywh is not None and len(xywh) > 0:
                for box in xywh:
                    x, y, w, h = box
                    # 直接使用x, y坐标，无需计算left, top, right, bottom
                    t_len = dmt.tend - dmt.tstart
                    # 直接使用y坐标计算dm
                    dm = y * (dmt.dm_high - dmt.dm_low) / 512 + dmt.dm_low
                    dm_flag = self.check_dm(dm)

                    # 直接使用x坐标计算toa
                    toa = x * (t_len / 512) + dmt.tstart
                    toa = np.round(toa, 3)
                    dm = np.round(dm, 3)
                    if dm_flag:
                        candidate.append([dm, toa, dmt.freq_start, dmt.freq_end])
        return candidate


class ResNetBinaryChecker(BinaryChecker):
    def __init__(self, confidence=0.5):
        self.confidence = confidence
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

                frb_indices = np.where(pred_probs > self.confidence)[0]
                if frb_indices.size > 0:
                    total_pred.extend((i + frb_indices).tolist())

        # Log and return results
        if total_pred:
            print(f"Found FRBs at indices: {total_pred}")
            return total_pred
        return []


class CenterNetFrbDetector(FrbDetector):
    def __init__(self, dm_limt=None, preprocess=None, confidence=0.5):
        super().__init__(dm_limt, preprocess, confidence)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(
            f"Using device: {self.device} NAME: {torch.cuda.get_device_name(self.device)}"
        )
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
        lo, hi = np.percentile(dmt, (5, 99))
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
    def mutidetect(self, dmt_list: List[DmTime]):
        pass

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
                dm_flag = self.check_dm(dm)
                toa = ((left + right) / 2) * (t_len / 512) + dmt.tstart
                toa = np.round(toa, 3)
                dm = np.round(dm, 3)
                if dm_flag:
                    print(f"Confidence: {np.min(top_conf):.3f}")
                    result.append([dm, toa, dmt.freq_start, dmt.freq_end])

        return result
