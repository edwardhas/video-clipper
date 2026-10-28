"""Face detection with OpenCV's YuNet (``cv2.FaceDetectorYN``).

Detects faces only. Face detection is not active-speaker detection: callers must not
describe its output as "the speaker".
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..config import ASSETS_DIR

log = logging.getLogger(__name__)

MODEL_PATH: Path = ASSETS_DIR / "models" / "face_detection_yunet_2023mar.onnx"
DETECTOR_NAME = "yunet-2023mar"
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 50
DETECT_MAX_WIDTH = 640


class FaceBox(NamedTuple):
    """Face box ``[x, y, w, h, score]`` in the pixels of the frame given to ``detect``."""

    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@lru_cache(maxsize=4)
def _can_load(path: str, _mtime_ns: int) -> bool:
    try:
        import cv2

        cv2.FaceDetectorYN.create(path, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
        return True
    except Exception as exc:  # cv2 missing, wrong build or corrupt model
        log.warning("face detector could not be loaded: %s", type(exc).__name__)
        return False


def detector_available(model_path: Path | None = None) -> bool:
    """True when the YuNet model file exists and OpenCV can load it."""
    path = Path(model_path) if model_path is not None else MODEL_PATH
    try:
        if not path.is_file():
            return False
        return _can_load(str(path), path.stat().st_mtime_ns)
    except OSError:
        return False


class FaceDetector:
    """Thin YuNet wrapper. Not thread-safe; create one per analysis."""

    name = DETECTOR_NAME

    def __init__(self, model_path: Path | None = None, *, score_threshold: float = SCORE_THRESHOLD,
                 max_width: int = DETECT_MAX_WIDTH) -> None:
        import cv2

        path = Path(model_path) if model_path is not None else MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError("face detection model is missing")
        self._cv2 = cv2
        self._size = (320, 320)
        self._max_width = max(32, int(max_width))
        self._det = cv2.FaceDetectorYN.create(str(path), "", self._size, float(score_threshold),
                                              NMS_THRESHOLD, TOP_K)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        """Faces in ``frame_bgr`` (H x W x 3 uint8), largest first, in that frame's pixels."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("expected a BGR image")
        h, w = frame_bgr.shape[:2]
        if w < 8 or h < 8:
            return []
        scale = 1.0
        image = frame_bgr
        if w > self._max_width:
            scale = self._max_width / w
            new_size = (self._max_width, max(8, int(round(h * scale))))
            image = self._cv2.resize(frame_bgr, new_size, interpolation=self._cv2.INTER_AREA)
        image = np.ascontiguousarray(image)
        size = (image.shape[1], image.shape[0])
        if size != self._size:
            self._det.setInputSize(size)
            self._size = size
        _n, faces = self._det.detect(image)
        if faces is None:
            return []
        sx = w / image.shape[1]
        sy = h / image.shape[0]
        boxes = [
            FaceBox(float(row[0]) * sx, float(row[1]) * sy, float(row[2]) * sx, float(row[3]) * sy, float(row[14]))
            for row in faces
            if row[2] > 0 and row[3] > 0
        ]
        boxes.sort(key=lambda b: b.area, reverse=True)
        return boxes
