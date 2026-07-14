"""
face_landmarks.py

Uses MediaPipe Face Mesh to:
  1. Locate faces and produce tight bounding boxes (for drawing + emotion cropping)
  2. Compute mouth/lip geometry (mouth-open ratio, smile ratio) per face
     used as auxiliary signal to nudge the emotion classifier.

Now supports multi-face: process() returns List[FaceGeometry].

MediaPipe landmark indices used (468-point face mesh):
  - Mouth corners: 61 (left), 291 (right)
  - Upper inner lip: 13   Lower inner lip: 14
  - Left face edge: 234   Right face edge: 454   (for bbox width)
  - Top of forehead-ish: 10   Bottom of chin: 152  (for bbox height)
"""

import logging
from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


@dataclass
class FaceGeometry:
    bbox: tuple              # (x, y, w, h) in pixel coords
    mouth_open_ratio: float  # 0 (closed) .. ~1+ (wide open)
    smile_ratio: float       # >0 corners raised (smile-like), <0 drooping


class FaceLandmarkDetector:
    """
    Detects up to max_num_faces faces per frame.
    Returns a list of FaceGeometry objects (empty list = no faces found).
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 5,
    ):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "MediaPipe FaceMesh initialised "
            "(max_faces=%d, det_conf=%.2f, trk_conf=%.2f).",
            max_num_faces, min_detection_confidence, min_tracking_confidence,
        )

    def process(self, frame_bgr) -> List[FaceGeometry]:
        """
        Run face mesh on a BGR frame.

        Returns:
            List of FaceGeometry — one per detected face, sorted left-to-right.
            Empty list if no faces found.
        """
        h, w = frame_bgr.shape[:2]

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
        except Exception:
            logger.exception("MediaPipe face mesh processing error.")
            return []

        if not results.multi_face_landmarks:  # type: ignore
            return []

        geometries: List[FaceGeometry] = []

        for face_landmarks in results.multi_face_landmarks:  # type: ignore
            landmarks = face_landmarks.landmark

            def px(idx):
                lm = landmarks[idx]
                return int(lm.x * w), int(lm.y * h)

            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            # Pad bounding box so FER gets a full face with margin
            pad_x = int((x_max - x_min) * 0.15)
            pad_y = int((y_max - y_min) * 0.15)
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(w, x_max + pad_x)
            y_max = min(h, y_max + pad_y)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            # Mouth geometry
            left_corner  = px(61)
            right_corner = px(291)
            upper_lip    = px(13)
            lower_lip    = px(14)

            face_height = max(1, y_max - y_min)
            mouth_open_px = (
                (lower_lip[0] - upper_lip[0]) ** 2
                + (lower_lip[1] - upper_lip[1]) ** 2
            ) ** 0.5
            mouth_open_ratio = mouth_open_px / face_height

            mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2.0
            corner_avg_y   = (left_corner[1] + right_corner[1]) / 2.0
            smile_ratio    = (mouth_center_y - corner_avg_y) / face_height

            geometries.append(FaceGeometry(
                bbox=bbox,
                mouth_open_ratio=mouth_open_ratio,
                smile_ratio=smile_ratio,
            ))

        # Sort faces left-to-right by x-center for stable per-frame ordering
        geometries.sort(key=lambda g: g.bbox[0] + g.bbox[2] // 2)

        return geometries

    def close(self) -> None:
        self._face_mesh.close()
        logger.debug("MediaPipe FaceMesh closed.")
