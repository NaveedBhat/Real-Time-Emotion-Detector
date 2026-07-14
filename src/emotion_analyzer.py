"""
emotion_analyzer.py

Wraps the FER (Facial Expression Recognition) pretrained model and blends
its output with the simple mouth-geometry signal from MediaPipe to get a
slightly more stable/opinionated final prediction.

FER returns a dict of emotion -> probability for these 7 classes:
  angry, disgust, fear, happy, sad, surprise, neutral
"""

import logging
from typing import Optional

from fer import FER

from .constants import EMOTIONS
from .face_landmarks import FaceGeometry

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    def __init__(self, use_mtcnn: bool = False):
        # mtcnn=False uses OpenCV's Haar cascade internally for FER's own
        # face detection step, which is faster (we already have a face
        # region from MediaPipe, so FER's detector just needs to confirm it).
        logger.info("Loading FER model (use_mtcnn=%s)...", use_mtcnn)
        self._detector = FER(mtcnn=use_mtcnn)
        logger.info("FER model loaded.")

    def analyze(self, face_crop_bgr, geometry: Optional[FaceGeometry]) -> Optional[dict]:
        """
        face_crop_bgr: cropped face region (BGR numpy array)
        geometry: optional FaceGeometry from MediaPipe for the same face

        Returns a dict of {emotion: probability} summing to ~1.0, or None
        if FER couldn't find a face in the crop.
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            logger.debug("Skipping analyze: empty face crop.")
            return None

        try:
            results = self._detector.detect_emotions(face_crop_bgr)
        except Exception:
            logger.exception("FER detect_emotions raised an exception.")
            return None

        if not results:
            logger.debug(
                "FER found no face in crop (shape=%s).", face_crop_bgr.shape[:2]
            )
            return None

        scores = dict(results[0]["emotions"])  # copy

        if geometry is not None:
            scores = self._apply_landmark_nudge(scores, geometry)

        return scores

    @staticmethod
    def _apply_landmark_nudge(scores: dict, geometry: FaceGeometry) -> dict:
        """
        Lightweight heuristic adjustment using mouth geometry:
          - Wide open mouth -> nudge 'surprise' and 'fear' up slightly
          - Raised mouth corners (smile-like) -> nudge 'happy' up slightly
          - Drooping corners -> nudge 'sad' up slightly
        The nudges are small so FER's model output remains dominant; this
        just helps resolve close calls using a second, independent signal.
        """
        adjusted = dict(scores)

        if geometry.mouth_open_ratio > 0.35:
            adjusted["surprise"] = adjusted.get("surprise", 0) + 0.08
            adjusted["fear"]     = adjusted.get("fear",     0) + 0.04

        if geometry.smile_ratio > 0.02:
            adjusted["happy"] = adjusted.get("happy", 0) + 0.10
        elif geometry.smile_ratio < -0.015:
            adjusted["sad"] = adjusted.get("sad", 0) + 0.06

        # Re-normalize so probabilities still sum to ~1.0
        total = sum(adjusted.values()) or 1.0
        return {k: v / total for k, v in adjusted.items()}
