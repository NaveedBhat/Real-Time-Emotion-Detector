"""
model_backend.py

Pluggable model backend for emotion classification.
Swap between 'fer' and 'deepface' via config.yaml:

    inference:
      model_backend: fer      # default, no extra install
      model_backend: deepface # run: pip install deepface

Both backends return the same dict format:
  {emotion: probability}  summing to ~1.0
  Emotions: angry, disgust, fear, happy, sad, surprise, neutral
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from .constants import EMOTIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ModelBackend(ABC):
    """Common interface for all emotion classification backends."""

    @abstractmethod
    def analyze(self, face_crop_bgr, geometry) -> Optional[dict]:
        """
        Classify emotions in a face crop.

        Args:
            face_crop_bgr: BGR numpy array of the cropped face region
            geometry:      FaceGeometry from MediaPipe (may be None)

        Returns:
            {emotion: probability} dict summing to ~1.0, or None on failure.
        """

    @classmethod
    def create(cls, backend_name: str, use_mtcnn: bool = False) -> "ModelBackend":
        """Factory — returns the correct backend by name."""
        name = backend_name.lower().strip()
        if name == "fer":
            logger.info("Using FER backend (use_mtcnn=%s).", use_mtcnn)
            return FerBackend(use_mtcnn=use_mtcnn)
        elif name == "deepface":
            logger.info("Using DeepFace backend.")
            return DeepFaceBackend()
        else:
            raise ValueError(
                f"Unknown model backend {backend_name!r}. "
                "Valid choices: 'fer', 'deepface'."
            )


# ---------------------------------------------------------------------------
# FER backend  (default — no extra install)
# ---------------------------------------------------------------------------

class FerBackend(ModelBackend):
    """Wraps the fer==22.5.1 pretrained CNN with the landmark-nudge heuristic."""

    def __init__(self, use_mtcnn: bool = False):
        from fer import FER  # lazy import keeps startup fast when not used
        logger.info("Loading FER model...")
        self._fer = FER(mtcnn=use_mtcnn)
        logger.info("FER model ready.")

    def analyze(self, face_crop_bgr, geometry) -> Optional[dict]:
        log = logger

        if face_crop_bgr is None or face_crop_bgr.size == 0:
            log.warning(
                "analyze() returning None because face_crop_bgr is invalid. "
                "size: %s",
                face_crop_bgr.size if face_crop_bgr is not None else "None",
            )
            return None

        try:
            # 1. Grayscale & Resize
            gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            
            # 2. Preprocess (same as FER.__preprocess_input(v2=True))
            gray = gray.astype("float32") / 255.0
            gray = (gray - 0.5) * 2.0
            
            # 3. Add batch and channel dimensions for Keras 3: (1, 64, 64, 1)
            x = np.expand_dims(gray, axis=0)
            x = np.expand_dims(x, axis=-1)

            # 4. Predict
            # _FER__emotion_classifier is the internal keras model
            preds = self._fer._FER__emotion_classifier(x)  # type: ignore
            if hasattr(preds, "numpy"):
                preds = preds.numpy()
                
            scores_array = preds[0]
            
            # 5. Map to labels
            labels = self._fer._get_labels()
            scores = {labels[i]: round(float(scores_array[i]), 2) for i in range(len(labels))}
            
        except Exception as e:
            log.exception("FER classifier error: %s", e)
            return None

        if geometry is not None:
            scores = self._apply_nudge(scores, geometry)

        return scores

    @staticmethod
    def _apply_nudge(scores: dict, geometry) -> dict:
        """Lightweight heuristic adjustment using mouth/smile geometry."""
        adjusted = dict(scores)

        if geometry.mouth_open_ratio > 0.35:
            adjusted["surprise"] = adjusted.get("surprise", 0) + 0.08
            adjusted["fear"]     = adjusted.get("fear",     0) + 0.04

        if geometry.smile_ratio > 0.02:
            adjusted["happy"] = adjusted.get("happy", 0) + 0.10
        elif geometry.smile_ratio < -0.015:
            adjusted["sad"] = adjusted.get("sad", 0) + 0.06

        total = sum(adjusted.values()) or 1.0
        return {k: v / total for k, v in adjusted.items()}


# ---------------------------------------------------------------------------
# DeepFace backend  (optional — pip install deepface)
# ---------------------------------------------------------------------------

class DeepFaceBackend(ModelBackend):
    """
    Uses DeepFace for higher-accuracy emotion recognition.

    Requires:  pip install deepface
    First run will download model weights (~300 MB).
    """

    def __init__(self):
        try:
            from deepface import DeepFace  # type: ignore
            self._deepface = DeepFace
            logger.info("DeepFace backend ready.")
        except ImportError as exc:
            raise ImportError(
                "DeepFace is not installed.\n"
                "  Run:  pip install deepface\n"
                "  Or set 'inference.model_backend: fer' in config.yaml"
            ) from exc

    def analyze(self, face_crop_bgr, geometry) -> Optional[dict]:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None
        try:
            results = self._deepface.analyze(
                face_crop_bgr,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if not results:
                return None

            raw = results[0]["emotion"]
            # Normalize keys to lowercase and filter to known emotions
            scores = {k.lower(): v for k, v in raw.items() if k.lower() in EMOTIONS}
            total = sum(scores.values()) or 1.0
            return {k: v / total for k, v in scores.items()}
        except Exception:
            logger.exception("DeepFace analysis error.")
            return None
