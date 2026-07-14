"""
smoothing.py

Averages emotion probability distributions over the last N frames so the
displayed label doesn't flicker rapidly between frames.
"""

from collections import deque
from typing import Optional


class EmotionSmoother:
    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self._history = deque(maxlen=window_size)

    def update(self, scores: Optional[dict]) -> Optional[dict]:
        """
        Push the latest frame's emotion scores (or None if no face this frame)
        and return the smoothed distribution based on the current window.
        """
        if scores is not None:
            self._history.append(scores)

        if not self._history:
            return None

        totals = {}
        for frame_scores in self._history:
            for emotion, prob in frame_scores.items():
                totals[emotion] = totals.get(emotion, 0.0) + prob

        n = len(self._history)
        return {emotion: total / n for emotion, total in totals.items()}

    def reset(self):
        self._history.clear()

    @staticmethod
    def top_emotion(scores: Optional[dict]):
        """Return (label, confidence) for the highest scoring emotion, or (None, 0.0)."""
        if not scores:
            return None, 0.0
        label = max(scores, key=scores.get)
        return label, scores[label]
