"""
test_smoothing.py

Unit tests for EmotionSmoother — the rolling-average prediction stabiliser.
"""

import pytest
from src.smoothing import EmotionSmoother


class TestEmotionSmoother:

    def test_empty_smoother_returns_none_on_none_input(self):
        s = EmotionSmoother()
        assert s.update(None) is None

    def test_single_frame_scores_pass_through(self):
        s = EmotionSmoother()
        scores = {"happy": 0.8, "sad": 0.1, "neutral": 0.1}
        result = s.update(scores)
        assert result["happy"]   == pytest.approx(0.8)
        assert result["sad"]     == pytest.approx(0.1)
        assert result["neutral"] == pytest.approx(0.1)

    def test_rolling_average_across_two_frames(self):
        s = EmotionSmoother(window_size=2)
        s.update({"happy": 1.0, "sad": 0.0})
        result = s.update({"happy": 0.0, "sad": 1.0})
        assert result["happy"] == pytest.approx(0.5)
        assert result["sad"]   == pytest.approx(0.5)

    def test_window_size_limits_history(self):
        """With window_size=2, only the last 2 frames contribute."""
        s = EmotionSmoother(window_size=2)
        s.update({"happy": 1.0})      # frame 1 — will be evicted
        s.update({"sad":   1.0})      # frame 2
        result = s.update({"angry":  1.0})  # frame 3 — frame 1 evicted
        # Only frames 2 and 3 remain → sad=0.5, angry=0.5, happy=0 (evicted)
        assert result.get("happy", 0.0) == pytest.approx(0.0)
        assert result["sad"]   == pytest.approx(0.5)
        assert result["angry"] == pytest.approx(0.5)

    def test_none_input_does_not_add_to_history(self):
        """None frames are silently ignored; existing window is returned."""
        s = EmotionSmoother(window_size=4)
        s.update({"happy": 1.0})
        result = s.update(None)         # should not clear or add
        assert result is not None
        assert result["happy"] == pytest.approx(1.0)

    def test_reset_clears_history(self):
        s = EmotionSmoother()
        s.update({"happy": 1.0})
        s.reset()
        assert s.update(None) is None

    def test_reset_then_new_scores(self):
        s = EmotionSmoother()
        s.update({"happy": 1.0})
        s.reset()
        result = s.update({"sad": 0.9, "neutral": 0.1})
        assert result["sad"] == pytest.approx(0.9)

    def test_top_emotion_returns_highest(self):
        scores = {"happy": 0.6, "sad": 0.2, "angry": 0.2}
        label, conf = EmotionSmoother.top_emotion(scores)
        assert label == "happy"
        assert conf  == pytest.approx(0.6)

    def test_top_emotion_on_none_returns_none_zero(self):
        label, conf = EmotionSmoother.top_emotion(None)
        assert label is None
        assert conf  == 0.0

    def test_top_emotion_on_empty_dict(self):
        label, conf = EmotionSmoother.top_emotion({})
        assert label is None
        assert conf  == 0.0
