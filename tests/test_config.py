"""
test_config.py

Unit tests for AppConfig — YAML loading, defaults, and CLI-override logic.
"""

import os
import tempfile

import pytest
from src.config import AppConfig, CameraConfig, InferenceConfig, DisplayConfig


class TestAppConfigDefaults:

    def test_default_camera_index(self):
        cfg = AppConfig()
        assert cfg.camera.index == 0

    def test_default_resolution(self):
        cfg = AppConfig()
        assert cfg.camera.width  == 1280
        assert cfg.camera.height == 720

    def test_default_skip_frames(self):
        cfg = AppConfig()
        assert cfg.inference.skip_frames == 2

    def test_default_smoothing_window(self):
        cfg = AppConfig()
        assert cfg.inference.smoothing_window == 8

    def test_default_show_bars(self):
        cfg = AppConfig()
        assert cfg.display.show_probability_bars is True

    def test_default_show_history(self):
        cfg = AppConfig()
        assert cfg.display.show_emotion_history is True


class TestAppConfigFromYaml:

    def _write_yaml(self, content: str) -> str:
        """Write a temporary YAML file and return its path."""
        fh = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        fh.write(content)
        fh.flush()
        return fh.name

    def test_loads_camera_section(self):
        path = self._write_yaml("""
camera:
  index: 2
  width: 640
  height: 480
""")
        try:
            cfg = AppConfig.from_yaml(path)
            assert cfg.camera.index  == 2
            assert cfg.camera.width  == 640
            assert cfg.camera.height == 480
        finally:
            os.unlink(path)

    def test_loads_inference_section(self):
        path = self._write_yaml("""
inference:
  skip_frames: 5
  smoothing_window: 12
""")
        try:
            cfg = AppConfig.from_yaml(path)
            assert cfg.inference.skip_frames     == 5
            assert cfg.inference.smoothing_window == 12
        finally:
            os.unlink(path)

    def test_missing_key_falls_back_to_default(self):
        """A YAML with only one section should not break other sections."""
        path = self._write_yaml("""
camera:
  index: 1
""")
        try:
            cfg = AppConfig.from_yaml(path)
            assert cfg.camera.index           == 1
            assert cfg.inference.skip_frames  == 2   # default
            assert cfg.camera.width           == 1280  # default
        finally:
            os.unlink(path)

    def test_nonexistent_file_returns_defaults(self):
        cfg = AppConfig.from_yaml("/tmp/__nonexistent_config_xyz__.yaml")
        assert cfg.camera.index == 0

    def test_empty_yaml_returns_defaults(self):
        path = self._write_yaml("")
        try:
            cfg = AppConfig.from_yaml(path)
            assert cfg.camera.index == 0
        finally:
            os.unlink(path)

    def test_loads_display_section(self):
        path = self._write_yaml("""
display:
  show_probability_bars: false
  history_length: 30
""")
        try:
            cfg = AppConfig.from_yaml(path)
            assert cfg.display.show_probability_bars is False
            assert cfg.display.history_length        == 30
        finally:
            os.unlink(path)
