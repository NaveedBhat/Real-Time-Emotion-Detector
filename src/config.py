"""
config.py

Typed configuration dataclasses loaded from config.yaml.
CLI arguments in main.py / server.py override any value here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720


@dataclass
class InferenceConfig:
    skip_frames: int = 2
    smoothing_window: int = 8
    use_mtcnn: bool = False
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_faces: int = 5
    model_backend: str = "fer"      # "fer" | "deepface"


@dataclass
class DisplayConfig:
    window_title: str = "Emotion Detector — press Q to quit"
    show_probability_bars: bool = True
    show_emotion_history: bool = True
    history_length: int = 90
    fps_ema_alpha: float = 0.1


@dataclass
class WebServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    mjpeg_quality: int = 85         # JPEG quality for video stream (1-95)


@dataclass
class SessionConfig:
    save_dir: str = "sessions"
    chart_window_seconds: int = 60


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/emotion_detector.log"
    max_bytes: int = 5_242_880  # 5 MB
    backup_count: int = 3


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    camera:    CameraConfig    = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    display:   DisplayConfig   = field(default_factory=DisplayConfig)
    web:       WebServerConfig = field(default_factory=WebServerConfig)
    session:   SessionConfig   = field(default_factory=SessionConfig)
    logging:   LoggingConfig   = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load config from a YAML file. Missing keys fall back to defaults."""
        if not _YAML_AVAILABLE:
            logger.warning(
                "PyYAML is not installed; using built-in defaults. "
                "Run: pip install pyyaml"
            )
            return cls()

        if not os.path.exists(path):
            logger.warning("Config file %r not found; using defaults.", path)
            return cls()

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        cam  = raw.get("camera",    {})
        inf  = raw.get("inference", {})
        disp = raw.get("display",   {})
        web  = raw.get("web",       {})
        sess = raw.get("session",   {})
        log  = raw.get("logging",   {})

        return cls(
            camera=CameraConfig(
                index=cam.get("index", 0),
                width=cam.get("width", 1280),
                height=cam.get("height", 720),
            ),
            inference=InferenceConfig(
                skip_frames=inf.get("skip_frames", 2),
                smoothing_window=inf.get("smoothing_window", 8),
                use_mtcnn=inf.get("use_mtcnn", False),
                min_detection_confidence=inf.get("min_detection_confidence", 0.5),
                min_tracking_confidence=inf.get("min_tracking_confidence", 0.5),
                max_num_faces=inf.get("max_num_faces", 5),
                model_backend=inf.get("model_backend", "fer"),
            ),
            display=DisplayConfig(
                window_title=disp.get("window_title", "Emotion Detector — press Q to quit"),
                show_probability_bars=disp.get("show_probability_bars", True),
                show_emotion_history=disp.get("show_emotion_history", True),
                history_length=disp.get("history_length", 90),
                fps_ema_alpha=disp.get("fps_ema_alpha", 0.1),
            ),
            web=WebServerConfig(
                host=web.get("host", "127.0.0.1"),
                port=web.get("port", 8080),
                mjpeg_quality=web.get("mjpeg_quality", 85),
            ),
            session=SessionConfig(
                save_dir=sess.get("save_dir", "sessions"),
                chart_window_seconds=sess.get("chart_window_seconds", 60),
            ),
            logging=LoggingConfig(
                level=log.get("level", "INFO"),
                log_to_file=log.get("log_to_file", True),
                log_file=log.get("log_file", "logs/emotion_detector.log"),
                max_bytes=log.get("max_bytes", 5_242_880),
                backup_count=log.get("backup_count", 3),
            ),
        )
