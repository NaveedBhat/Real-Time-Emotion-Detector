"""
Real-Time Facial Emotion Detector
==================================

Advanced, production-ready entry point.

Architecture:
  • CameraStream    — background thread reads webcam continuously
  • InferencePipeline — background thread runs MediaPipe + FER + smoother
  • Main thread     — display loop: draws HUD, never blocks on inference

Pipeline (inside InferencePipeline worker thread):
  1. MediaPipe Face Mesh   -> face bounding box + mouth/lip geometry
  2. Crop face region      -> feed to FER (pretrained CNN)         [every N frames]
  3. FER probabilities     -> nudged slightly using mouth geometry
  4. EmotionSmoother       -> rolling average over last N predictions

Controls:
  q / ESC  —  quit

Run:
  python main.py
  python main.py --config config.yaml
  python main.py --camera 1 --skip 3 --debug
"""

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from collections import deque

import cv2

from src.camera_stream import CameraStream
from src.config import AppConfig
from src.pipeline import InferencePipeline
from src import overlay

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_cfg) -> None:
    """Configure root logger with console + optional rotating file handler."""
    level = getattr(logging, log_cfg.level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-20s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # catch everything; handlers filter

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler (rotating)
    if log_cfg.log_to_file:
        log_dir = os.path.dirname(log_cfg.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_cfg.log_file,
            maxBytes=log_cfg.max_bytes,
            backupCount=log_cfg.backup_count,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time facial emotion detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",           default="config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--camera",           type=int, default=None,
                   help="Webcam index (overrides config)")
    p.add_argument("--width",            type=int, default=None,
                   help="Capture width (overrides config)")
    p.add_argument("--height",           type=int, default=None,
                   help="Capture height (overrides config)")
    p.add_argument("--skip",             type=int, default=None,
                   help="Run FER every N frames (overrides config)")
    p.add_argument("--smoothing-window", type=int, default=None,
                   help="Rolling average window size (overrides config)")
    p.add_argument("--no-bars",          action="store_true",
                   help="Disable probability bars panel")
    p.add_argument("--no-history",       action="store_true",
                   help="Disable emotion history strip")
    p.add_argument("--debug",            action="store_true",
                   help="Set log level to DEBUG")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Config ---
    cfg = AppConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.camera is not None:
        cfg.camera.index = args.camera
    if args.width is not None:
        cfg.camera.width = args.width
    if args.height is not None:
        cfg.camera.height = args.height
    if args.skip is not None:
        cfg.inference.skip_frames = args.skip
    if args.smoothing_window is not None:
        cfg.inference.smoothing_window = args.smoothing_window
    if args.no_bars:
        cfg.display.show_probability_bars = False
    if args.no_history:
        cfg.display.show_emotion_history = False
    if args.debug:
        cfg.logging.level = "DEBUG"

    # --- Logging ---
    setup_logging(cfg.logging)
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting Emotion Detector — camera=%d  %dx%d  skip=%d  smooth=%d",
        cfg.camera.index, cfg.camera.width, cfg.camera.height,
        cfg.inference.skip_frames, cfg.inference.smoothing_window,
    )

    # --- Graceful shutdown flag ---
    stop_event = threading.Event()

    def _on_signal(sig, frame):
        logger.info("Signal %s received — shutting down...", signal.Signals(sig).name)
        stop_event.set()

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # --- Pipeline ---
    pipeline = InferencePipeline(
        skip_frames=cfg.inference.skip_frames,
        smoothing_window=cfg.inference.smoothing_window,
        use_mtcnn=cfg.inference.use_mtcnn,
        min_detection_confidence=cfg.inference.min_detection_confidence,
        min_tracking_confidence=cfg.inference.min_tracking_confidence,
    )

    # --- Emotion history buffer ---
    emotion_history: deque = deque(maxlen=cfg.display.history_length)

    # -----------------------------------------------------------------------
    # Display loop
    # -----------------------------------------------------------------------
    start_time   = time.time()
    frame_count  = 0
    fps          = 0.0
    prev_time    = time.time()
    alpha        = cfg.display.fps_ema_alpha
    latest       = None                 # most recent InferenceResult

    try:
        with CameraStream(
            camera_index=cfg.camera.index,
            width=cfg.camera.width,
            height=cfg.camera.height,
        ) as camera:

            pipeline.start()
            logger.info("Press Q or ESC in the video window to quit.")

            while not stop_event.is_set():

                # --- Capture ---
                ok, frame = camera.read()
                if not ok:
                    logger.warning("Camera read returned no frame; retrying...")
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)          # mirror for natural selfie view

                # --- Submit to inference thread (non-blocking) ---
                pipeline.submit(frame.copy())

                # --- Pull latest result (non-blocking) ---
                result = pipeline.get_result(timeout=0.005)
                if result is not None:
                    latest = result
                    if result.face_detected and result.label:
                        emotion_history.append(result.label)
                    elif not result.face_detected:
                        emotion_history.append(None)

                # --- Draw HUD ---
                if latest is not None:
                    overlay.draw_all(
                        frame,
                        label=latest.label,
                        confidence=latest.confidence,
                        bbox=latest.bbox,
                        all_scores=latest.all_scores,
                        face_detected=latest.face_detected,
                        fps=fps,
                        frame_count=frame_count,
                        inference_ms=latest.inference_ms,
                        emotion_history=emotion_history,
                        show_probability_bars=cfg.display.show_probability_bars,
                        show_emotion_history=cfg.display.show_emotion_history,
                    )
                else:
                    # Pipeline hasn't produced a result yet — show minimal HUD
                    overlay.draw_all(
                        frame,
                        label=None, confidence=0.0, bbox=None, all_scores=None,
                        face_detected=False, fps=fps, frame_count=frame_count,
                        inference_ms=0.0, emotion_history=emotion_history,
                        show_probability_bars=False, show_emotion_history=False,
                    )

                # --- FPS (exponential moving average) ---
                now   = time.time()
                dt    = now - prev_time
                prev_time = now
                if dt > 0:
                    fps = fps * (1 - alpha) + (1.0 / dt) * alpha

                # --- Display ---
                cv2.imshow(cfg.display.window_title, frame)
                frame_count += 1

                # --- Key handling ---
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):   # q, Q, ESC
                    logger.info("Quit key pressed.")
                    stop_event.set()

                # --- Sanity-check threads ---
                if not camera.is_alive():
                    logger.error("Camera reader thread died unexpectedly.")
                    stop_event.set()
                if not pipeline.is_alive():
                    logger.error("Inference thread died unexpectedly.")
                    stop_event.set()

    except RuntimeError as exc:
        logger.critical("Fatal startup error: %s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Unhandled exception in main loop.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        logger.info(
            "Stopped. Runtime: %.1fs | Frames displayed: %d | Avg FPS: %.1f",
            elapsed, frame_count, frame_count / max(elapsed, 1),
        )


if __name__ == "__main__":
    main()
