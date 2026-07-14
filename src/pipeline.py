"""
pipeline.py

Threaded inference worker.

The InferencePipeline runs MediaPipe face detection + FER emotion
classification in a dedicated background thread, completely decoupled
from the display loop. This means:

  • Display always runs at full camera FPS
  • Inference runs at its own pace without blocking the UI
  • Frames are dropped (not queued up) when inference is slower than capture

Architecture::

    Main thread                  Inference thread
    ──────────────               ──────────────────────────────
    camera.read()   ──submit──▶  MediaPipe Face Mesh
    draw overlay    ◀─result──   FER emotion classifier
    cv2.imshow()                 Landmark nudge
                                 EmotionSmoother
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """One complete inference output, passed from the worker to the display loop."""
    label: Optional[str]           # dominant emotion string, e.g. "happy"
    confidence: float              # 0.0 – 1.0 smoothed confidence
    bbox: Optional[tuple]          # (x, y, w, h) in pixels
    all_scores: Optional[dict]     # {emotion: probability} for all 7 classes
    face_detected: bool
    inference_ms: float            # wall-clock time for one inference pass


class InferencePipeline:
    """
    Manages a background thread that runs face detection + emotion inference.

    All MediaPipe and FER objects are created *inside* the worker thread
    to respect their single-thread affinity requirements.
    """

    def __init__(
        self,
        skip_frames: int = 2,
        smoothing_window: int = 8,
        use_mtcnn: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._skip_frames = skip_frames
        self._smoothing_window = smoothing_window
        self._use_mtcnn = use_mtcnn
        self._min_det = min_detection_confidence
        self._min_trk = min_tracking_confidence

        # maxsize=2 keeps latency low: if inference is slow, the oldest
        # unprocessed frame is discarded when a newer one arrives.
        self._input_q: queue.Queue = queue.Queue(maxsize=2)
        self._output_q: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        self._thread = threading.Thread(
            target=self._worker,
            name="inference-worker",
            daemon=True,
        )

        # Performance counters (read from main thread; written from worker thread)
        self._total_frames_processed = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background inference thread."""
        self._thread.start()
        logger.info("Inference pipeline started (skip_frames=%d).", self._skip_frames)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("Inference thread did not exit cleanly within %.1fs.", timeout)
        else:
            logger.info(
                "Inference pipeline stopped. Total frames processed: %d.",
                self._total_frames_processed,
            )

    def submit(self, frame) -> None:
        """
        Submit a frame for inference. Non-blocking.
        If the input queue is full (inference is slow) the oldest frame is
        dropped and the new one is enqueued — keeps results fresh.
        """
        try:
            self._input_q.put_nowait(frame)
        except queue.Full:
            try:
                self._input_q.get_nowait()   # discard stale frame
            except queue.Empty:
                pass
            try:
                self._input_q.put_nowait(frame)
            except queue.Full:
                pass

    def get_result(self, timeout: float = 0.005) -> Optional[InferenceResult]:
        """
        Get the latest inference result. Returns None if nothing is ready yet.
        The caller should render the previous result while waiting.
        """
        try:
            return self._output_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """
        Background inference loop.
        All heavy objects (MediaPipe, FER) are created here so they live
        entirely within this thread.
        """
        # Lazy imports keep the main thread startup fast
        from .face_landmarks import FaceLandmarkDetector
        from .emotion_analyzer import EmotionAnalyzer
        from .smoothing import EmotionSmoother

        face_detector = FaceLandmarkDetector(
            min_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_trk,
        )
        emotion_analyzer = EmotionAnalyzer(use_mtcnn=self._use_mtcnn)
        smoother = EmotionSmoother(window_size=self._smoothing_window)

        frame_count = 0
        last_scores = None

        logger.debug("Inference worker ready.")

        try:
            while not self._stop_event.is_set():
                # Wait for a frame (short timeout so we can check stop_event)
                try:
                    frame = self._input_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                t0 = time.perf_counter()

                try:
                    geometries = face_detector.process(frame)
                except Exception:
                    logger.exception("Face detection error; skipping frame.")
                    continue

                geometry = geometries[0] if geometries else None

                if geometry is None:
                    smoother.reset()
                    last_scores = None
                    result = InferenceResult(
                        label=None,
                        confidence=0.0,
                        bbox=None,
                        all_scores=None,
                        face_detected=False,
                        inference_ms=0.0,
                    )
                else:
                    frame_count += 1
                    self._total_frames_processed += 1

                    # Only run the heavy FER model every skip_frames
                    if frame_count % self._skip_frames == 0:
                        x, y, w, h = geometry.bbox
                        face_crop = frame[y: y + h, x: x + w]
                        try:
                            last_scores = emotion_analyzer.analyze(face_crop, geometry)
                        except Exception:
                            logger.exception("Emotion analysis error; using cached scores.")

                    smoothed = smoother.update(last_scores)
                    label, confidence = smoother.top_emotion(smoothed)

                    inference_ms = (time.perf_counter() - t0) * 1000
                    result = InferenceResult(
                        label=label,
                        confidence=confidence,
                        bbox=geometry.bbox,
                        all_scores=smoothed,
                        face_detected=True,
                        inference_ms=inference_ms,
                    )

                # Publish result — discard stale output if consumer is slow
                try:
                    self._output_q.put_nowait(result)
                except queue.Full:
                    try:
                        self._output_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._output_q.put_nowait(result)
                    except queue.Full:
                        pass

        except Exception:
            logger.exception("Unhandled exception in inference worker.")
        finally:
            face_detector.close()
            logger.debug("Inference worker exited.")
