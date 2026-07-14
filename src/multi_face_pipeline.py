"""
multi_face_pipeline.py

Threaded multi-face inference pipeline for the web dashboard.

Differences from single-face pipeline.py:
  - Detects up to N faces per frame (configurable)
  - Assigns stable face IDs across frames using centroid matching (FaceTracker)
  - Each face slot has its own EmotionSmoother
  - Uses pluggable ModelBackend (fer | deepface)
  - Output: MultiInferenceResult with a List[FaceResult]
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .smoothing import EmotionSmoother

if TYPE_CHECKING:
    from .face_landmarks import FaceGeometry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    """Emotion inference output for a single detected face."""
    face_id:     int
    label:       Optional[str]
    confidence:  float
    bbox:        tuple           # (x, y, w, h)
    all_scores:  Optional[dict]  # {emotion: probability}


@dataclass
class MultiInferenceResult:
    """One complete inference output covering all detected faces."""
    faces:          List[FaceResult]
    frame_count:    int
    inference_ms:   float
    timestamp:      float = field(default_factory=time.time)

    @property
    def face_detected(self) -> bool:
        return len(self.faces) > 0


# ---------------------------------------------------------------------------
# Face tracker — stable IDs across frames via centroid matching
# ---------------------------------------------------------------------------

class FaceTracker:
    """
    Assigns persistent integer IDs to faces across frames.

    Algorithm:
      Each frame, existing slots are aged by 1. New face centroids
      are matched to the nearest slot (greedy, within a pixel threshold).
      Unmatched faces get a new ID. Slots unseen for disappear_frames
      are evicted.
    """

    def __init__(self, max_faces: int = 5, disappear_frames: int = 20,
                 match_threshold: float = 180.0, smoothing_window: int = 8):
        self._max_faces        = max_faces
        self._disappear_frames = disappear_frames
        self._match_threshold  = match_threshold
        self._smoothing_window = smoothing_window
        self._next_id          = 0
        # {face_id: {centroid, frames_since_seen, smoother}}
        self._slots: Dict[int, dict] = {}

    def update(self, geometries: list) -> List[Tuple[int, Any, EmotionSmoother]]:
        """
        Match geometries to slots.

        Returns:
            List of (face_id, FaceGeometry, EmotionSmoother)
        """
        # Age all slots
        for slot in self._slots.values():
            slot["frames_since_seen"] += 1

        # Evict stale slots
        stale = [fid for fid, slot in self._slots.items()
                 if slot["frames_since_seen"] > self._disappear_frames]
        for fid in stale:
            self._slots[fid]["smoother"].reset()
            del self._slots[fid]
            logger.debug("Face slot %d evicted (not seen for %d frames).", fid, self._disappear_frames)

        if not geometries:
            return []

        # Build centroid list for new detections
        centroids = []
        for g in geometries:
            cx = g.bbox[0] + g.bbox[2] // 2
            cy = g.bbox[1] + g.bbox[3] // 2
            centroids.append((cx, cy))

        available_slots = list(self._slots.keys())
        assignments: Dict[int, int] = {}   # geom_index -> face_id

        for gi, (cx, cy) in enumerate(centroids):
            best_dist = float("inf")
            best_fid  = None

            for fid in available_slots:
                scx, scy = self._slots[fid]["centroid"]
                dist = ((cx - scx) ** 2 + (cy - scy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_fid  = fid

            if best_fid is not None and best_dist <= self._match_threshold:
                # Match to existing slot
                self._slots[best_fid]["centroid"]          = (cx, cy)
                self._slots[best_fid]["frames_since_seen"] = 0
                available_slots.remove(best_fid)
                assignments[gi] = best_fid
            elif len(self._slots) < self._max_faces:
                # New slot
                new_id = self._next_id
                self._next_id += 1
                self._slots[new_id] = {
                    "centroid":          (cx, cy),
                    "frames_since_seen": 0,
                    "smoother":          EmotionSmoother(window_size=self._smoothing_window),
                }
                assignments[gi] = new_id
                logger.debug("New face slot assigned: ID=%d.", new_id)

        result = []
        for gi, geom in enumerate(geometries):
            if gi in assignments:
                fid     = assignments[gi]
                smoother = self._slots[fid]["smoother"]
                result.append((fid, geom, smoother))

        return result

    def reset(self) -> None:
        for slot in self._slots.values():
            slot["smoother"].reset()
        self._slots.clear()


# ---------------------------------------------------------------------------
# Multi-face pipeline
# ---------------------------------------------------------------------------

class MultiInferencePipeline:
    """
    Background thread that runs multi-face detection + emotion inference.

    All MediaPipe and model objects live entirely within the worker thread.
    """

    def __init__(
        self,
        max_num_faces:               int   = 5,
        skip_frames:                 int   = 2,
        smoothing_window:            int   = 8,
        model_backend:               str   = "fer",
        use_mtcnn:                   bool  = False,
        min_detection_confidence:    float = 0.5,
        min_tracking_confidence:     float = 0.5,
    ):
        self._max_num_faces    = max_num_faces
        self._skip_frames      = skip_frames
        self._smoothing_window = smoothing_window
        self._model_backend    = model_backend
        self._use_mtcnn        = use_mtcnn
        self._min_det          = min_detection_confidence
        self._min_trk          = min_tracking_confidence

        self._input_q:  queue.Queue = queue.Queue(maxsize=2)
        self._output_q: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event             = threading.Event()

        self._thread = threading.Thread(
            target=self._worker,
            name="multi-inference-worker",
            daemon=True,
        )
        self._total_frames_processed = 0

    # ------------------------------------------------------------------
    # Public API (same as single-face InferencePipeline)
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()
        logger.info(
            "Multi-face inference pipeline started "
            "(max_faces=%d, skip=%d, backend=%s).",
            self._max_num_faces, self._skip_frames, self._model_backend,
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("Multi-inference thread did not exit cleanly.")
        else:
            logger.info(
                "Multi-face pipeline stopped. Frames processed: %d.",
                self._total_frames_processed,
            )

    def submit(self, frame) -> None:
        """Non-blocking submit. Drops oldest frame if queue is full."""
        try:
            self._input_q.put_nowait(frame)
        except queue.Full:
            try:
                self._input_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._input_q.put_nowait(frame)
            except queue.Full:
                pass

    def get_result(self, timeout: float = 0.005) -> Optional[MultiInferenceResult]:
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
        from .face_landmarks import FaceLandmarkDetector
        from .model_backend   import ModelBackend

        face_detector = FaceLandmarkDetector(
            min_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_trk,
            max_num_faces=self._max_num_faces,
        )
        backend = ModelBackend.create(
            self._model_backend,
            use_mtcnn=self._use_mtcnn,
        )
        tracker = FaceTracker(
            max_faces=self._max_num_faces,
            smoothing_window=self._smoothing_window,
        )

        frame_count     = 0
        last_scores_map: Dict[int, Optional[dict]] = {}   # face_id -> last known scores

        logger.debug("Multi-inference worker ready.")

        try:
            while not self._stop_event.is_set():
                try:
                    frame = self._input_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                t0 = time.perf_counter()

                try:
                    geometries = face_detector.process(frame)
                except Exception:
                    logger.exception("Multi-face detection error.")
                    continue

                face_results: List[FaceResult] = []
                
                # Always age tracker and preserve IDs even if no faces found in this specific frame
                frame_count += 1
                self._total_frames_processed += 1
                matched = tracker.update(geometries)

                for fid, geom, smoother in matched:
                    # Run inference if it's time, OR if we have no prior scores for this new face ID
                    run_inference = (frame_count % self._skip_frames == 0) or (fid not in last_scores_map)

                    if run_inference:
                        x, y, w, h = geom.bbox
                        face_crop = frame[y: y + h, x: x + w]
                        try:
                            scores = backend.analyze(face_crop, geom)
                            last_scores_map[fid] = scores
                        except Exception:
                            logger.exception("Backend error for face %d.", fid)

                    scores    = last_scores_map.get(fid)
                    smoothed  = smoother.update(scores)

                    label, confidence = EmotionSmoother.top_emotion(smoothed)

                    face_results.append(FaceResult(
                        face_id=fid,
                        label=label,
                        confidence=confidence,
                        bbox=geom.bbox,
                        all_scores=smoothed,
                    ))

                # Cleanup last_scores_map for slots that were truly evicted by the tracker
                active_fids = set(tracker._slots.keys())
                for k in list(last_scores_map.keys()):
                    if k not in active_fids:
                        del last_scores_map[k]

                inference_ms = (time.perf_counter() - t0) * 1000
                result = MultiInferenceResult(
                    faces=face_results,
                    frame_count=self._total_frames_processed,
                    inference_ms=inference_ms,
                )

                # Publish (drop stale if consumer is slow)
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
            logger.exception("Unhandled exception in multi-inference worker.")
        finally:
            face_detector.close()
            logger.debug("Multi-inference worker exited.")
