"""
session_recorder.py

Records per-frame emotion data to CSV and keeps a rolling in-memory buffer
for the web dashboard's live chart.

CSV row format:
  timestamp, face_id, label, confidence, happy, neutral, surprise, sad, angry, fear, disgust
"""

from __future__ import annotations

import csv
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import List, Optional

from .constants import EMOTIONS

logger = logging.getLogger(__name__)

CSV_FIELDS = ["timestamp", "face_id", "label", "confidence"] + EMOTIONS


class SessionRecorder:
    """
    Thread-safe session recorder.

    Usage::

        recorder = SessionRecorder(save_dir="sessions", chart_window_seconds=60)
        recorder.record(faces, timestamp=time.time())
        chart_data = recorder.get_chart_data(seconds=60)
        csv_path   = recorder.csv_path
    """

    def __init__(self, save_dir: str = "sessions", chart_window_seconds: int = 60):
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._chart_window = chart_window_seconds
        self._prune_extra = 10     # keep 10 seconds extra beyond chart window

        # One CSV file per session, named by start time
        session_ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        self._csv_path = self._save_dir / f"session_{session_ts}.csv"

        self._lock = Lock()
        # deque of row dicts: {timestamp, face_id, label, confidence, <emotion>: prob, ...}
        self._history: deque = deque()

        # Cumulative stats for the entire session
        self._total_records_all_time: int = 0
        self._emotion_counts_all_time: dict = {e: 0 for e in EMOTIONS}
        self._sum_confidence_all_time: float = 0.0

        # Write CSV header
        with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

        logger.info("Session recorder started — %s", self._csv_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, faces, timestamp: Optional[float] = None) -> None:
        """
        Record one frame's worth of face results.

        faces: list of FaceResult objects (from multi_face_pipeline)
        """
        if timestamp is None:
            timestamp = time.time()

        rows = []
        for face in faces:
            if face.label is None:
                continue
            scores = face.all_scores or {}
            row = {
                "timestamp":  timestamp,
                "face_id":    face.face_id,
                "label":      face.label,
                "confidence": round(face.confidence, 4),
            }
            for e in EMOTIONS:
                row[e] = round(scores.get(e, 0.0), 4)
            rows.append(row)

        if not rows:
            return

        with self._lock:
            for row in rows:
                self._history.append(row)
                self._total_records_all_time += 1
                self._emotion_counts_all_time[row["label"]] += 1
                self._sum_confidence_all_time += row["confidence"]
            self._prune_history()

        # Append to CSV (outside lock to avoid holding it during I/O)
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writerows(rows)
        except OSError:
            logger.exception("Failed to write to session CSV.")

    def get_chart_data(self, seconds: int = 60) -> dict:
        """
        Return time-bucketed emotion averages for Chart.js.

        Returns::
            {
              "labels": ["18:30:00", "18:30:01", ...],
              "datasets": {
                "happy":   [12.5, 18.0, ...],   # 0-100 percentages
                "neutral": [...],
                ...
              }
            }
        """
        cutoff = time.time() - seconds
        with self._lock:
            recent = [r for r in self._history if r["timestamp"] >= cutoff]

        if not recent:
            return {"labels": [], "datasets": {e: [] for e in EMOTIONS}}

        # Group rows into 1-second buckets using a dict — O(n) instead of O(n * window)
        buckets: dict = defaultdict(list)
        for r in recent:
            buckets[int(r["timestamp"])].append(r)

        start_ts = int(recent[0]["timestamp"])
        end_ts   = int(recent[-1]["timestamp"])
        labels   = []
        datasets = {e: [] for e in EMOTIONS}

        for t in range(start_ts, end_ts + 1):
            labels.append(time.strftime("%H:%M:%S", time.localtime(t)))
            bucket = buckets.get(t, [])
            for e in EMOTIONS:
                if bucket:
                    avg = sum(r.get(e, 0) for r in bucket) / len(bucket)
                else:
                    avg = 0.0
                datasets[e].append(round(avg * 100, 1))

        return {"labels": labels, "datasets": datasets}

    def get_session_stats(self) -> dict:
        """High-level session statistics for the dashboard stats bar."""
        with self._lock:
            if self._total_records_all_time == 0:
                return {"dominant": None, "total_records": 0, "avg_confidence": 0.0}

            dominant = max(self._emotion_counts_all_time.items(), key=lambda x: x[1])[0]
            avg_conf = self._sum_confidence_all_time / self._total_records_all_time
            total = self._total_records_all_time

        return {
            "dominant":       dominant,
            "total_records":  total,
            "avg_confidence": round(avg_conf * 100, 1),
        }

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune_history(self) -> None:
        """Remove entries older than chart_window + prune_extra seconds. Call with lock held."""
        cutoff = time.time() - self._chart_window - self._prune_extra
        while self._history and self._history[0]["timestamp"] < cutoff:
            self._history.popleft()
