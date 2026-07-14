"""
camera_stream.py

Threaded webcam wrapper.

A background reader thread continuously pulls frames from cv2.VideoCapture
so the main loop never blocks waiting for the next frame. The main thread
always gets the most recent frame immediately via read().
"""

import logging
import threading
import time

import cv2

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Thread-safe webcam capture.

    Usage (context manager recommended)::

        with CameraStream(index=0, width=1280, height=720) as cam:
            ok, frame = cam.read()
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self._cap = cv2.VideoCapture(camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {camera_index}.\n"
                "  • Check the camera is connected and not in use by another app.\n"
                "  • On macOS: System Settings → Privacy & Security → Camera → "
                "enable access for Terminal / your app."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Camera %d opened — requested %dx%d, got %dx%d",
            camera_index, width, height, actual_w, actual_h,
        )

        self._frame = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="camera-reader",
            daemon=True,
        )
        self._thread.start()
        self._wait_for_first_frame(timeout=5.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self):
        """Return (success, frame). Frame is a BGR numpy array (copy)."""
        with self._frame_lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def is_alive(self) -> bool:
        """True if the background reader thread is still running."""
        return self._thread.is_alive()

    def release(self) -> None:
        """Stop the reader thread and release the camera."""
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            logger.info("Camera %d released.", self.camera_index)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Background thread: continuously read frames from the webcam."""
        logger.debug("Camera reader thread started.")
        consecutive_failures = 0

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if ok:
                consecutive_failures = 0
                with self._frame_lock:
                    self._frame = frame
            else:
                consecutive_failures += 1
                logger.warning(
                    "Camera read failed (attempt %d).", consecutive_failures
                )
                if consecutive_failures >= 10:
                    logger.error("Too many consecutive camera failures — stopping reader.")
                    break
                time.sleep(0.05)

        logger.debug("Camera reader thread exited.")

    def _wait_for_first_frame(self, timeout: float = 5.0) -> None:
        """Block until the first frame is available or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._frame_lock:
                if self._frame is not None:
                    return
            time.sleep(0.05)
        logger.warning("Timed out waiting for first camera frame.")
