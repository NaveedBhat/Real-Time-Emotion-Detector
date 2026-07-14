"""
server.py

FastAPI entry point for the Emotion Detector web dashboard.
Combines:
  1. MultiInferencePipeline (threading)
  2. SessionRecorder (CSV & memory buffer)
  3. FastAPI WebSocket (live chart/stats)
  4. FastAPI StreamingResponse (MJPEG video feed)
"""

import asyncio
import logging
import logging.handlers
import threading
import time
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.camera_stream import CameraStream
from src.config import AppConfig
from src.multi_face_pipeline import MultiInferencePipeline
from src.overlay import draw_multi_face_overlay
from src.session_recorder import SessionRecorder

# ---------------------------------------------------------------------------
# Configuration  (loaded once — no module-level object construction yet)
# ---------------------------------------------------------------------------
config = AppConfig.from_yaml("config.yaml")

# Setup Logging (console + optional rotating file from config)
_log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=config.logging.level, format=_log_format)

if config.logging.log_to_file:
    _log_file = Path(config.logging.log_file)
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.handlers.RotatingFileHandler(
        _log_file,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count,
        encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter(_log_format))
    logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------
_result_lock = threading.Lock()
last_inference_result = None
last_inference_fps = 0.0

# ---------------------------------------------------------------------------
# App Lifespan (Startup / Shutdown) — objects created ONCE here, not at import
# ---------------------------------------------------------------------------
camera: CameraStream = None       # type: ignore[assignment]
pipeline: MultiInferencePipeline = None  # type: ignore[assignment]
recorder: SessionRecorder = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera, pipeline, recorder
    logger.info("Starting up inference pipeline...")
    camera = CameraStream(
        camera_index=config.camera.index,
        width=config.camera.width,
        height=config.camera.height,
    )
    pipeline = MultiInferencePipeline(
        max_num_faces=config.inference.max_num_faces,
        skip_frames=config.inference.skip_frames,
        smoothing_window=config.inference.smoothing_window,
        model_backend=config.inference.model_backend,
        use_mtcnn=config.inference.use_mtcnn,
        min_detection_confidence=config.inference.min_detection_confidence,
        min_tracking_confidence=config.inference.min_tracking_confidence,
    )
    recorder = SessionRecorder(
        save_dir=config.session.save_dir,
        chart_window_seconds=config.session.chart_window_seconds,
    )
    pipeline.start()
    yield
    logger.info("Shutting down...")
    pipeline.stop()
    camera.release()


app = FastAPI(title="Emotion Detector Dashboard", lifespan=lifespan)

# Mount Static Files
dashboard_dir = Path(__file__).parent / "dashboard"
app.mount("/static", StaticFiles(directory=dashboard_dir), name="static")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    """Serve the dashboard HTML."""
    return FileResponse(dashboard_dir / "index.html")


@app.get("/export_csv")
def export_csv():
    """Download the current session's CSV file."""
    return FileResponse(
        path=recorder.csv_path,
        media_type="text/csv",
        filename=recorder.csv_path.name,
    )


# ---------------------------------------------------------------------------
# Video Stream (MJPEG)
# ---------------------------------------------------------------------------

def generate_frames():
    """Generator for MJPEG stream. Grabs frames, renders overlay, yields JPEG bytes."""
    global last_inference_result, last_inference_fps
    fps_alpha = config.display.fps_ema_alpha
    fps = 0.0
    last_frame_time = time.perf_counter()
    local_last_result = None
    local_last_fps = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                if not camera.is_alive():
                    logger.error("Camera reader thread has died — stopping video stream.")
                    break
                time.sleep(0.01)
                continue

            # Pipeline
            pipeline.submit(frame)
            result = pipeline.get_result(timeout=0.005)

            # FPS calculation
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now
            current_fps = 1.0 / dt if dt > 0 else 0.0
            fps = (fps_alpha * current_fps) + ((1.0 - fps_alpha) * fps)

            # Update shared state under lock
            if result:
                local_last_result = result
                local_last_fps = fps
                with _result_lock:
                    last_inference_result = result
                    last_inference_fps = fps
                # Record data only when a fresh result arrives
                recorder.record(result.faces, timestamp=time.time())

            # Render overlay using the most recent known result (prevents blinking)
            if local_last_result:
                draw_multi_face_overlay(
                    frame,
                    faces=local_last_result.faces,
                    fps=local_last_fps,
                    frame_count=local_last_result.frame_count,
                    inference_ms=local_last_result.inference_ms,
                    emotion_history=None,   # Not used in web mode (chart does this)
                    show_emotion_history=False,
                )

            # Encode to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), config.web.mjpeg_quality]
            success, buffer = cv2.imencode(".jpg", frame, encode_param)
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

            # Yield control slightly so asyncio doesn't block entirely on this sync generator
            time.sleep(0.001)
    except GeneratorExit:
        # Client disconnected — clean exit, no log spam
        pass


@app.get("/video_feed")
def video_feed():
    """MJPEG streaming endpoint."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# WebSocket (Live Analytics)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Pushes chart data and HUD stats to the client every 500ms."""
    await websocket.accept()
    logger.info("WebSocket connected.")
    try:
        while True:
            try:
                # Read shared state under lock
                with _result_lock:
                    res = last_inference_result
                    res_fps = last_inference_fps

                hud_data = None
                if res:
                    hud_data = {
                        "n_faces": len(res.faces),
                        "fps": res_fps,
                        "faces": [
                            {
                                "id": f.face_id,
                                "label": f.label,
                                "confidence": f.confidence,
                            }
                            for f in res.faces
                        ],
                    }

                payload = {
                    "chart": recorder.get_chart_data(),
                    "stats": recorder.get_session_stats(),
                    "hud": hud_data,
                }

                await websocket.send_json(payload)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected.")
                break
            except RuntimeError as e:
                msg = str(e)
                if "close message has been sent" in msg or "WebSocket is not connected" in msg:
                    logger.info("WebSocket closed.")
                else:
                    logger.error("RuntimeError in websocket loop: %s", e)
                break
            except asyncio.CancelledError:
                # Server is shutting down — exit silently, no traceback
                break
            except Exception as e:
                logger.error("Error in websocket loop: %s", e)

            await asyncio.sleep(0.5)  # 2 Hz update rate
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except asyncio.CancelledError:
        # Raised by uvicorn during graceful shutdown — not an error
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting server on http://%s:%d", config.web.host, config.web.port)
    # Pass the app object directly (not a string) so uvicorn does NOT re-import
    # this module, preventing the double-initialization of camera/pipeline/recorder.
    uvicorn.run(
        app,
        host=config.web.host,
        port=config.web.port,
        reload=False,
    )
