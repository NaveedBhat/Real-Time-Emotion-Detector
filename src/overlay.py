"""
overlay.py

Rich visual overlay for the emotion detector HUD.

Visual elements:
  1. Color-coded corner-bracket bounding box  (emotion-specific color)
  2. Semi-transparent emotion label pill       (above the face box)
  3. Probability bars panel                    (top-right corner)
  4. Emotion history timeline strip            (bottom of frame)
  5. HUD stats bar                             (top-left: FPS, inference, frame#)
  6. "No face" centered message
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# BGR colors per emotion
EMOTION_COLORS: dict[str, tuple] = {
    "happy":    (0,   210,  90),   # vibrant green
    "sad":      (200,  80,  20),   # deep blue-orange
    "angry":    ( 30,  30, 220),   # strong red
    "surprise": (  0, 200, 230),   # yellow-orange
    "fear":     (190,  40, 190),   # purple
    "disgust":  ( 40, 170,  80),   # teal-green
    "neutral":  (150, 150, 150),   # gray
}

# Canonical display order for the probability panel
EMOTION_ORDER = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]

EMOTION_EMOJI = {
    "happy":    ":)",
    "sad":      ":(",
    "angry":    ">:(",
    "surprise": ":O",
    "fear":     "D:",
    "disgust":  ":S",
    "neutral":  ":|",
}

_DEFAULT_COLOR = (150, 150, 150)
_TEXT_WHITE    = (255, 255, 255)
_TEXT_DIM      = (180, 180, 180)
_PANEL_BG      = (18,  18,  18)
_HUD_GREEN     = (  0, 210,  90)
_HUD_YELLOW    = (  0, 200, 230)
_HUD_RED       = ( 30,  30, 220)


# ---------------------------------------------------------------------------
# Master draw function
# ---------------------------------------------------------------------------

def draw_all(
    frame,
    *,
    label: Optional[str],
    confidence: float,
    bbox: Optional[tuple],
    all_scores: Optional[dict],
    face_detected: bool,
    fps: float,
    frame_count: int,
    inference_ms: float,
    emotion_history: deque,
    show_probability_bars: bool = True,
    show_emotion_history: bool = True,
) -> None:
    """
    Single call that renders the complete HUD onto *frame* in-place.

    Call this once per display frame in the main loop.
    """
    h, w = frame.shape[:2]
    color = EMOTION_COLORS.get(label, _DEFAULT_COLOR) if label else _DEFAULT_COLOR

    # 1 — Face box & label
    if face_detected and bbox:
        _draw_corner_box(frame, bbox, color)
        if label:
            _draw_emotion_pill(frame, bbox, label, confidence, color, w, h)
    else:
        _draw_no_face_message(frame, w, h)

    # 2 — Probability bars panel (top-right)
    if show_probability_bars and all_scores:
        _draw_probability_panel(frame, all_scores, label, w, h)

    # 3 — History timeline strip (bottom)
    if show_emotion_history:
        _draw_history_strip(frame, emotion_history, w, h)

    # 4 — HUD stats (top-left)
    _draw_hud(frame, fps, frame_count, inference_ms)


# ---------------------------------------------------------------------------
# Element renderers
# ---------------------------------------------------------------------------

def _draw_corner_box(frame, bbox: tuple, color: tuple) -> None:
    """
    Draw four corner-bracket markers instead of a plain rectangle.
    Looks more modern / professional.
    """
    x, y, w, h = bbox
    t = 3                           # line thickness
    arm = min(w, h) // 5            # length of each bracket arm

    pts = [
        # (start, end_x, end_y)  — top-left
        ((x, y),           (x + arm, y),     (x, y + arm)),
        # top-right
        ((x + w, y),       (x + w - arm, y), (x + w, y + arm)),
        # bottom-left
        ((x, y + h),       (x + arm, y + h), (x, y + h - arm)),
        # bottom-right
        ((x + w, y + h),   (x + w - arm, y + h), (x + w, y + h - arm)),
    ]
    for corner, horiz, vert in pts:
        cv2.line(frame, corner, horiz, color, t, cv2.LINE_AA)
        cv2.line(frame, corner, vert,  color, t, cv2.LINE_AA)

    # Subtle inner fill at 8% opacity for a "scanning" look
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)


def _draw_emotion_pill(
    frame, bbox: tuple, label: str, confidence: float,
    color: tuple, frame_w: int, frame_h: int
) -> None:
    """Semi-transparent pill label above the bounding box (edge-clamped)."""
    x, y, w, _ = bbox
    emoji = EMOTION_EMOJI.get(label, "")
    text = f"{emoji}  {label.upper()}   {confidence * 100:.0f}%"

    font_scale = 0.72
    thick = 2
    (tw, th), baseline = cv2.getTextSize(text, FONT_BOLD, font_scale, thick)

    pad_x, pad_y = 12, 7
    pill_w = tw + pad_x * 2
    pill_h = th + baseline + pad_y * 2

    # Position: centre above the face box, clamped to frame
    rx = max(0, min(x, frame_w - pill_w))
    ry = max(pill_h, y - 8)          # never goes above top edge

    # Semi-transparent background
    ovl = frame.copy()
    cv2.rectangle(ovl, (rx, ry - pill_h), (rx + pill_w, ry), color, -1)
    cv2.addWeighted(ovl, 0.82, frame, 0.18, 0, frame)

    # 1-px border for definition
    cv2.rectangle(frame, (rx, ry - pill_h), (rx + pill_w, ry), color, 1, cv2.LINE_AA)

    # Text
    cv2.putText(
        frame, text,
        (rx + pad_x, ry - pad_y - baseline),
        FONT_BOLD, font_scale, _TEXT_WHITE, thick, cv2.LINE_AA,
    )


def _draw_probability_panel(
    frame, scores: dict, dominant: Optional[str], frame_w: int, frame_h: int
) -> None:
    """
    Vertical panel in the top-right corner showing all 7 emotion bars.
    """
    panel_w   = 215
    bar_h     = 16
    row_gap   = 6
    pad       = 10
    label_w   = 72        # pixels reserved for the emotion name
    pct_w     = 38        # pixels reserved for "100%" text
    title_h   = 22
    n         = len(EMOTION_ORDER)
    panel_h   = title_h + n * (bar_h + row_gap) + pad

    px = frame_w - panel_w - 12
    py = 12

    # Panel background
    ovl = frame.copy()
    cv2.rectangle(ovl, (px, py), (px + panel_w, py + panel_h), _PANEL_BG, -1)
    cv2.addWeighted(ovl, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (70, 70, 70), 1)

    # Title
    cv2.putText(
        frame, "EMOTIONS", (px + pad, py + title_h - 5),
        FONT, 0.42, (190, 190, 190), 1, cv2.LINE_AA,
    )

    bar_area_w = panel_w - label_w - pct_w - pad

    for i, emotion in enumerate(EMOTION_ORDER):
        prob  = scores.get(emotion, 0.0)
        color = EMOTION_COLORS.get(emotion, _DEFAULT_COLOR)
        is_dominant = (emotion == dominant)

        row_y = py + title_h + i * (bar_h + row_gap)

        # Emotion name — bold if dominant
        name_font  = FONT_BOLD if is_dominant else FONT
        name_color = color if is_dominant else _TEXT_DIM
        cv2.putText(
            frame, emotion.capitalize(),
            (px + pad, row_y + bar_h - 3),
            name_font, 0.38, name_color, 1, cv2.LINE_AA,
        )

        # Bar background (dark slot)
        bx = px + label_w
        cv2.rectangle(frame, (bx, row_y), (bx + bar_area_w, row_y + bar_h), (45, 45, 45), -1)

        # Filled portion
        filled = int(bar_area_w * min(prob, 1.0))
        if filled > 0:
            cv2.rectangle(frame, (bx, row_y), (bx + filled, row_y + bar_h), color, -1)

        # Thin highlight at top of bar
        if filled > 2:
            highlight = tuple(min(c + 60, 255) for c in color)
            cv2.line(frame, (bx, row_y), (bx + filled, row_y), highlight, 1)

        # Percentage text
        pct = f"{prob * 100:.0f}%"
        cv2.putText(
            frame, pct,
            (bx + bar_area_w + 4, row_y + bar_h - 2),
            FONT, 0.37, name_color if is_dominant else _TEXT_DIM, 1, cv2.LINE_AA,
        )


def _draw_history_strip(frame, emotion_history: deque, frame_w: int, frame_h: int) -> None:
    """
    Colored timeline strip along the bottom of the frame.
    Each slot = one inference result; color = emotion color.
    """
    strip_h = 20
    y0      = frame_h - strip_h

    # Background
    cv2.rectangle(frame, (0, y0), (frame_w, frame_h), (15, 15, 15), -1)

    n = len(emotion_history)
    if n == 0:
        return

    slot_w = max(1, frame_w // max(n, 1))

    for i, emotion in enumerate(emotion_history):
        x0 = i * slot_w
        x1 = min(x0 + slot_w - 1, frame_w - 1)
        color = EMOTION_COLORS.get(emotion, (50, 50, 50)) if emotion else (38, 38, 38)
        cv2.rectangle(frame, (x0, y0 + 3), (x1, frame_h - 3), color, -1)

    cv2.putText(
        frame, "HISTORY",
        (5, frame_h - 6),
        FONT, 0.33, (110, 110, 110), 1, cv2.LINE_AA,
    )


def _draw_hud(frame, fps: float, frame_count: int, inference_ms: float) -> None:
    """
    Top-left stats panel: FPS | Inference latency | Frame counter.
    Color-coded: green = good, yellow = ok, red = struggling.
    """
    # Semi-transparent background
    ovl = frame.copy()
    cv2.rectangle(ovl, (5, 5), (235, 88), _PANEL_BG, -1)
    cv2.addWeighted(ovl, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (5, 5), (235, 88), (60, 60, 60), 1)

    # FPS
    fps_color = _HUD_GREEN if fps >= 20 else (_HUD_YELLOW if fps >= 10 else _HUD_RED)
    cv2.putText(frame, f"FPS    {fps:5.1f}", (15, 32),
                FONT_BOLD, 0.60, fps_color, 1, cv2.LINE_AA)

    # Inference latency
    inf_color = _HUD_GREEN if inference_ms < 80 else (_HUD_YELLOW if inference_ms < 150 else _HUD_RED)
    inf_text = f"INFER  {inference_ms:5.0f} ms" if inference_ms > 0 else "INFER  --"
    cv2.putText(frame, inf_text, (15, 57),
                FONT, 0.52, inf_color, 1, cv2.LINE_AA)

    # Frame counter
    cv2.putText(frame, f"FRAME  {frame_count:6d}", (15, 78),
                FONT, 0.45, (110, 110, 110), 1, cv2.LINE_AA)


def _draw_no_face_message(frame, frame_w: int, frame_h: int) -> None:
    """Centered 'No face detected' message."""
    text = "NO FACE DETECTED"
    (tw, th), _ = cv2.getTextSize(text, FONT_BOLD, 0.9, 2)
    x = (frame_w - tw) // 2
    y = frame_h // 2

    # Shadow
    cv2.putText(frame, text, (x + 2, y + 2), FONT_BOLD, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    # Main
    cv2.putText(frame, text, (x, y), FONT_BOLD, 0.9, (60, 60, 220), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Multi-face overlay  (used by server.py web mode)
# ---------------------------------------------------------------------------

def draw_multi_face_overlay(
    frame,
    faces: list,            # List[FaceResult] from multi_face_pipeline
    fps: float,
    frame_count: int,
    inference_ms: float,
    emotion_history: deque,
    show_emotion_history: bool = True,
) -> None:
    """
    Draw overlays for all detected faces.
    Each face gets its own corner-bracket box, pill label, and face-ID badge.
    """
    h, w = frame.shape[:2]

    if not faces:
        _draw_no_face_message(frame, w, h)
    else:
        for face in faces:
            color = EMOTION_COLORS.get(face.label, _DEFAULT_COLOR) if face.label else _DEFAULT_COLOR
            if face.bbox:
                _draw_corner_box(frame, face.bbox, color)
                _draw_face_id_badge(frame, face.bbox, face.face_id, color)
                if face.label:
                    _draw_emotion_pill(frame, face.bbox, face.label, face.confidence, color, w, h)

    if show_emotion_history:
        _draw_history_strip(frame, emotion_history, w, h)

    _draw_hud(frame, fps, frame_count, inference_ms)

    # Face count badge (top-center)
    _draw_face_count(frame, len(faces), w)


def _draw_face_id_badge(frame, bbox: tuple, face_id: int, color: tuple) -> None:
    """Small ID chip in the bottom-right corner of the bounding box."""
    x, y, bw, bh = bbox
    text = f"#{face_id}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    rx = x + bw - tw - 8
    ry = y + bh + th + 4

    # Badge background
    cv2.rectangle(frame, (rx - 3, ry - th - 2), (rx + tw + 3, ry + 2), color, -1)
    cv2.putText(frame, text, (rx, ry - 1), FONT, 0.45, (10, 10, 10), 1, cv2.LINE_AA)


def _draw_face_count(frame, n_faces: int, frame_w: int) -> None:
    """Center-top badge showing number of detected faces."""
    text = f"{n_faces} FACE{'S' if n_faces != 1 else ''} DETECTED"
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    x = (frame_w - tw) // 2
    color = _HUD_GREEN if n_faces > 0 else (80, 80, 80)
    cv2.putText(frame, text, (x, 22), FONT, 0.45, color, 1, cv2.LINE_AA)

