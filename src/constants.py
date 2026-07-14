"""
constants.py

Single source of truth for shared constants used across multiple modules.
Import from here — never re-define these lists in other files.
"""

# Canonical emotion list — order matches the web dashboard display order
# and the CSV column order written by session_recorder.py.
EMOTIONS: list[str] = [
    "happy",
    "neutral",
    "surprise",
    "sad",
    "angry",
    "fear",
    "disgust",
]
