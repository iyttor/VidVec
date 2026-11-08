"""The evaluation protocol's constants, and the step that applies its decode settings.

Stdlib only -- no torch, no transformers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "EMB_TOKEN",
    "VIDEO_FPS",
    "VIDEO_MAX_FRAMES",
    "VIDEO_PROMPT",
    "TEXT_PROMPT",
    "normalize_media",
    "video_path_in",
]

#: Marker token appended in a trailing assistant turn. The embedding is read from
#: the hidden state at the position *before* it, so its input embedding vector is
#: never consumed -- only the id matters.
EMB_TOKEN = "<emb>"

#: Default decode settings, as in the paper (``--video-fps`` / ``--video-max-frames``).
VIDEO_FPS = 2
VIDEO_MAX_FRAMES = 180

#: Default prompts, as in the paper (``--video-prompt`` / ``--text-prompt``).
VIDEO_PROMPT = "Summarize above video in one word: "
TEXT_PROMPT = "Summarize above sentence in one word: "


def normalize_media(
    conversation: Sequence[Dict[str, Any]],
    *,
    fps: int = VIDEO_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
) -> List[Dict[str, Any]]:
    """Expand bare video paths into the dicts VideoLLaMA3's processor expects.

    ``{"type": "video", "video": "/p.mp4"}`` becomes
    ``{"type": "video", "video": {"video_path": "/p.mp4", "fps": .., "max_frames": ..}}``.
    The processor only reads ``content[content["type"]]``, so any sibling keys the
    caller left behind are inert. A video already given as a dict is left alone.
    """
    out: List[Dict[str, Any]] = []
    for msg in conversation:
        content = []
        for ele in msg["content"]:
            if ele.get("type") == "video" and isinstance(ele.get("video"), str):
                ele = {**ele, "video": {"video_path": ele["video"], "fps": fps, "max_frames": max_frames}}
            content.append(ele)
        out.append({**msg, "content": content})
    return out


def video_path_in(conversation: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Return the first video path in a normalized conversation."""
    for msg in conversation:
        for ele in msg["content"]:
            if ele.get("type") == "video":
                return ele["video"]["video_path"]
    return None
