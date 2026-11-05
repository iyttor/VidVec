"""Running VideoLLaMA3's processor over one conversation, with the short-video retry.

The eval collator calls :func:`process_conversation`. The Hub model's
``modeling_vidvec.py`` carries a copy of it, so a clip that needs the fallback
behaves the same in a benchmark run and through ``AutoModel``.

Note on decoding: ``.mp4`` goes through the **ffmpeg binary** via ``ffmpeg-python``
inside ``processing_videollama3.py``; decord is only used for frame directories,
GIFs, and sub-second clips. ``ffmpeg`` must be on PATH.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .prompts import VIDEO_FPS, VIDEO_MAX_FRAMES, normalize_media, video_path_in

__all__ = ["probe_duration_seconds", "process_conversation"]


def probe_duration_seconds(video_path: str) -> Optional[float]:
    """Duration in seconds via OpenCV, or None if it cannot be determined."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count / fps if fps and fps > 0 else None
    finally:
        cap.release()


def _force_single_frame(conversation: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in conversation:
        if not (isinstance(msg, dict) and isinstance(msg.get("content"), list)):
            out.append(msg)
            continue
        content = []
        for ele in msg["content"]:
            if isinstance(ele, dict) and ele.get("type") == "video":
                spec = ele.get("video") or {}
                ele = {
                    **ele,
                    "video": {
                        "video_path": spec.get("video_path", ""),
                        "fps": 2,
                        "max_frames": 1,
                    },
                }
            content.append(ele)
        out.append({**msg, "content": content})
    return out


def process_conversation(
    processor,
    conversation: Sequence[Dict[str, Any]],
    *,
    fps: int = VIDEO_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
    already_normalized: bool = False,
):
    """Tokenize + decode one conversation into model inputs.

    Videos shorter than a second sample zero frames and raise ``IndexError`` deep
    inside the processor; those retry with a single frame. Any other ``IndexError``
    is re-raised with the probed duration attached, because it means something
    other than a short clip went wrong.
    """
    conv = conversation if already_normalized else normalize_media(
        conversation, fps=fps, max_frames=max_frames
    )

    try:
        return processor(conversation=conv, padding=True, return_tensors="pt")
    except IndexError:
        video_path = video_path_in(conv)
        duration = probe_duration_seconds(video_path) if video_path else None

        if duration is None or duration >= 1.0:
            raise IndexError(
                f"Processor raised IndexError but the video duration is {duration}s "
                f"(expected < 1.0s for the short-clip fallback). Video: {video_path}"
            ) from None

        print(f"[vidvec] Video is {duration:.3f}s (< 1s); retrying with fps=2, max_frames=1")
        return processor(
            conversation=_force_single_frame(conv), padding=True, return_tensors="pt"
        )
