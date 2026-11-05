from typing import Dict, Sequence

import torch

from vidvec.media import process_conversation
from vidvec.prompts import VIDEO_FPS, VIDEO_MAX_FRAMES, normalize_media

IGNORE_TOKEN_ID = -100


class EvalDataCollator:
    """Collate exactly ONE ``(conversation, id)`` item.

    VideoLLaMA3 cannot batch video (``videollama3_ret.py`` asserts ``batch==1``
    once ``pixel_values`` is set).
    """

    def __init__(self, tokenizer, processor, video_fps=VIDEO_FPS, video_max_frames=VIDEO_MAX_FRAMES):
        self.tokenizer = tokenizer
        self.processor = processor
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames

    def __call__(self, items: Sequence) -> Dict[str, torch.Tensor]:
        assert len(items) == 1, f"EvalDataCollator handles one conversation at a time, got {len(items)}"
        conv, item_id = items[0]

        norm_conv = normalize_media(conv, fps=self.video_fps, max_frames=self.video_max_frames)
        inputs = process_conversation(self.processor, norm_conv, already_normalized=True)

        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        inputs["labels"] = labels
        inputs["ids"] = [item_id]
        return inputs
