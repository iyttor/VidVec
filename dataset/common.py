"""Conversation builders shared by the benchmark datasets.

Every string here is part of the evaluation protocol: changing a character changes
the numbers. The prompts themselves (``VIDEO_PROMPT`` / ``TEXT_PROMPT``) are passed in
by the caller so that ``scripts/eval.sh --video-prompt/--text-prompt`` reach them.
"""

import json
from typing import Dict, List

from torch.utils.data import Dataset

EMB_ANSWER = {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]}

RERANK_INSTRUCTION = (
    "Recover the main subject or subjects, appearance and setting, and main activity in "
    "the video. Does the video match the caption: \"{caption}\"?\n"
    "Respond in a single word - Yes or No."
)


def text_message(text: str, text_prompt: str) -> List[Dict]:
    """A caption to embed: the caption, the prompt, then ``<emb>`` as the answer."""
    return [
        {"role": "user", "content": [{"type": "text", "text": f"{text}\n{text_prompt}"}]},
        EMB_ANSWER,
    ]


def video_message(video_path: str, video_prompt: str) -> List[Dict]:
    """A video to embed: the clip, the prompt, then ``<emb>`` as the answer."""
    return [
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": 2},
            {"type": "text", "text": video_prompt},
        ]},
        EMB_ANSWER,
    ]


def format_caption(s: str) -> str:
    """Capitalise and end with punctuation, as the reranking prompt expects."""
    s = (s or "").replace("\r", "").strip().strip('"')
    if s:
        s = s[0].upper() + s[1:]
        if s[-1] not in [".", "?", "!"]:
            s = s + "."
    return s


def rerank_message(caption: str, video_path: str) -> List[Dict]:
    """Ask the MLLM whether ``video_path`` matches ``caption``; scored by P(Yes)."""
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "Video: "},
            {"type": "video", "video": {"video_path": video_path, "fps": 2, "max_frames": 180}},
            {"type": "text", "text": RERANK_INSTRUCTION.format(caption=caption)},
        ]},
    ]


class RetrievalDataset(Dataset):
    """One benchmark, one side of it: ``split="video"`` yields the clips to embed and
    ``split="text"`` the captions. Items are ``(conversation, index)``.

    Subclasses fill ``self.texts`` and implement :meth:`video_conversation` and
    :meth:`num_videos`.
    """

    def __init__(self, split: str, video_prompt: str, text_prompt: str) -> None:
        super().__init__()
        if split not in ("video", "text"):
            raise ValueError(f"split must be 'video' or 'text', got {split!r}")
        self.split = split
        self.video_prompt = video_prompt
        self.text_prompt = text_prompt
        self.texts: List[str] = []

    def num_videos(self) -> int:
        raise NotImplementedError

    def video_conversation(self, index: int) -> List[Dict]:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.num_videos() if self.split == "video" else len(self.texts)

    def __getitem__(self, index: int):
        if self.split == "video":
            return self.video_conversation(index), index
        return text_message(self.texts[index], self.text_prompt), index


class RerankDataset(Dataset):
    """Text-to-video reranking: one item per (caption, top-K candidate video) pair.

    ``ret_query_data`` lists the caption indices and ``ret_cand_data`` each caption's
    retrieved video indices, best first, as written by ``eval/run_retrieval_eval.py``.
    Item ``i`` is caption ``i // rank_num`` against its ``i % rank_num``-th candidate.
    Subclasses fill ``self.texts`` and ``self.text2video_gt_index`` and implement
    :meth:`video_path`.
    """

    def __init__(self, ret_query_data_path: str, ret_cand_data_path: str, rank_num: int) -> None:
        super().__init__()
        with open(ret_query_data_path) as fh:
            self.ret_query_data = [int(q) for q in json.load(fh)]
        with open(ret_cand_data_path) as fh:
            self.ret_cand_data = [[int(c) for c in row] for row in json.load(fh)]
        self.rank_num = rank_num
        self.texts: List[str] = []
        self.text2video_gt_index: List[int] = []

    def video_path(self, video_index: int) -> str:
        raise NotImplementedError

    def query_caption(self, text_index: int) -> str:
        return format_caption(self.texts[text_index])

    def __len__(self) -> int:
        return len(self.ret_query_data) * self.rank_num

    def __getitem__(self, index: int):
        query_id = self.ret_query_data[index // self.rank_num]
        cand_id = self.ret_cand_data[index // self.rank_num][index % self.rank_num]
        return rerank_message(self.query_caption(query_id), self.video_path(cand_id)), index
