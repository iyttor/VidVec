"""The retrieval benchmarks: their loaders, datasets and collator.

Each benchmark loads into a :class:`Benchmark`. :class:`RetrievalDataset` yields its
clips or captions to embed, and :class:`RerankDataset` one caption-video question per
top-K candidate. Every prompt string here is part of the evaluation protocol.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from .prompts import VIDEO_FPS, VIDEO_MAX_FRAMES, normalize_media

__all__ = ["Benchmark", "BENCHMARKS", "load_benchmark", "RetrievalDataset", "RerankDataset",
           "EvalDataCollator"]


# ---------------------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------------------


@dataclass
class Benchmark:
    videos: List[str]       # video file paths
    texts: List[str]        # captions
    text2video: List[int]   # index into ``videos`` of each caption's video
    video_prompt_prefix: str = ""      # MSVD alone puts a newline before the video prompt
    rerank_drop_period: bool = False   # MSR-VTT alone drops the closing period when reranking


def _msrvtt(root: str) -> Benchmark:
    """JSFUSION 1k-A test split: 1,000 videos, one caption each."""
    data = pd.read_csv(f"{root}/msrvtt/annotations/MSRVTT_JSFUSION_test.csv")
    videos = [f"{root}/msrvtt/videos/{v}.mp4" for v in data["video_id"]]
    return Benchmark(videos, data["sentence"].tolist(), list(range(len(videos))),
                     rerank_drop_period=True)


def _msvd(root: str) -> Benchmark:
    """Test split: 670 videos, about 41 captions each."""
    with open(f"{root}/MSVD/raw-captions.pkl", "rb") as fh:
        captions = pickle.load(fh)
    with open(f"{root}/MSVD/test_list.txt") as fh:
        video_ids = [line.strip() for line in fh]
    texts, text2video = [], []
    for index, video_id in enumerate(video_ids):
        for words in captions[video_id]:
            texts.append(" ".join(words))
            text2video.append(index)
    videos = [f"{root}/MSVD/YouTubeClips/{v}.avi" for v in video_ids]
    return Benchmark(videos, texts, text2video, video_prompt_prefix="\n")


def _didemo(root: str) -> Benchmark:
    """Test split, paragraph-to-video: 1,004 videos, one paragraph each."""
    with open(f"{root}/didemo/didemo_test.json") as fh:
        items = json.load(fh)
    # Video keys carry a "test/" prefix that is not part of the file name.
    videos = [f"{root}/didemo/raw_videos/" + it["video"].replace("test/", "") for it in items]
    return Benchmark(videos, [it["caption"] for it in items], list(range(len(videos))))


def _activitynet(root: str) -> Benchmark:
    """Captions val1, paragraph-to-video. Videos have mixed extensions."""
    with open(f"{root}/ActivityNet/activitynet_captions_val1.json") as fh:
        items = json.load(fh)
    video_dir = f"{root}/ActivityNet/Activity_Videos"
    on_disk = set(os.listdir(video_dir))
    videos = [f"{video_dir}/" + next(it["video_id"] + ext for ext in (".mp4", ".mkv", ".avi", ".webm")
                                     if it["video_id"] + ext in on_disk)
              for it in items]
    return Benchmark(videos, [it["caption"] for it in items], list(range(len(videos))))


def _vatex(root: str) -> Benchmark:
    """The last 1,500 clips of the public validation set, 10 English captions each."""
    with open(f"{root}/VATEX/jsons/vatex_validation_v1.0.json", encoding="utf-8") as fh:
        items = json.load(fh)[-1500:]
    texts, text2video = [], []
    for index, item in enumerate(items):
        texts.extend(item["enCap"])
        text2video.extend([index] * len(item["enCap"]))
    videos = [f"{root}/VATEX/val_all/{it['videoID']}.mp4" for it in items]
    return Benchmark(videos, texts, text2video)


BENCHMARKS = {"msrvtt": _msrvtt, "msvd": _msvd, "didemo": _didemo,
              "activitynet": _activitynet, "vatex": _vatex}


def load_benchmark(name: str) -> Benchmark:
    root = os.environ.get("VIDVEC_DATA_ROOT")
    if not root:
        raise SystemExit("Set VIDVEC_DATA_ROOT to the directory holding the benchmarks "
                         "(see 'Prepare the benchmarks' in the README).")
    return BENCHMARKS[name](root)


# ---------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------

EMB_ANSWER = {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]}

RERANK_INSTRUCTION = (
    "Recover the main subject or subjects, appearance and setting, and main activity in "
    "the video. Does the video match the caption: \"{caption}\"?\n"
    "Respond in a single word - Yes or No."
)


def format_caption(s: str) -> str:
    """Capitalise and end with punctuation, as the reranking prompt expects."""
    s = (s or "").replace("\r", "").strip().strip('"')
    if s:
        s = s[0].upper() + s[1:]
        if s[-1] not in [".", "?", "!"]:
            s = s + "."
    return s


class RetrievalDataset(Dataset):
    """One side of a benchmark to embed: its videos or its captions.

    Items are ``(conversation, index)``; the answer turn is ``<emb>``.
    """

    def __init__(self, bench: Benchmark, split: str, video_prompt: str, text_prompt: str) -> None:
        super().__init__()
        assert split in ("video", "text"), split
        self.bench = bench
        self.split = split
        self.video_prompt = bench.video_prompt_prefix + video_prompt
        self.text_prompt = text_prompt

    def __len__(self) -> int:
        return len(self.bench.videos if self.split == "video" else self.bench.texts)

    def __getitem__(self, index: int):
        if self.split == "video":
            content = [{"type": "video", "video": self.bench.videos[index], "fps": 2},
                       {"type": "text", "text": self.video_prompt}]
        else:
            content = [{"type": "text", "text": f"{self.bench.texts[index]}\n{self.text_prompt}"}]
        return [{"role": "user", "content": content}, EMB_ANSWER], index


class RerankDataset(Dataset):
    """Text-to-video reranking: one item per (caption, top-K candidate video) pair.

    ``queries`` lists the caption indices and ``candidates`` each caption's video
    indices, best first, as written by ``evaluate.py retrieve``. Item ``i`` is caption
    ``i // rank_num`` against its ``i % rank_num``-th candidate.
    """

    def __init__(self, bench: Benchmark, queries_path: str, candidates_path: str,
                 rank_num: int) -> None:
        super().__init__()
        self.bench = bench
        with open(queries_path) as fh:
            self.queries = [int(q) for q in json.load(fh)]
        with open(candidates_path) as fh:
            self.candidates = [[int(c) for c in row] for row in json.load(fh)]
        self.rank_num = rank_num

    def __len__(self) -> int:
        return len(self.queries) * self.rank_num

    def __getitem__(self, index: int):
        query, rank = divmod(index, self.rank_num)
        caption = format_caption(self.bench.texts[self.queries[query]])
        if self.bench.rerank_drop_period and caption.endswith("."):
            caption = caption[:-1]
        video = self.bench.videos[self.candidates[query][rank]]
        return [{"role": "user", "content": [
            {"type": "text", "text": "Video: "},
            {"type": "video", "video": {"video_path": video, "fps": 2, "max_frames": 180}},
            {"type": "text", "text": RERANK_INSTRUCTION.format(caption=caption)},
        ]}], index


# ---------------------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------------------


class EvalDataCollator:
    """Collate exactly ONE ``(conversation, id)`` item, for retrieval and reranking.

    VideoLLaMA3 cannot batch video (``VideoLLaMA3RetrievalWrapper`` asserts
    ``batch==1`` once ``pixel_values`` is set).
    """

    def __init__(self, processor, video_fps=VIDEO_FPS, video_max_frames=VIDEO_MAX_FRAMES):
        self.processor = processor
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames

    def __call__(self, items: Sequence) -> Dict[str, torch.Tensor]:
        assert len(items) == 1, f"EvalDataCollator handles one conversation at a time, got {len(items)}"
        conv, item_id = items[0]
        conv = normalize_media(conv, fps=self.video_fps, max_frames=self.video_max_frames)
        inputs = self.processor(conversation=conv, padding=True, return_tensors="pt")
        inputs["ids"] = [item_id]
        return inputs
