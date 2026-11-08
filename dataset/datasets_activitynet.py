"""ActivityNet Captions val1, paragraph-to-video: one paragraph per video.

Videos are stored with mixed extensions, so each id is resolved against the files
actually on disk. Retrieval skips videos that are missing.
"""

import json
import os

from .common import RerankDataset, RetrievalDataset, video_message

EXTENSIONS = (".mp4", ".mkv", ".avi", ".webm")


def _load(data_path: str, video_data_path: str):
    """``(video_id, caption)`` pairs, and the extension of every video found on disk."""
    with open(data_path) as fh:
        pairs = [(it["video_id"], it["caption"]) for it in json.load(fh)]
    on_disk = set(os.listdir(video_data_path))
    extensions = {}
    for video_id, _ in pairs:
        for ext in EXTENSIONS:
            if video_id + ext in on_disk:
                extensions[video_id] = ext
                break
    return pairs, extensions


class ActivityNetDataset(RetrievalDataset):

    def __init__(self, data_path: str, video_data_path: str, split: str,
                 video_prompt: str, text_prompt: str) -> None:
        super().__init__(split, video_prompt, text_prompt)
        pairs, self.video_extensions = _load(data_path, video_data_path)
        kept = [(v, t) for v, t in pairs if v in self.video_extensions]
        if len(kept) < len(pairs):
            print(f"[activitynet] {len(pairs) - len(kept)} videos not found in {video_data_path}; skipped")
        self.video_data_path = video_data_path
        self.videos = [v for v, _ in kept]
        self.texts = [t for _, t in kept]
        self.caption_to_video = {i: i for i in range(len(self.texts))}

    def num_videos(self) -> int:
        return len(self.videos)

    def video_conversation(self, index: int):
        video_id = self.videos[index]
        path = os.path.join(self.video_data_path, video_id + self.video_extensions[video_id])
        return video_message(path, self.video_prompt)


class ActivityNetRerankDataset(RerankDataset):

    def __init__(self, data_path: str, video_data_path: str,
                 ret_query_data_path: str, ret_cand_data_path: str, rank_num: int) -> None:
        super().__init__(ret_query_data_path, ret_cand_data_path, rank_num)
        pairs, self.video_extensions = _load(data_path, video_data_path)
        self.video_data_path = video_data_path
        self.videos = [v for v, _ in pairs]
        self.texts = [t for _, t in pairs]
        self.text2video_gt_index = list(range(len(self.texts)))

    def video_path(self, video_index: int) -> str:
        video_id = self.videos[video_index]
        if video_id.endswith(EXTENSIONS):
            return os.path.join(self.video_data_path, video_id)
        return os.path.join(self.video_data_path, video_id + self.video_extensions.get(video_id, ".mp4"))
