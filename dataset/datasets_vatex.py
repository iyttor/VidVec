"""VATEX: the last 1,500 clips of the public validation set, 10 English captions each."""

import json
import os

from .common import RerankDataset, RetrievalDataset, video_message


def _load(data_path: str):
    """Video ids in file order, captions, and the video index of every caption."""
    with open(data_path, encoding="utf-8") as fh:
        items = json.load(fh)[-1500:]
    video_ids, texts, text2video = [], [], []
    for index, item in enumerate(items):
        video_ids.append(item["videoID"])
        for caption in item["enCap"]:
            texts.append(caption)
            text2video.append(index)
    return video_ids, texts, text2video


class VATEXDataset(RetrievalDataset):

    def __init__(self, data_path: str, video_data_path: str, split: str,
                 video_prompt: str, text_prompt: str) -> None:
        super().__init__(split, video_prompt, text_prompt)
        self.video_data_path = video_data_path
        self.video_ids, self.texts, text2video = _load(data_path)
        # Captions are paired with videos through their id, as the registry expects.
        self.video_to_captions = {v: [] for v in self.video_ids}
        self.caption_to_video = {}
        for caption_index, video_index in enumerate(text2video):
            video_id = self.video_ids[video_index]
            self.video_to_captions[video_id].append(caption_index)
            self.caption_to_video[caption_index] = video_id

    def num_videos(self) -> int:
        return len(self.video_to_captions)

    def video_conversation(self, index: int):
        video_id = list(self.video_to_captions)[index]
        return video_message(os.path.join(self.video_data_path, video_id + ".mp4"), self.video_prompt)


class VATEXRerankDataset(RerankDataset):

    def __init__(self, data_path: str, video_data_path: str,
                 ret_query_data_path: str, ret_cand_data_path: str, rank_num: int) -> None:
        super().__init__(ret_query_data_path, ret_cand_data_path, rank_num)
        self.video_data_path = video_data_path
        self.video_ids, self.texts, self.text2video_gt_index = _load(data_path)

    def video_path(self, video_index: int) -> str:
        return os.path.join(self.video_data_path, self.video_ids[video_index] + ".mp4")
