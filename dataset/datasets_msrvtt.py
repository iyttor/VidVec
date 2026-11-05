"""MSR-VTT, JSFUSION 1k-A test split: 1,000 videos, one caption each."""

import os

import pandas as pd

from .common import RerankDataset, RetrievalDataset, format_caption, video_message


class MSRVTTDataset(RetrievalDataset):

    def __init__(self, data_path: str, video_data_path: str, split: str,
                 video_prompt: str, text_prompt: str) -> None:
        super().__init__(split, video_prompt, text_prompt)
        data = pd.read_csv(data_path)
        self.video_data_path = video_data_path
        self.videos = data["video_id"].tolist()
        self.texts = data["sentence"].tolist()

    def num_videos(self) -> int:
        return len(self.videos)

    def video_conversation(self, index: int):
        path = os.path.join(self.video_data_path, self.videos[index] + ".mp4")
        return video_message(path, self.video_prompt)


class MSRVTTRerankDataset(RerankDataset):

    def __init__(self, data_path: str, video_data_path: str,
                 ret_query_data_path: str, ret_cand_data_path: str, rank_num: int) -> None:
        super().__init__(ret_query_data_path, ret_cand_data_path, rank_num)
        data = pd.read_csv(data_path)
        self.video_data_path = video_data_path
        self.videos = data["video_id"].tolist()
        self.texts = data["sentence"].tolist()
        self.text2video_gt_index = list(range(len(self.texts)))

    def query_caption(self, text_index: int) -> str:
        # MSR-VTT alone drops the closing period inside the quotes.
        caption = format_caption(self.texts[text_index])
        return caption[:-1] if caption[-1] == "." else caption

    def video_path(self, video_index: int) -> str:
        return os.path.join(self.video_data_path, self.videos[video_index] + ".mp4")
