"""DiDeMo test split, paragraph-to-video: 1,004 videos, one paragraph each."""

import json
import os

from .common import RerankDataset, RetrievalDataset, video_message


def _load(data_path: str):
    with open(data_path) as fh:
        items = json.load(fh)
    # Video keys carry a "test/" prefix that is not part of the file name.
    return [it["video"].replace("test/", "") for it in items], [it["caption"] for it in items]


class DiDeMoDataset(RetrievalDataset):

    def __init__(self, data_path: str, video_data_path: str, split: str,
                 video_prompt: str, text_prompt: str) -> None:
        super().__init__(split, video_prompt, text_prompt)
        self.video_data_path = video_data_path
        self.videos, self.texts = _load(data_path)
        self.caption_to_video = {i: i for i in range(len(self.texts))}

    def num_videos(self) -> int:
        return len(self.videos)

    def video_conversation(self, index: int):
        return video_message(os.path.join(self.video_data_path, self.videos[index]), self.video_prompt)


class DiDeMoRerankDataset(RerankDataset):

    def __init__(self, data_path: str, video_data_path: str,
                 ret_query_data_path: str, ret_cand_data_path: str, rank_num: int) -> None:
        super().__init__(ret_query_data_path, ret_cand_data_path, rank_num)
        self.video_data_path = video_data_path
        self.videos, self.texts = _load(data_path)
        self.text2video_gt_index = list(range(len(self.texts)))

    def video_path(self, video_index: int) -> str:
        return os.path.join(self.video_data_path, self.videos[video_index])
