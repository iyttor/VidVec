"""MSVD test split: 670 videos, about 41 captions each."""

import os
import pickle

from .common import EMB_ANSWER, RerankDataset, RetrievalDataset


def _load(video_path_prefix: str, test_video_path: str, captions_path: str):
    """Video paths, captions, and the video index of every caption."""
    with open(captions_path, "rb") as fh:
        captions = pickle.load(fh)
    with open(test_video_path) as fh:
        video_ids = [line.strip() for line in fh]
    videos, texts, text2video = [], [], []
    for index, video_id in enumerate(video_ids):
        videos.append(os.path.join(video_path_prefix, video_id + ".avi"))
        for cap in captions[video_id]:
            texts.append(" ".join(cap))
            text2video.append(index)
    return videos, texts, text2video


class MSVDDataset(RetrievalDataset):

    def __init__(self, video_path_prefix: str, test_video_path: str, captions_path: str,
                 split: str, video_prompt: str, text_prompt: str) -> None:
        super().__init__(split, video_prompt, text_prompt)
        self.videos, self.texts, self.text2video_gt_index = _load(
            video_path_prefix, test_video_path, captions_path
        )

    def num_videos(self) -> int:
        return len(self.videos)

    def video_conversation(self, index: int):
        # MSVD alone puts a newline before the video prompt.
        return [
            {"role": "user", "content": [
                {"type": "video", "video": self.videos[index], "fps": 2},
                {"type": "text", "text": f"\n{self.video_prompt}"},
            ]},
            EMB_ANSWER,
        ]


class MSVDRerankDataset(RerankDataset):

    def __init__(self, video_path_prefix: str, test_video_path: str, captions_path: str,
                 ret_query_data_path: str, ret_cand_data_path: str, rank_num: int) -> None:
        super().__init__(ret_query_data_path, ret_cand_data_path, rank_num)
        self.videos, self.texts, self.text2video_gt_index = _load(
            video_path_prefix, test_video_path, captions_path
        )

    def video_path(self, video_index: int) -> str:
        return self.videos[video_index]
