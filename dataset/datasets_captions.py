from typing import Dict, List, Tuple

import pandas as pd
from torch.utils.data import Dataset


def caption_message(text: str, dense: bool) -> List[Dict]:
    """Wrap one caption in the prompt whose ``<emb>`` position becomes the embedding.

    The two prompts are the evaluation defaults (``vidvec.prompts``):

      dense=False  "Summarize above sentence in one word: "  -- the brief caption,
                   and at eval time the query text.
      dense=True   "Summarize above video in one word: "     -- the detailed
                   caption, which stands in for the video during training and is
                   replaced by the actual frames at eval time.
    """
    return [
        {"role": "user", "content": [
            {"type": "text", "text": f"{text}\nSummarize above {'video' if dense else 'sentence'} in one word: "}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
    ]


class CaptionPairDataset(Dataset):
    """VideoUFO caption pairs: ``Brief Caption`` (anchor) and ``Detailed Caption`` (positive).

    Rows missing either caption are dropped.
    """

    def __init__(self, data_path: str) -> None:
        super().__init__()
        csv = pd.read_csv(data_path, low_memory=False)
        missing = {"Brief Caption", "Detailed Caption"} - set(csv.columns)
        if missing:
            raise ValueError(f"{data_path} lacks the VideoUFO column(s) {sorted(missing)}")
        self.csv = csv.dropna(subset=["Brief Caption", "Detailed Caption"], how="any").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, index: int) -> Tuple[List[Dict], List[Dict]]:
        brief = self.csv["Brief Caption"][index]
        detailed = self.csv["Detailed Caption"][index]
        return caption_message(brief, dense=False), caption_message(detailed, dense=True)
