"""The training set: VideoUFO caption pairs and their collator."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = [
    "IGNORE_TOKEN_ID",
    "caption_message",
    "CaptionPairDataset",
    "CaptionPairCollator",
]

IGNORE_TOKEN_ID = -100


# ---------------------------------------------------------------------------------------
# Training data: VideoUFO caption pairs
# ---------------------------------------------------------------------------------------


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
        self.csv = csv.dropna(subset=["Brief Caption", "Detailed Caption"], how="any").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, index: int) -> Tuple[List[Dict], List[Dict]]:
        brief = self.csv["Brief Caption"][index]
        detailed = self.csv["Detailed Caption"][index]
        return caption_message(brief, dense=False), caption_message(detailed, dense=True)


class CaptionPairCollator:
    """Pack (brief caption, detailed caption) conversations for contrastive training.

    The batch is all anchors followed by all positives, the order
    ``VideoLLaMA3RetrievalWrapper`` splits it in. Each conversation is tokenised on
    its own, then right-padded.
    """

    def __init__(self, tokenizer, processor) -> None:
        self.tokenizer = tokenizer
        self.processor = processor

    def __call__(self, pairs: Sequence[Sequence[List[Dict]]]) -> Dict[str, torch.Tensor]:
        conversations = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
        encoded = [
            self.processor(
                conversation=conv,
                add_system_prompt=False,
                add_generation_prompt=False,
                return_tensors="pt",
            )["input_ids"][0]
            for conv in conversations
        ]

        pad_id = self.tokenizer.pad_token_id
        max_len = max(ids.size(0) for ids in encoded)
        input_ids = torch.full((len(encoded), max_len), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for i, ids in enumerate(encoded):
            input_ids[i, : ids.size(0)] = ids
            attention_mask[i, : ids.size(0)] = 1

        labels = input_ids.clone()
        labels[labels == pad_id] = IGNORE_TOKEN_ID
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
