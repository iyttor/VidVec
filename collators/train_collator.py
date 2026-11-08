from typing import Dict, List, Sequence

import torch

IGNORE_TOKEN_ID = -100


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
