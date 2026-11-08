"""Argument dataclasses for ``train/train_captions_videollama3.py``.

``TrainingArguments`` extends the HuggingFace one, so ``scripts/train_vidvec.sh``
passes the stock optimisation flags (``--learning_rate``, ``--deepspeed``, ...)
through unchanged; only the VidVec-specific additions are declared here.
"""

from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to cache the base model. Unset uses the HF default."}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to VideoUFO.csv."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    stop_at_step: Optional[int] = field(
        default=None,
        metadata={"help": "Stop training once this optimiser step completes, without changing "
                          "the LR schedule. The schedule is still built from num_train_epochs, "
                          "so this truncates a run rather than redefining it -- which is how the "
                          "released checkpoint (step 210 of a 7582-step cosine schedule) is "
                          "reproduced."}
    )

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False
