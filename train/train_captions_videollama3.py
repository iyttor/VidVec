#!/usr/bin/env python3
"""Contrastive LoRA fine-tuning of VideoLLaMA3 on caption pairs -- the VidVec recipe.

Each row of the training CSV is one pair: a short ``Brief Caption`` (the anchor) and a
long ``Detailed Caption`` (the positive). Both sides are wrapped in the same "summarise
in one word" prompt and read out at the ``<emb>`` position, so text and video end up in
one embedding space (the video side uses the same prompt at evaluation time, with the
frames in place of the caption). The loss is Dual Softmax Loss over the in-batch
similarity matrix, with negatives gathered across data-parallel ranks -- which is why
the global batch size, not the per-device one, is the number that matters.

Launch it with ``scripts/train_vidvec.sh``, which pins the hyper-parameters behind the
released checkpoint.

Only LoRA adapters on the language model are trained. The vision encoder, the vision
projector and the base LLM weights stay frozen.
"""

import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
import transformers
import yaml
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState

from arguments import DataArguments, ModelArguments, TrainingArguments
from collators.train_collator import CaptionPairCollator
from dataset.datasets_captions import CaptionPairDataset
from vidvec.loading import BASE_MODEL, load_vidvec

# Parameter-name prefixes of the frozen multimodal components, and of the LLM.
VISION_ENCODER = ["model.vision_encoder"]
VISION_PROJECTOR = ["model.mm_projector"]
LLM = ["model"]

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
MODEL_MAX_LENGTH = 1024


def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any([module_name in name for module_name in target_modules]):
            continue

        if isinstance(module, cls):
            lora_module_names.add(name)

    for name in list(lora_module_names):
        if 'lm_head' in name: # needed for 16-bit
            lora_module_names.remove(name)

    return list(lora_module_names)


def rank0_print(*args):
    # Also prints when running single-process (dist uninitialised); the old
    # `if dist.is_initialized()` guard silenced the whole script in that case.
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


class StopAtStepCallback(TrainerCallback):
    """Stop after ``stop_at_step`` without altering the learning-rate schedule.

    The released checkpoint is step 210 of a schedule defined by ``num_train_epochs 2``
    over the full 1.09 M-row CSV, i.e. 7582 steps with 227 warmup steps. Passing
    ``--max_steps 210`` would reproduce the step count but not the schedule: warmup
    would collapse to 6 steps and the cosine decay would be nearly complete by the end.
    So the schedule is left alone and the run is truncated instead.

    This runs after ``DefaultFlowCallback`` has already set ``should_save`` for the
    step, so a checkpoint due at the stopping step is still written before the loop
    exits.
    """

    def __init__(self, stop_at_step: int):
        if stop_at_step <= 0:
            raise ValueError(f"stop_at_step must be positive, got {stop_at_step}")
        self.stop_at_step = stop_at_step

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step >= self.stop_at_step:
            rank0_print(f"Reached --stop_at_step {self.stop_at_step}; ending training.")
            control.should_training_stop = True
        return control


def lora_targets(model) -> list:
    """Every linear layer of the LLM, excluding the frozen vision encoder and projector."""
    frozen = VISION_ENCODER + VISION_PROJECTOR
    modules = find_all_linear_names(dict(model.named_modules()), LLM)
    return [
        m for m in modules
        if not any(m == k or m.startswith(k + ".") or ("." + k + ".") in m for k in frozen)
    ]


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    output_path = Path(training_args.output_dir)

    # Record the exact configuration next to the weights.
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        args_dir = output_path / "arguments"
        args_dir.mkdir(parents=True, exist_ok=True)
        yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
        yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
        yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    rank0_print(f"Loading {BASE_MODEL} ...")
    # DeepSpeed ZeRO-3 shards parameters as they are created, and transformers refuses
    # low_cpu_mem_usage=True in that mode.
    bundle = load_vidvec(
        BASE_MODEL,
        device_map=None,
        cache_dir=model_args.cache_dir,
        low_cpu_mem_usage=False,
    )
    model, tokenizer, processor = bundle.model, bundle.tokenizer, bundle.processor
    tokenizer.model_max_length = MODEL_MAX_LENGTH

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    for module in VISION_ENCODER + VISION_PROJECTOR:
        model.get_submodule(module).requires_grad_(False)

    model = get_peft_model(model, LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=lora_targets(model),
        modules_to_save=[],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    ))

    # Re-assert the freezes after PEFT wrapping, by parameter name. The module paths
    # used above no longer resolve once PeftModel is in the way, and a vision parameter
    # that comes back trainable would be updated in full precision by the optimiser.
    for name, param in model.named_parameters():
        if any(("." + key + ".") in ("." + name) for key in VISION_ENCODER + VISION_PROJECTOR):
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"{n_trainable:,} trainable parameters.")

    rank0_print(f"Loading caption pairs from {data_args.data_path} ...")
    train_dataset = CaptionPairDataset(data_args.data_path)
    rank0_print(f"{len(train_dataset):,} training pairs.")

    callbacks = []
    if training_args.stop_at_step:
        callbacks.append(StopAtStepCallback(training_args.stop_at_step))

    # use_reentrant=True is required for gradient checkpointing under DeepSpeed ZeRO-3
    # with PEFT; the non-reentrant path loses the adapter grads.
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=CaptionPairCollator(tokenizer=tokenizer, processor=processor),
        train_dataset=train_dataset,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
