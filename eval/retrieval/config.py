"""Resolved configuration for one evaluation run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from vidvec.prompts import TEXT_PROMPT, VIDEO_FPS, VIDEO_MAX_FRAMES, VIDEO_PROMPT

__all__ = ["DEFAULT_CKPT", "EvalConfig", "add_arguments", "build_config"]

DEFAULT_CKPT = "iyttor/VidVec-7B"


@dataclass
class EvalConfig:
    dataset: str
    ckpt: str
    video_prompt: str
    text_prompt: str
    video_fps: int
    video_max_frames: int
    results_json: Optional[str]
    rerank_out_dir: Optional[str]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Hugging Face model id")
    parser.add_argument("--video-prompt", default=VIDEO_PROMPT)
    parser.add_argument("--text-prompt", default=TEXT_PROMPT)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--video-max-frames", type=int, default=VIDEO_MAX_FRAMES)
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--rerank-out-dir", default=None,
                        help="write the text-to-video candidate lists here for reranking")


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        dataset=args.dataset,
        ckpt=args.ckpt,
        video_prompt=args.video_prompt,
        text_prompt=args.text_prompt,
        video_fps=args.video_fps,
        video_max_frames=args.video_max_frames,
        results_json=args.results_json,
        rerank_out_dir=args.rerank_out_dir,
    )
