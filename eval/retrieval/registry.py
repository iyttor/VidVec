"""The benchmarks: where their files live under ``$VIDVEC_DATA_ROOT``, which dataset
classes read them, and how captions are paired with videos.

One trap worth naming: for VATEX the video dataset yields 1,500 items in
``video_to_captions`` key order, while there are 15,000 captions. Its video keys must
therefore read that dict.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Sequence, Tuple

from dataset.datasets_activitynet import ActivityNetDataset, ActivityNetRerankDataset
from dataset.datasets_didemo import DiDeMoDataset, DiDeMoRerankDataset
from dataset.datasets_msrvtt import MSRVTTDataset, MSRVTTRerankDataset
from dataset.datasets_msvd import MSVDDataset, MSVDRerankDataset
from dataset.datasets_vatex import VATEXDataset, VATEXRerankDataset

__all__ = ["PairSpec", "DatasetSpec", "DATASETS", "get_spec", "list_datasets",
           "data_root", "dataset_paths", "build_datasets"]


@dataclass(frozen=True)
class PairSpec:
    """How to decide which caption belongs to which video.

    ``video_keys(ds)`` returns one hashable key per *video dataset index*;
    ``text_keys(ds)`` returns, per *text dataset index*, the key of its video.
    """

    kind: str  # "diagonal" | "gt_index" | "caption_map"
    video_keys: Callable[[Any], Sequence[Hashable]]
    text_keys: Callable[[Any], Sequence[Hashable]]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    retrieval_cls: type
    rerank_cls: type
    paths: Dict[str, str]  # constructor argument -> path relative to the data root
    pairs: PairSpec
    expected_sizes: Tuple[int, int]  # (videos, texts), advisory


def _identity_keys(ds) -> Sequence[Hashable]:
    return range(len(ds))


def _caption_map_text_keys(ds) -> List[Hashable]:
    return [ds.caption_to_video[i] for i in range(len(ds.texts))]


def _vatex_video_keys(ds) -> List[Hashable]:
    return list(ds.video_to_captions.keys())


DIAGONAL = PairSpec("diagonal", _identity_keys, _identity_keys)
CAPTION_MAP = PairSpec("caption_map", _identity_keys, _caption_map_text_keys)


DATASETS: Dict[str, DatasetSpec] = {
    "msrvtt": DatasetSpec(
        name="msrvtt",
        retrieval_cls=MSRVTTDataset,
        rerank_cls=MSRVTTRerankDataset,
        paths={"data_path": "msrvtt/annotations/MSRVTT_JSFUSION_test.csv",
               "video_data_path": "msrvtt/videos"},
        pairs=DIAGONAL,
        expected_sizes=(1000, 1000),
    ),
    "msvd": DatasetSpec(
        name="msvd",
        retrieval_cls=MSVDDataset,
        rerank_cls=MSVDRerankDataset,
        paths={"video_path_prefix": "MSVD/YouTubeClips",
               "test_video_path": "MSVD/test_list.txt",
               "captions_path": "MSVD/raw-captions.pkl"},
        pairs=PairSpec("gt_index", _identity_keys, lambda ds: list(ds.text2video_gt_index)),
        expected_sizes=(670, 27763),
    ),
    "didemo": DatasetSpec(
        name="didemo",
        retrieval_cls=DiDeMoDataset,
        rerank_cls=DiDeMoRerankDataset,
        paths={"data_path": "didemo/didemo_test.json",
               "video_data_path": "didemo/raw_videos"},
        pairs=CAPTION_MAP,
        expected_sizes=(1004, 1004),
    ),
    "activitynet": DatasetSpec(
        name="activitynet",
        retrieval_cls=ActivityNetDataset,
        rerank_cls=ActivityNetRerankDataset,
        paths={"data_path": "ActivityNet/activitynet_captions_val1.json",
               "video_data_path": "ActivityNet/Activity_Videos"},
        pairs=CAPTION_MAP,
        expected_sizes=(4917, 4917),
    ),
    "vatex": DatasetSpec(
        name="vatex",
        retrieval_cls=VATEXDataset,
        rerank_cls=VATEXRerankDataset,
        paths={"data_path": "VATEX/jsons/vatex_validation_v1.0.json",
               "video_data_path": "VATEX/val_all"},
        pairs=PairSpec("caption_map", _vatex_video_keys, _caption_map_text_keys),
        expected_sizes=(1500, 15000),
    ),
}


def list_datasets() -> List[str]:
    return list(DATASETS)


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise SystemExit(
            f"Unknown dataset {name!r}. Available: {', '.join(list_datasets())}"
        ) from None


def data_root() -> str:
    root = os.environ.get("VIDVEC_DATA_ROOT")
    if not root:
        raise SystemExit("Set VIDVEC_DATA_ROOT to the directory holding the benchmarks "
                         "(see 'Prepare the benchmarks' in the README).")
    return root


def dataset_paths(spec: DatasetSpec) -> Dict[str, str]:
    """Absolute paths for ``spec``, checked to exist."""
    root = data_root()
    paths = {key: f"{root}/{rel}" for key, rel in spec.paths.items()}
    for key, path in paths.items():
        if not os.path.exists(path):
            raise SystemExit(f"[{spec.name}] {key} does not exist: {path}")
    return paths


def build_datasets(spec: DatasetSpec, video_prompt: str, text_prompt: str):
    """Instantiate the (video, text) dataset pair for one benchmark."""
    paths = dataset_paths(spec)
    video_ds = spec.retrieval_cls(split="video", video_prompt=video_prompt,
                                  text_prompt=text_prompt, **paths)
    text_ds = spec.retrieval_cls(split="text", video_prompt=video_prompt,
                                 text_prompt=text_prompt, **paths)
    if (len(video_ds), len(text_ds)) != spec.expected_sizes:
        print(f"[{spec.name}] size differs from the published run: got "
              f"{len(video_ds)}/{len(text_ds)}, expected {spec.expected_sizes[0]}/{spec.expected_sizes[1]}")
    return video_ds, text_ds
