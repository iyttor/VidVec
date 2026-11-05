"""The exact conversations each benchmark feeds the model.

Every string here is part of the evaluation protocol -- a changed character changes
the reported numbers -- so the expected messages are spelled out in full. The
benchmarks are faked with tiny files; no data, model or GPU is needed.

    python -m pytest eval/tests/test_datasets.py -q
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from dataset.datasets_activitynet import ActivityNetDataset, ActivityNetRerankDataset  # noqa: E402
from dataset.datasets_didemo import DiDeMoDataset, DiDeMoRerankDataset  # noqa: E402
from dataset.datasets_msrvtt import MSRVTTDataset, MSRVTTRerankDataset  # noqa: E402
from dataset.datasets_msvd import MSVDDataset, MSVDRerankDataset  # noqa: E402
from dataset.datasets_vatex import VATEXDataset, VATEXRerankDataset  # noqa: E402
from vidvec.prompts import TEXT_PROMPT, VIDEO_PROMPT, normalize_media  # noqa: E402

EMB = {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]}
PROMPTS = dict(video_prompt=VIDEO_PROMPT, text_prompt=TEXT_PROMPT)


def text_conv(caption):
    return [{"role": "user", "content": [
        {"type": "text", "text": f"{caption}\nSummarize above sentence in one word: "}]}, EMB]


def video_conv(path, prompt="Summarize above video in one word: "):
    return [{"role": "user", "content": [
        {"type": "video", "video": path, "fps": 2},
        {"type": "text", "text": prompt}]}, EMB]


def rerank_conv(caption, path):
    return [{"role": "user", "content": [
        {"type": "text", "text": "Video: "},
        {"type": "video", "video": {"video_path": path, "fps": 2, "max_frames": 180}},
        {"type": "text", "text": (
            "Recover the main subject or subjects, appearance and setting, and main activity "
            f"in the video. Does the video match the caption: \"{caption}\"?\n"
            "Respond in a single word - Yes or No.")}]}]


def _rerank_files(tmp, queries, cands):
    q, c = os.path.join(tmp, "q.json"), os.path.join(tmp, "c.json")
    json.dump([str(x) for x in queries], open(q, "w"))
    json.dump([[str(x) for x in row] for row in cands], open(c, "w"))
    return dict(ret_query_data_path=q, ret_cand_data_path=c)


def test_msrvtt():
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "test.csv")
        open(csv, "w").write("key,vid_key,video_id,sentence\nk,v,video7,a man is cooking.\n"
                             "k,v,video9,dogs run\n")
        paths = dict(data_path=csv, video_data_path="/v")
        assert MSRVTTDataset(split="video", **paths, **PROMPTS)[1] == (video_conv("/v/video9.mp4"), 1)
        assert MSRVTTDataset(split="text", **paths, **PROMPTS)[0] == (text_conv("a man is cooking."), 0)
        rr = MSRVTTRerankDataset(**paths, rank_num=2, **_rerank_files(tmp, [0], [[1, 0]]))
        # MSR-VTT drops the closing period inside the quotes.
        assert rr[0][0] == rerank_conv("A man is cooking", "/v/video9.mp4")
        assert rr[1][0] == rerank_conv("A man is cooking", "/v/video7.mp4")


def test_msvd():
    with tempfile.TemporaryDirectory() as tmp:
        caps, lst = os.path.join(tmp, "caps.pkl"), os.path.join(tmp, "list.txt")
        pickle.dump({"a": [["a", "cat"], ["one", "cat"]], "b": [["a", "dog"]]}, open(caps, "wb"))
        open(lst, "w").write("a\nb\n")
        paths = dict(video_path_prefix="/v", test_video_path=lst, captions_path=caps)
        video = MSVDDataset(split="video", **paths, **PROMPTS)
        text = MSVDDataset(split="text", **paths, **PROMPTS)
        # MSVD puts a newline before the video prompt.
        assert video[0] == (video_conv("/v/a.avi", "\nSummarize above video in one word: "), 0)
        assert text[2] == (text_conv("a dog"), 2)
        assert text.text2video_gt_index == [0, 0, 1]
        rr = MSVDRerankDataset(**paths, rank_num=1, **_rerank_files(tmp, [1], [[1]]))
        assert rr[0][0] == rerank_conv("One cat.", "/v/b.avi")


def test_didemo():
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "d.json")
        json.dump([{"video": "test/x.mp4", "caption": "a girl jumps."}], open(data, "w"))
        paths = dict(data_path=data, video_data_path="/v")
        assert DiDeMoDataset(split="video", **paths, **PROMPTS)[0] == (video_conv("/v/x.mp4"), 0)
        assert DiDeMoDataset(split="text", **paths, **PROMPTS)[0] == (text_conv("a girl jumps."), 0)
        rr = DiDeMoRerankDataset(**paths, rank_num=1, **_rerank_files(tmp, [0], [[0]]))
        assert rr[0][0] == rerank_conv("A girl jumps.", "/v/x.mp4")


def test_activitynet_resolves_extensions_and_skips_missing_videos():
    with tempfile.TemporaryDirectory() as tmp:
        videos = os.path.join(tmp, "videos")
        os.makedirs(videos)
        open(os.path.join(videos, "v_a.mkv"), "w").close()
        data = os.path.join(tmp, "a.json")
        json.dump([{"video_id": "v_missing", "caption": "gone"},
                   {"video_id": "v_a", "caption": "people dance"}], open(data, "w"))
        paths = dict(data_path=data, video_data_path=videos)
        video = ActivityNetDataset(split="video", **paths, **PROMPTS)
        assert len(video) == 1 and video[0] == (video_conv(f"{videos}/v_a.mkv"), 0)
        assert ActivityNetDataset(split="text", **paths, **PROMPTS)[0] == (text_conv("people dance"), 0)
        rr = ActivityNetRerankDataset(**paths, rank_num=1, **_rerank_files(tmp, [1], [[1]]))
        assert rr[0][0] == rerank_conv("People dance.", f"{videos}/v_a.mkv")


def test_vatex_uses_the_last_1500_clips():
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "x.json")
        items = [{"videoID": f"id{i}", "enCap": [f"cap {i} a", f"cap {i} b"]} for i in range(1501)]
        json.dump(items, open(data, "w"))
        paths = dict(data_path=data, video_data_path="/v")
        video = VATEXDataset(split="video", **paths, **PROMPTS)
        text = VATEXDataset(split="text", **paths, **PROMPTS)
        assert len(video) == 1500 and video[0] == (video_conv("/v/id1.mp4"), 0)
        assert len(text) == 3000 and text[1] == (text_conv("cap 1 b"), 1)
        assert text.caption_to_video[3] == "id2"
        rr = VATEXRerankDataset(**paths, rank_num=1, **_rerank_files(tmp, [2], [[1]]))
        assert rr[0][0] == rerank_conv("Cap 2 a.", "/v/id2.mp4")


def test_custom_prompts_reach_the_conversation():
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "d.json")
        json.dump([{"video": "test/x.mp4", "caption": "c"}], open(data, "w"))
        paths = dict(data_path=data, video_data_path="/v")
        video = DiDeMoDataset(split="video", video_prompt="V? ", text_prompt="T? ", **paths)
        text = DiDeMoDataset(split="text", video_prompt="V? ", text_prompt="T? ", **paths)
        assert video[0][0][0]["content"][1]["text"] == "V? "
        assert text[0][0][0]["content"][0]["text"] == "c\nT? "


def test_normalize_media_applies_the_decode_settings():
    conv = normalize_media(video_conv("/v/x.mp4"), fps=1, max_frames=64)
    assert conv[0]["content"][0]["video"] == {"video_path": "/v/x.mp4", "fps": 1, "max_frames": 64}
    rerank = rerank_conv("c", "/v/x.mp4")
    assert normalize_media(rerank, fps=1, max_frames=64) == rerank  # reranking keeps 2 fps / 180
