"""``scripts/eval.sh`` bookkeeping: which stages run, resume, failures, the summary.

The real script and ``merge_results.py`` run in a scratch copy of the repository whose
two GPU stages are replaced by stubs that write the same files instantly. No model,
data or GPU is needed.

    python -m pytest eval/tests/test_eval_script.py -q
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RETRIEVAL_STUB = r'''
import argparse, json, os, sys
ap = argparse.ArgumentParser()
for a in ("--dataset", "--ckpt", "--results-json", "--rerank-out-dir"):
    ap.add_argument(a)
args, _ = ap.parse_known_args()
open(os.environ["CALLS"], "a").write(f"retrieval {args.dataset}\n")
if args.dataset in os.environ.get("FAIL_RETRIEVAL", "").split():
    sys.exit(1)
r = lambda x: {"R@1": x, "R@5": x + 0.2, "R@10": x + 0.3, "median_rank": 1.0}
json.dump({"status": "ok", "dataset": args.dataset, "model": {"ckpt": args.ckpt},
           "results": {"t2v": {"scaled": r(0.5)}, "v2t": {"scaled": r(0.6)}}},
          open(args.results_json, "w"))
os.makedirs(args.rerank_out_dir, exist_ok=True)
for kind in ("query", "cand"):
    json.dump([], open(os.path.join(args.rerank_out_dir, f"{args.dataset}_t2v_{kind}_names.json"), "w"))
'''

RERANK_STUB = r'''
import argparse, json, os, sys
ap = argparse.ArgumentParser()
for a in ("--dataset", "--rank-num", "--cands-dir", "--out-dir"):
    ap.add_argument(a)
args = ap.parse_args()
open(os.environ["CALLS"], "a").write(f"rerank {args.dataset} {args.rank_num}\n")
# Stage 1 must already have saved the VidVec-O results.
summary = os.path.join(os.path.dirname(args.out_dir), "results.json")
saved = json.load(open(summary))["datasets"] if os.path.exists(summary) else {}
open(os.environ["CALLS"], "a").write(f"saved {' '.join(sorted(saved))}\n")
if args.dataset in os.environ.get("FAIL_RERANK", "").split():
    sys.exit(1)
k = int(args.rank_num)
m = {"R@1": 60.0 + k, "R@5": 70.0, "R@10": 80.0}
json.dump({"before_rerank": m, "after_rerank": m, "rank_num": k, "dataset": args.dataset},
          open(os.path.join(args.out_dir, f"{args.dataset}_rerank.json"), "w"))
json.dump({}, open(os.path.join(args.out_dir, f"{args.dataset}_rerank_scores.json"), "w"))
'''

# accelerate launch --num_processes 1 --main_process_port N SCRIPT ARGS...
ACCELERATE_STUB = "#!/bin/bash\nshift 5\nexec python \"$@\"\n"


class Repo:
    def __init__(self, tmp):
        self.root = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(self.root, "scripts"))
        os.makedirs(os.path.join(self.root, "eval"))
        shutil.copy(os.path.join(CODE_ROOT, "scripts", "eval.sh"), os.path.join(self.root, "scripts"))
        shutil.copy(os.path.join(CODE_ROOT, "eval", "merge_results.py"), os.path.join(self.root, "eval"))
        open(os.path.join(self.root, "eval", "run_retrieval_eval.py"), "w").write(RETRIEVAL_STUB)
        open(os.path.join(self.root, "eval", "run_rerank_eval.py"), "w").write(RERANK_STUB)
        bin_dir = os.path.join(tmp, "bin")
        os.makedirs(bin_dir)
        acc = os.path.join(bin_dir, "accelerate")
        open(acc, "w").write(ACCELERATE_STUB)
        os.chmod(acc, os.stat(acc).st_mode | stat.S_IEXEC)
        self.out = os.path.join(tmp, "out")
        self.calls = os.path.join(tmp, "calls.txt")
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "VIDVEC_DATA_ROOT": tmp, "CALLS": self.calls}

    def run(self, *args, **env):
        open(self.calls, "w").close()
        proc = subprocess.run(["bash", os.path.join(self.root, "scripts", "eval.sh"),
                               "--out", self.out, *args],
                              capture_output=True, text=True, env={**self.env, **env})
        calls = open(self.calls).read().split("\n")[:-1]
        return proc, calls

    def results(self):
        return json.load(open(os.path.join(self.out, "results.json")))["datasets"]


def test_fresh_run_then_resume():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Repo(tmp)
        proc, calls = repo.run("--datasets", "msrvtt didemo")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert calls == ["retrieval msrvtt", "rerank msrvtt 5", "saved msrvtt",
                         "retrieval didemo", "rerank didemo 5", "saved didemo msrvtt"]
        assert "msrvtt        VidVec-O" in proc.stdout and "msrvtt        VidVec  " in proc.stdout
        assert repo.results()["msrvtt"]["rerank"]["K"] == 5

        proc, calls = repo.run("--datasets", "msrvtt didemo", "--resume")
        assert proc.returncode == 0 and calls == []
        assert repo.results()["didemo"]["rerank"]["after"]["R@1"] == 65.0


def test_resume_reruns_reranking_when_k_changes():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Repo(tmp)
        repo.run()
        proc, calls = repo.run("--resume", "--rerank-k", "3")
        assert calls == ["rerank msrvtt 3", "saved msrvtt"]
        assert repo.results()["msrvtt"]["rerank"]["K"] == 3


def test_rerank_k_zero_reports_vidvec_o_only():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Repo(tmp)
        repo.run()
        proc, calls = repo.run("--resume", "--rerank-k", "0")
        assert proc.returncode == 0 and calls == []
        assert "VidVec-O" in proc.stdout and "VidVec  " not in proc.stdout
        assert "rerank" not in repo.results()["msrvtt"]


def test_failed_benchmark_is_retried_on_resume():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Repo(tmp)
        proc, calls = repo.run("--datasets", "msrvtt didemo", FAIL_RETRIEVAL="didemo")
        assert proc.returncode == 1 and "FAILED: didemo" in proc.stdout
        assert calls == ["retrieval msrvtt", "rerank msrvtt 5", "saved msrvtt", "retrieval didemo"]
        assert repo.results()["didemo"]["status"] == "failed"

        proc, calls = repo.run("--datasets", "msrvtt didemo", "--resume")
        assert proc.returncode == 0
        assert calls == ["retrieval didemo", "rerank didemo 5", "saved didemo msrvtt"]
        assert repo.results()["didemo"]["status"] == "ok"


def test_stale_rerank_result_is_not_reported():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Repo(tmp)
        repo.run()
        # Retrieval runs again (no --resume) and the new reranking fails: the earlier
        # VidVec numbers belong to the old candidates and must not be shown.
        proc, calls = repo.run(FAIL_RERANK="msrvtt")
        assert proc.returncode == 1 and "FAILED: msrvtt_rerank" in proc.stdout
        assert "rerank" not in repo.results()["msrvtt"]
        assert not os.path.exists(os.path.join(repo.out, "per_dataset", "msrvtt_rerank.json"))


def test_missing_data_root_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Repo(tmp)
        env = dict(repo.env)
        env.pop("VIDVEC_DATA_ROOT")
        proc = subprocess.run(["bash", os.path.join(repo.root, "scripts", "eval.sh")],
                              capture_output=True, text=True, env=env)
        assert proc.returncode == 1 and "VIDVEC_DATA_ROOT" in proc.stderr


def test_each_benchmark_prints_and_saves_vidvec_o_before_its_reranking():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Repo(tmp)
        proc, calls = repo.run("--datasets", "msrvtt didemo")
        out = proc.stdout
        # per benchmark, in order: retrieval, its VidVec-O row, its reranking, its VidVec row
        order = [out.index(x) for x in (
            "[run ] msrvtt retrieval", "msrvtt        VidVec-O", "[run ] msrvtt rerank",
            "msrvtt        VidVec  ", "[run ] didemo retrieval", "didemo        VidVec-O",
            "[run ] didemo rerank", "didemo        VidVec  ", "All benchmarks:")]
        assert order == sorted(order)
        # when msrvtt's reranking ran, its VidVec-O results were already in results.json
        assert calls[:3] == ["retrieval msrvtt", "rerank msrvtt 5", "saved msrvtt"]


def test_resume_prints_each_row_once_per_stage():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Repo(tmp)
        repo.run()
        proc, _ = repo.run("--resume")
        before_final = proc.stdout.split("All benchmarks:")[0]
        assert before_final.count("msrvtt        VidVec-O") == 1
        assert before_final.count("msrvtt        VidVec  ") == 0  # reranking was skipped
