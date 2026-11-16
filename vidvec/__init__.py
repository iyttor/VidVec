"""VidVec: video MLLM embeddings for video-text retrieval.

Modules:

* ``prompts``     the evaluation protocol's constants (stdlib only)
* ``model``       the embedding model, its training loss, and its loader
* ``train_data``  the training set: caption pairs and their collator
* ``benchmarks``  the five retrieval benchmarks: loaders, datasets, collator
* ``retrieval``   first stage: encoding, scoring, metrics (VidVec-O)
* ``rerank``      second stage: zero-shot MLLM reranking of the top-K (VidVec)
* ``report``      per-benchmark output and the merged summary (stdlib only)

Nothing is imported here, so ``vidvec.prompts`` and ``vidvec.report`` stay cheap.
"""
