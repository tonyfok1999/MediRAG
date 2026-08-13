"""Day 2 — embed the corpus and load it into Qdrant.

Run:  python -m ingest.embed

Budget 1-3 hours on CPU. Start it early, then build rag/retriever.py while
it runs. Do not sit and watch the progress bar.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
from config import Config
from datasets import load_from_disk
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def load_corpus(cfg: Config, cache_dir: str = ".cache/corpus") -> Iterator[dict]:
    """Stream corpus records. Yield dicts with at least: id, title, content.

    Inspect the actual schema before writing this — MedRAG corpora field names
    are not guaranteed to match what you assume. Print one record first
    (`python -m ingest.download` does this for you).

    Reads from the local cache `ingest/download.py` already wrote to disk
    (`.cache/corpus` by default) — `cfg.corpus` ("MedRAG/textbooks") is the
    HuggingFace Hub id used to *fetch* the corpus, not a local path.
    """
    return load_from_disk(cache_dir)["train"]


def embed_batch(texts: list[str], encoder, tokenizer) -> np.ndarray:

    with torch.no_grad():
        # tokenize the articles
        # (64, 182)         [2D: text × token]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=512,
        )

        # (64, 182, 768)    [3D: text × token × dim]
        # [:, 0, :] keeps only the [CLS] token per text ->
        # (64, 768)         [2D: text × dim]  ← final embeddings
        embeds = encoder(**encoded).last_hidden_state[:, 0, :]

    return embeds.numpy().astype(np.float32)



def build_index(cfg: Config, batch_size: int = 64, checkpoint_every: int = 50):

    CACHE_DIR = Path(cfg.cache_embed)

    #load corpus
    dataset = load_corpus(cfg)

    #set up batch processing
    n = len(dataset)
    n_batches = math.ceil(n / batch_size)

    #load model
    model = AutoModel.from_pretrained(cfg.article_encoder)
    tokenizer = AutoTokenizer.from_pretrained(cfg.article_encoder)
    model.eval()

    #set up checkpoint (in case accidental pause)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CACHE_DIR / "checkpoint.json"

    #load checkpoint if interrupted
    start_batch = 0
    if checkpoint_path.exists():
        start_batch = json.loads(checkpoint_path.read_text())["last_completed_batch"] + 1

    #set progress bar
    progress = tqdm(
        range(start_batch, n_batches),
        initial=start_batch,
        total=n_batches,
        desc="embedding",
        unit="batch",
    )

    #loop through batches
    for batch_idx in progress:

        #load index of chunks in this batch
        lo, hi = batch_idx * batch_size, min((batch_idx + 1) * batch_size, n)

        #embed batch
        cache_file = CACHE_DIR / f"batch_{batch_idx:06d}.npy"
        if not cache_file.exists():
            texts = dataset[lo:hi]["contents"]
            embeds = embed_batch(texts, model, tokenizer)
            np.save(cache_file, embeds)  # vectors hit disk before Qdrant ever sees them

        #check out at every 50 batches
        if (batch_idx + 1) % checkpoint_every == 0:
            checkpoint_path.write_text(json.dumps({"last_completed_batch": batch_idx}))

    #save final checkpoint
    checkpoint_path.write_text(json.dumps({"last_completed_batch": n_batches - 1}))



if __name__ == "__main__":
    build_index(Config())
