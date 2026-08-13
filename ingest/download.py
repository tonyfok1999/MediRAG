"""Day 1 — pull the corpus and look at it before you build anything on top of it.

Run:  python -m ingest.download
"""

from __future__ import annotations
from config import Config
from datasets import load_dataset
from datasets import load_from_disk
import random

def download(cfg: Config):
    """Fetch the corpus from HuggingFace and cache it locally.

    Use `datasets.load_dataset(cfg.corpus)`. Cache to disk so you never
    re-download — you'll be loading this repeatedly for the next three weeks.
    """

    ds = load_dataset(cfg.corpus)

    # cache ds to disk
    ds.save_to_disk(cfg.cache_dir)
    

def inspect(cfg: Config, n: int = 5) -> None:
    """Print the schema and a few random records. Do this BEFORE writing embed.py.

    Print, at minimum:
      - the field names (do NOT assume they're id/title/content)
      - total row count
      - n random records, in full
      - the character-length distribution: min / median / mean / p95 / max

    The MedRAG paper reports ~125.8k snippets averaging ~182 characters for the
    textbooks corpus. Confirm that yourself. If your numbers disagree, find out
    why before continuing — everything downstream assumes this shape.

    That average length is the reason top_k=5 is probably too low here: five
    snippets of ~182 chars is barely a paragraph of context. You'll test that
    properly on Day 10, but form the hypothesis now while you're looking at
    the data.
    """
    ds = load_from_disk(".cache/corpus")["train"]

    for i in random.sample(range(len(ds)), 5):
        print(f"--- record {i} ---")
        print(ds[i])
        print()



if __name__ == "__main__":
    cfg = Config()
    download(cfg)
    inspect(cfg)
