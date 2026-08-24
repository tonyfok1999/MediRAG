"""Day 6 — patient language -> clinical terminology.

This is not an optional enhancement. MedCPT's query encoder truncates at 64
tokens (~45 words), and a real patient message routinely exceeds that:

    "So I've been having this headache for about three days now, it's mostly
     on the right side, and this morning I noticed my neck feels really stiff
     when I try to look down..."

Everything past the limit is silently discarded. Without a rewrite step your
retrieval runs on a truncated fragment of the complaint.
"""

from __future__ import annotations

import functools

from transformers import AutoTokenizer

from config import Config
from schema import Message, render_transcript


@functools.lru_cache(maxsize=1)
def _tokenizer(name: str):
    """MedCPT's query tokenizer, loaded once per process.

    A tokenizer is just a vocab file — a few MB, no model weights — so this is
    not a second copy of MedCPT sitting in memory alongside the retriever's.
    Cached because rewrite_query runs once per turn, and re-reading the vocab
    every call is the same mistake Retriever.__init__ warns about, just smaller.
    """
    return AutoTokenizer.from_pretrained(name)


def count_query_tokens(query: str, cfg: Config) -> int:
    """Length of `query` in MedCPT query-encoder tokens.

    Count tokens, never words. "pleuritic" is one word and several tokens, so
    a word count will read 40 while the encoder is already past its 64-token
    limit — and everything past that limit is dropped without an error.
    """
    return len(_tokenizer(cfg.query_encoder).tokenize(query))


def rewrite_query(conversation: list[Message], cfg: Config) -> str:
    """Convert the conversation so far into a short clinical search query.

    Input:  the FULL conversation, not just the last message. Key details
            arrive across turns — onset in turn 1, the red-flag symptom in
            turn 3. Rewriting from the last message alone throws that away.

    Output: a short query in clinical terminology.

        "chest hurts when I breathe in, worse lying down, 3 days"
          -> "pleuritic chest pain worse supine acute onset"

    Assert on the token count before returning. Log a warning above ~50 tokens.
    A silent truncation that quietly degrades retrieval is exactly the kind of
    bug that survives to production because nothing ever raises.
    """
    raise NotImplementedError


def rewrite_multi(conversation: list[Message], cfg: Config, n: int = 3) -> list[str]:
    """Optional Day 10 ablation: generate n diverse queries, retrieve for each,
    fuse with RRF. Multi-query retrieval often beats single-query because it
    hedges against one bad rewrite. Free extra row in your results table.
    """
    raise NotImplementedError
