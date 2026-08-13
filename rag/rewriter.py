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

from config import Config
from schema import Message


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
