"""Day 2-3 — search. Day 10 — hybrid.

The rest of the system talks to this module through `search()` and nothing else.
Keeping that boundary clean is what lets you swap strategies on Day 10 by
changing one config flag instead of editing four files.
"""

from __future__ import annotations

import warnings

import torch
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoTokenizer

from config import Config
from schema import Chunk

QDRANT_URL = "http://localhost:6333"
MAX_QUERY_TOKENS = 64  # MedCPT's query encoder truncates here and drops the rest silently


class Retriever:
    def __init__(self, cfg: Config):
        """Load the QUERY encoder and connect to Qdrant.

        Load the encoder once, here — not per call. Loading a transformer on
        every query adds seconds of latency and you will not notice until you
        wonder why the bot feels slow.
        """
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.query_encoder)
        self.model = AutoModel.from_pretrained(cfg.query_encoder)
        self.model.eval()
        self.client = QdrantClient(url=QDRANT_URL)

    def search(self, query: str, k: int | None = None) -> list[Chunk]:
        """Dense search: QUERY encoder → Qdrant. k defaults to cfg.top_k."""
        k = k or self.cfg.top_k

        # 1. Guard query length: warn if it'll get silently truncated.
        n_tokens = len(self.tokenizer.tokenize(query))
        if n_tokens > MAX_QUERY_TOKENS:
            warnings.warn(
                f"query is {n_tokens} tokens; MedCPT's query encoder truncates "
                f"at {MAX_QUERY_TOKENS} — the tail of this query will be dropped."
            )

        # 2. Encode the query with the QUERY encoder ([CLS] pooling, same
        #    convention as the article encoder used to build the index —
        #    but NOT the same model; query/article encoders aren't interchangeable).
        with torch.no_grad():
            encoded = self.tokenizer(
                query,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=MAX_QUERY_TOKENS,
            )
            query_vector = self.model(**encoded).last_hidden_state[:, 0, :][0].tolist()

        # 3. Search Qdrant for the k nearest article vectors (dot product,
        #    matching the DOT distance the collection was created with).
        hits = self.client.query_points(
            collection_name=self.cfg.article_collection,
            query=query_vector,
            limit=k,
        ).points

        # 4. Map Qdrant hits back into the shared Chunk type so callers never
        #    touch Qdrant's response shape directly.
        return [
            Chunk(
                id=hit.payload["chunk_id"],
                text=hit.payload["text"],
                title=hit.payload["title"],
                score=hit.score,
            )
            for hit in hits
        ]

    def search_bm25(self, query: str, k: int) -> list[Chunk]:
        """Lexical search. Day 10 only — skip until then."""

    def search_hybrid(self, query: str, k: int) -> list[Chunk]:
        """Dense + BM25 fused with RRF.

        RRF: score(d) = sum over rankers of 1 / (rrf_k + rank(d))
        Note it uses RANK not score — that's the whole point, it avoids
        having to normalize incomparable score scales. Understand why before
        implementing; it's a likely interview question given it's on your résumé.
        """