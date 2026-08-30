"""The orchestrator — the box the README architecture diagram implies but does
not name.

Every other module in rag/ is deliberately ignorant of the ones around it:
generator.py never imports retriever.py, retriever.py never learns what a system
prompt is. That is what lets Day 10 swap dense for hybrid by flipping a config
flag instead of editing four files — but it also means nothing owns the
SEQUENCE. This does.

It also owns the Retriever instance. Retriever.__init__ loads a transformer, so
constructing one per query would add seconds to every turn; built lazily here
and cached for the process lifetime.
"""

from __future__ import annotations

import logging
import time

from config import Config
from rag import generator, rewriter
from rag.retriever import Retriever
from schema import Message, RetrievalResult, Role

log = logging.getLogger(__name__)


class MediRAG:
    """rewrite -> search -> answer.

    The eval harness dictates the shape: run_pipeline_eval.py calls
    `system.answer_mcq(question, options)`, so this must be an object with
    methods rather than a module of functions.
    """

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._retriever: Retriever | None = None

    @property
    def retriever(self) -> Retriever:
        """Built on first use, then reused.

        Lazy rather than eager so NoRetrievalControl — which never searches —
        does not pay for a transformer load it will not use.
        """
        if self._retriever is None:
            self._retriever = Retriever(self.cfg)
        return self._retriever

    def build_query(self, conversation: list[Message]) -> str:
        """Conversation -> the string handed to the retriever.

        Gated on cfg.use_rewriter so the rewrite becomes a ROW IN THE RESULTS
        TABLE rather than an unmeasured assumption. The README claims rewriting
        matters; this flag is what turns that claim into a number.
        """
        if self.cfg.use_rewriter:
            return rewriter.rewrite_query(conversation, self.cfg)
        return conversation[-1].text

    def retrieve(self, conversation: list[Message]) -> RetrievalResult:
        """Search, timed, with the query that actually ran recorded on the result."""
        if self.cfg.use_hybrid:
            # Refuse rather than silently label a dense search "hybrid" —
            # Retriever.search_hybrid is still a Day 10 stub.
            raise NotImplementedError(
                "cfg.use_hybrid is set but Retriever.search_hybrid is unimplemented (Day 10)"
            )

        raw = conversation[-1].text
        query = self.build_query(conversation)

        # Both halves logged on purpose: when retrieval comes back irrelevant,
        # this is the only thing that tells you whether the rewrite was bad or
        # the search was.
        log.info("query: %r -> %r", raw, query)

        t0 = time.perf_counter()
        chunks = self.retriever.search(query)
        latency_ms = (time.perf_counter() - t0) * 1000

        log.info("retrieved %d chunks in %.0f ms", len(chunks), latency_ms)
        return RetrievalResult(
            query=query, chunks=chunks, strategy="dense", latency_ms=latency_ms
        )

    def answer(self, conversation: list[Message]) -> str:
        """The product path: full conversation in, cited answer out."""
        retrieval = self.retrieve(conversation)
        return generator.answer(conversation, retrieval, self.cfg)

    def ask(self, text: str) -> str:
        """Single-turn convenience for manual testing. Not used by the bot."""
        return self.answer([Message(role=Role.USER, text=text)])

    def answer_mcq(self, question: str, options: dict[str, str]) -> tuple[str, str]:
        """Tier 1 eval entry point. Day 4."""
        raise NotImplementedError


class NoRetrievalControl(MediRAG):
    """The Day 4 baseline: same prompts, same model, no retrieved context.

    A subclass rather than a copy so the control differs from the real system in
    EXACTLY one thing. Fork the code and the two stop being comparable, and the
    delta in your README stops meaning what it says.

    If this scores as well as MediRAG, retrieval is not earning its place — that
    is the single most important number Day 4 produces, and it is worth knowing
    on Day 4 rather than Day 14.
    """

    def retrieve(self, conversation: list[Message]) -> RetrievalResult:
        # as_context() renders empty chunks as an explicit "nothing retrieved"
        # sentence, so the model is told there is no material rather than being
        # handed a blank slot it will quietly fill from parametric knowledge.
        return RetrievalResult(query=conversation[-1].text, chunks=[], strategy="none")
