"""Shared dataclasses.

These live at the root, outside any package, so that rag/, agent/ and bot/ can
all import them without importing each other. Keeping shared types in one place
is what stops you from having a circular-import problem in week 2.

Named `schema.py` and not `types.py` on purpose: the repo root ends up on
sys.path when you run `python -m ingest.embed`, so a root-level types.py would
shadow the stdlib `types` module for every library you import. torch and
transformers both import it. That failure is confusing enough to lose an hour to.
"""

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    USER = "user"
    BOT = "bot"


@dataclass
class Message:
    role: Role
    text: str


@dataclass
class Chunk:
    """One retrieved corpus snippet.

    `title` is what you show the user as a citation; `id` is what you log for
    eval debugging. Keep both — when a bad answer comes out you need to trace
    exactly which chunks produced it.
    """

    id: str
    text: str
    title: str
    score: float


@dataclass
class RetrievalResult:
    """What the retriever hands to the generator.

    Wrapping the chunk list in an object (rather than returning a bare list)
    means you can add fields later — the rewritten query, timing, which
    strategy ran — without changing every call site.
    """

    query: str
    chunks: list[Chunk] = field(default_factory=list)
    strategy: str = "dense"
    latency_ms: float = 0.0

    def as_context(self, max_chunks: int | None = None) -> str:
        """Flatten chunks into the context string for the prompt.

        TODO(day 3): decide the format. Numbered? Titled? Separators matter
        more than you'd expect for citation accuracy — try at least two.
        """
        raise NotImplementedError
