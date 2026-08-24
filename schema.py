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


# How each turn is labelled when a conversation is rendered into a prompt.
# Deliberately NOT the raw Role values: "bot:" is a label no model has seen in
# training, and keeping these distinct from the API's own user/assistant roles
# stops the rendered transcript from being confused with real message roles.
#
# Lives here rather than in rag/generator.py because both the generator and the
# rewriter render transcripts, and rewriter.py importing generator.py would run
# the dependency backwards.
ROLE_LABELS = {Role.USER: "Patient", Role.BOT: "Assistant"}


def render_transcript(messages: list["Message"]) -> str:
    """Render a conversation as labelled plain text for a prompt.

    Returns "" for an empty list — the caller substitutes its own placeholder,
    because "(no prior turns)" and "(no conversation yet)" are prompt-specific
    wording, not a shared concern.

    Role(...) coerces both a Role member and a bare "user"/"bot" string, so a
    Message built without the enum doesn't KeyError one frame deeper. Role is a
    str-Enum, so Role.USER == "user" is True, but their hashes differ — the
    plain string would miss the dict lookup.
    """
    return "\n".join(f"{ROLE_LABELS[Role(m.role)]}: {m.text}" for m in messages)


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

        # Truncate the result to a maximum number of chunks if specified.
        chunks = self.chunks[:max_chunks] if max_chunks else self.chunks
        if not chunks:
            return "No reference material was retrieved for this query."

        # Format each chunk as an <reference> XML element.
        return "\n\n".join(
            f'<reference id="{i}" source="{c.title}">\n{c.text}\n</reference>'
            for i, c in enumerate(chunks, start=1)
        )
