"""Day 8 — per-conversation state.

Telegram gives you no session object. Every message arrives with a chat_id and
nothing else. Designing this properly is one of the more visible pieces of
engineering in the project, and Day 12 tests it with two concurrent users.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schema import Message

# Slots the agent tries to fill before answering. REQUIRED gives you a quality
# floor and guarantees the loop terminates; the rest are opportunistic.
REQUIRED_SLOTS = ["age", "sex", "onset", "duration", "severity"]
OPTIONAL_SLOTS = ["location", "associated_symptoms", "history", "medications"]


@dataclass
class Session:
    chat_id: int
    history: list[Message] = field(default_factory=list)
    slots: dict[str, str | None] = field(default_factory=dict)
    questions_asked: int = 0
    disclaimer_shown: bool = False

    def missing_required_slots(self) -> list[str]:
        raise NotImplementedError

    def add(self, message: Message) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear everything except chat_id. Backs the /reset command."""
        raise NotImplementedError


class SessionStore:
    """Day 8: an in-memory dict is fine for a laptop demo.
    Day 12: swap to SQLite so a restart doesn't wipe every conversation.

    Whichever you ship, state the limitation in your README. "In-memory;
    would use Redis in production" is a perfectly good answer. Pretending
    the problem doesn't exist is not.

    Concurrency: python-telegram-bot processes updates concurrently. Key
    everything by chat_id and never keep mutable state at module level —
    that is what stops two users' conversations from bleeding into each other.
    """

    def get(self, chat_id: int) -> Session:
        raise NotImplementedError

    def save(self, session: Session) -> None:
        raise NotImplementedError

    def clear(self, chat_id: int) -> None:
        raise NotImplementedError
