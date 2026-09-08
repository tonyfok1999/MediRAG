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


# Both rewrite_query() and build_prompt() render the whole transcript, so an
# uncapped session grows the prompt without bound and asks the rewriter to
# compress an ever-longer history into 64 tokens. Ten exchanges is plenty for
# one complaint; anything older is almost certainly a different one.
MAX_HISTORY = 20


@dataclass
class Session:
    chat_id: int
    history: list[Message] = field(default_factory=list)
    slots: dict[str, str | None] = field(default_factory=dict)
    questions_asked: int = 0
    disclaimer_shown: bool = False

    def missing_required_slots(self) -> list[str]:
        """Required slots with no value yet — drives the agent ASK decision.

        Treats "" and None alike: a slot the user answered with nothing is not
        a slot that has been filled.
        """
        return [slot for slot in REQUIRED_SLOTS if not self.slots.get(slot)]

    def add(self, message: Message) -> None:
        """Append a turn, trimming the oldest once past MAX_HISTORY."""
        self.history.append(message)
        if len(self.history) > MAX_HISTORY:
            del self.history[:-MAX_HISTORY]

    def reset(self) -> None:
        """Clear everything except chat_id. Backs the /new command.

        Mutates in place rather than returning a fresh Session so any caller
        still holding a reference sees the cleared state — otherwise a stale
        reference keeps answering from a conversation the user just ended.
        """
        self.history.clear()
        self.slots.clear()
        self.questions_asked = 0
        self.disclaimer_shown = False


class SessionStore:
    """Day 8: an in-memory dict is fine for a laptop demo.
    Day 12: swap to SQLite so a restart doesn't wipe every conversation.

    Whichever you ship, state the limitation in your README. "In-memory;
    would use Redis in production" is a perfectly good answer. Pretending
    the problem doesn't exist is not.

    Concurrency: python-telegram-bot processes updates concurrently. Key
    everything by chat_id and never keep mutable state at module level —
    that is what stops two users' conversations from bleeding into each other.
    Note this class is not itself a critical section: the caller holds a
    per-chat lock across the whole read-modify-write, because the gap between
    get() and save() spans a 20-45 second LLM call.
    """

    def __init__(self) -> None:
        self._sessions: dict[int, Session] = {}

    def get(self, chat_id: int) -> Session:
        """Fetch this chat's session, creating one on first contact."""
        if chat_id not in self._sessions:
            self._sessions[chat_id] = Session(chat_id=chat_id)
        return self._sessions[chat_id]

    def save(self, session: Session) -> None:
        """No-op for the in-memory backing — get() already handed out the live
        object. Kept because the SQLite version will need a real write here,
        and callers should be writing the call now so that swap is one class.
        """
        self._sessions[session.chat_id] = session

    def clear(self, chat_id: int) -> None:
        self._sessions.pop(chat_id, None)
