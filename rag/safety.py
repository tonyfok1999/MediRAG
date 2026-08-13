"""Day 7 — red-flag screening and scope enforcement.

Runs BEFORE retrieval and before the agent loop. Nothing else executes until
this returns OK.

Design note worth putting in your README: stage 1 is deterministic on purpose.
A regex cannot be talked out of firing; an LLM can. For the cases where being
wrong is worst, don't hand the decision to a probabilistic system when you
don't have to.
"""

from __future__ import annotations

from enum import Enum

from config import Config
from schema import Message


class SafetyVerdict(str, Enum):
    OK = "ok"
    RED_FLAG = "red_flag"          # emergency — short-circuit, no RAG
    OUT_OF_SCOPE = "out_of_scope"  # refuse with a redirect


# Stage 1 patterns. Start here, expand as your vignettes catch misses.
# Deliberately over-triaging: a false positive costs a user one unnecessary
# "seek care" message, a false negative costs far more. Tune accordingly.
RED_FLAG_PATTERNS: dict[str, list[str]] = {
    "cardiac": [],       # chest pain + radiation / sweating / nausea
    "stroke": [],        # FAST: face droop, arm weakness, speech difficulty
    "headache": [],      # sudden onset, "worst of my life", thunderclap
    "anaphylaxis": [],   # throat tightness, swelling, difficulty swallowing
    "respiratory": [],   # severe breathlessness, blue lips
    "self_harm": [],     # suicidal ideation -> crisis resources, not triage
}

OUT_OF_SCOPE_TOPICS = [
    "drug dosing / interactions / prescriptions",
    "pediatric cases (under 18)",
    "pregnancy-related presentations",
    "mental health crisis",
    "stopping or changing a medication",
    "interpreting a specific person's labs or imaging",
]


def screen(
    text: str,
    conversation: list[Message],
    cfg: Config,
) -> tuple[SafetyVerdict, str | None]:
    """Two-stage screen. Returns the verdict and, if not OK, the reply to send.

    Stage 1 — deterministic patterns above. Fast, no API call, no LLM judgment.
    Stage 2 — LLM check for what the patterns miss. Only runs if stage 1 is clean.

    Screen the CONVERSATION, not just the latest message. A user may mention
    chest pain in turn 1 and sweating in turn 3; neither alone trips the
    cardiac pattern, together they should.

    Target: red-flag path returns in under 2 seconds. It must never wait on
    retrieval — that's the whole reason it runs first.
    """
    raise NotImplementedError


def emergency_response(category: str) -> str:
    """The message sent when a red flag fires.

    Be direct and specific about the action to take. Do not hedge, do not
    offer a differential, do not continue the conversation as if nothing
    happened. This is the one place where being conversational is wrong.
    """
    raise NotImplementedError


def refusal_response(topic: str) -> str:
    """Refuse an out-of-scope request with a useful redirect.

    A refusal that just says "I can't help with that" is a bad product. Say
    what you can't do, why, and where the user should go instead.
    """
    raise NotImplementedError
