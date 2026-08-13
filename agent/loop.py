"""Days 8-9 — the ask-vs-answer decision. The intellectual core of the project.

Two approaches, and you should be able to argue the tradeoff:

  Slot filling      Predefined slots, ask until filled. Deterministic, easy to
                    evaluate, terminates, boring.
  Free-form LLM     Model decides each turn. Natural, flexible, hard to
                    evaluate, can loop forever.

What's specced below is the hybrid: required slots must be filled (quality
floor + guaranteed termination), but the LLM picks phrasing and ordering, and
may ask one free-form follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent.session import Session
from config import Config


class Action(str, Enum):
    ASK = "ask"
    ANSWER = "answer"


@dataclass
class Decision:
    action: Action
    question: str | None   # populated when action is ASK
    reasoning: str         # always log this


def decide(session: Session, cfg: Config) -> Decision:
    """Ask a clarifying question, or answer now?

    Policy:
      1. required slot missing AND questions_asked < cfg.max_clarifying_questions
             -> ASK
      2. questions_asked >= cfg.max_clarifying_questions
             -> ANSWER (hard stop: prevents infinite loops, and reflects that
                real users abandon after about four questions — say this in
                your README, it shows you thought about the human, not just
                the algorithm)
      3. otherwise
             -> let the LLM decide, with structured output

    Even when the slot is predetermined, let the LLM phrase the question.
    "How long has this been going on?" reads like a clinician;
    "DURATION:" reads like a form.

    Always populate `reasoning` and log it. When the bot asks something dumb
    at 11pm on Day 9, that field is the only thing that will tell you why.
    """
    raise NotImplementedError


def extract_slots(session: Session, cfg: Config) -> dict[str, str]:
    """Pull slot values out of the user's latest message.

    Users answer more than they're asked: "how long?" gets "about 3 days, and
    it's worse at night". Extract everything available, not just the slot you
    asked about — otherwise you'll ask a question the user already answered,
    which is the fastest way to make a bot feel stupid.
    """
    raise NotImplementedError
