"""Day 3 — prompt assembly and the LLM call.

Keep temperature at 0. You cannot evaluate a nondeterministic system on 150
questions and trust the difference between two runs.
"""

from __future__ import annotations

from config import Config
from schema import Message, RetrievalResult


SYSTEM_PROMPT = """TODO(day 3).

Must cover:
  - role: health information, explicitly NOT diagnosis
  - ground answers in the provided context; say so when context is insufficient
  - cite which snippets were used
  - the scope boundary from scope.md
  - urgency guidance: self-care / see a GP / urgent care / emergency
  - plain language, no jargon without explanation

Keep this in version control and treat prompt edits like code changes — when
your eval number moves, you need to know whether the prompt or the retriever
caused it. Rerun the eval after every prompt change.
"""


def build_prompt(
    conversation: list[Message],
    retrieval: RetrievalResult,
    cfg: Config,
) -> str:
    """Assemble system prompt + context + conversation into the final prompt.

    Decisions to make and write down:
      - context before or after the conversation? (order affects attention)
      - how many chunks? cfg.max_context_chunks
      - how to label chunks so citations are traceable back to chunk ids
    """
    raise NotImplementedError


def answer(
    conversation: list[Message],
    retrieval: RetrievalResult,
    cfg: Config,
) -> str:
    """Generate the user-facing answer."""
    raise NotImplementedError


def answer_mcq(question: str, options: dict[str, str], cfg: Config) -> tuple[str, str]:
    """Multiple-choice path used ONLY by the Tier 1 eval harness.

    Returns (predicted_option_key, context_used). The context string is
    returned so the harness can compute the retrieval proxy metric.

    This is a separate entry point from answer() on purpose: your eval task
    (pick A/B/C/D) is not your product task (converse). Sharing one function
    would force you to compromise both. Keeping them separate is also the
    honest framing for your README — say clearly that Tier 1 measures the
    pipeline, not the product.
    """
    raise NotImplementedError
