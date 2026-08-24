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

import functools
import warnings

from transformers import AutoTokenizer

from config import Config
from rag import llm
from schema import Message, render_transcript


@functools.lru_cache(maxsize=1)
def _tokenizer(name: str):
    """MedCPT's query tokenizer, loaded once per process.

    A tokenizer is just a vocab file — a few MB, no model weights — so this is
    not a second copy of MedCPT sitting in memory alongside the retriever's.
    Cached because rewrite_query runs once per turn, and re-reading the vocab
    every call is the same mistake Retriever.__init__ warns about, just smaller.
    """
    return AutoTokenizer.from_pretrained(name)


def count_query_tokens(query: str, cfg: Config) -> int:
    """Length of `query` in MedCPT query-encoder tokens.

    Count tokens, never words. "pleuritic" is one word and several tokens, so
    a word count will read 40 while the encoder is already past its 64-token
    limit — and everything past that limit is dropped without an error.
    """
    return len(_tokenizer(cfg.query_encoder).tokenize(query))


# Tokens at which we start complaining but still proceed. Headroom below
# cfg.max_query_tokens: a query that lands here is one clause away from being
# silently truncated, and you want to see that coming in the logs.
WARN_AT_TOKENS = 50

# How many times to re-ask when the model overshoots the budget. One. A model
# that ignores an explicit word cap twice is not going to obey a third time,
# and the bot has a human waiting.
MAX_RETRIES = 1


class QueryTooLongError(ValueError):
    """The rewriter could not get under cfg.max_query_tokens.

    Its own type so MediRAG.answer() can catch precisely this and decide the
    fallback — apologise to the user, or search on the raw message anyway.
    That policy needs to know whether a human is waiting; this module doesn't.
    """


REWRITE_SYSTEM_PROMPT = """
You convert a patient's description of their symptoms into a short search query \
in clinical terminology. The query is fed to a medical literature search engine, \
not to a person.

Rules:
- Output keywords and clinical noun phrases, not a sentence.
- Use standard clinical terminology where the patient used lay language.
- Keep: the symptom, its qualifiers, location, onset and duration, and anything \
that makes it better or worse.
- Drop: pleasantries, first-person narration, hedging, and the assistant's own \
questions.
- Use the whole conversation. Details arrive across turns — onset in the first \
message, the important symptom three turns later.
- Never exceed 15 words. Shorter is better.

Examples:

Conversation:
Patient: chest hurts when I breathe in, worse lying down, been about 3 days
Query: pleuritic chest pain worse supine acute onset

Conversation:
Patient: I've had a headache for three days now, mostly the right side
Assistant: Have you noticed any other symptoms alongside it?
Patient: this morning my neck feels really stiff when I try to look down
Query: unilateral headache neck stiffness three days

Conversation:
Patient: my stomach has been killing me since last night, low down on the right
Assistant: Has anything else changed since it started?
Patient: I threw up twice and I don't feel like eating
Query: right lower quadrant abdominal pain vomiting anorexia acute

Conversation:
Patient: I keep needing to pee and it burns, since Tuesday
Query: dysuria urinary frequency several days
""".strip()

# Appended on the retry. Deliberately blunter than the original instruction:
# if the model overshot, restating the same rule in the same words is unlikely
# to land differently.
STRICTER_SUFFIX = (
    "\n\nYour previous attempt was too long. Output at most 8 words. "
    "Keywords only."
)

REWRITE_TEMPLATE = """\
Conversation:
{transcript}
Query:"""

# One field, required, nothing else allowed. additionalProperties=False stops
# the model adding a chatty "explanation" key alongside the query.
QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


def _ask(transcript: str, cfg: Config, stricter: bool = False) -> str:
    """One rewrite attempt. Returns a cleaned single-line query."""
    system = REWRITE_SYSTEM_PROMPT + (STRICTER_SUFFIX if stricter else "")
    result = llm.complete_json(
        prompt=REWRITE_TEMPLATE.format(transcript=transcript),
        system=system,
        cfg=cfg,
        schema=QUERY_SCHEMA,
        effort="low",  # a one-line translation, not a reasoning problem
    )
    # Collapse any whitespace: a newline inside the query would corrupt logs and
    # tells us nothing useful, and the encoder ignores it either way.
    return " ".join(str(result.get("query", "")).split())


def rewrite_query(conversation: list[Message], cfg: Config) -> str:
    """Convert the conversation so far into a short clinical search query.

    Input:  the FULL conversation, not just the last message. Key details
            arrive across turns — onset in turn 1, the red-flag symptom in
            turn 3. Rewriting from the last message alone throws that away.

    Output: a short query in clinical terminology.

        "chest hurts when I breathe in, worse lying down, 3 days"
          -> "pleuritic chest pain worse supine acute onset"

    Raises QueryTooLongError rather than trimming an over-budget query. A
    silent truncation degrades retrieval with nothing ever failing, which is
    exactly the kind of bug that survives to production because nothing raises.
    """
    transcript = render_transcript(conversation)
    if not transcript:
        raise ValueError("rewrite_query requires at least one message")

    for attempt in range(MAX_RETRIES + 1):
        query = _ask(transcript, cfg, stricter=attempt > 0)
        if not query:
            raise ValueError("rewriter returned an empty query")

        n_tokens = count_query_tokens(query, cfg)
        if n_tokens <= cfg.max_query_tokens:
            if n_tokens > WARN_AT_TOKENS:
                warnings.warn(
                    f"rewritten query is {n_tokens} tokens, close to the "
                    f"{cfg.max_query_tokens}-token limit: {query!r}"
                )
            return query

        warnings.warn(
            f"rewritten query is {n_tokens} tokens, over the "
            f"{cfg.max_query_tokens}-token limit; retrying: {query!r}"
        )

    raise QueryTooLongError(
        f"could not rewrite under {cfg.max_query_tokens} tokens after "
        f"{MAX_RETRIES + 1} attempts; last attempt was {n_tokens} tokens: {query!r}"
    )


def rewrite_multi(conversation: list[Message], cfg: Config, n: int = 3) -> list[str]:
    """Optional Day 10 ablation: generate n diverse queries, retrieve for each,
    fuse with RRF. Multi-query retrieval often beats single-query because it
    hedges against one bad rewrite. Free extra row in your results table.
    """
    raise NotImplementedError
