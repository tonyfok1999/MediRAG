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

from transformers import AutoTokenizer

from config import Config
from rag import llm
from schema import Message, Role, render_transcript


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


REWRITE_SYSTEM_PROMPT = """
You convert patient descriptions into short clinical search queries for a medical textbook retrieval system.

Rules:
- Output ONLY the query. No preamble, no quotes, no "Query:" prefix, no explanation.
- Keep it under 12 words.
- Use the clinical terminology a textbook would use, not the patient's wording.
- Preserve the details that discriminate between conditions: onset, duration, laterality, aggravating and relieving factors, and associated symptoms.
- Read the WHOLE conversation. Details arrive across turns — onset in the first message, the decisive symptom three turns later.
- Describe, do not diagnose. Write the presentation, never a suspected condition.
- Never invent a detail the patient did not state.
""".strip()


# Few-shot pairs. These do more work than the rules above — for a task whose
# output is a fixed shape, examples beat instructions.
#
# Chosen to cover the four things that go wrong:
#   1. lay phrasing -> textbook term, single turn
#   2. detail split across turns (onset in turn 1, red flag in turn 3)
#   3. anatomical vagueness -> named region
#   4. the decisive detail arriving in an ANSWER to a clarifying question
#
# Note what none of them do: name a condition. "pleuritic chest pain worse
# supine" retrieves the differential; "pericarditis" retrieves only the answer
# you already guessed, and retrieval can no longer correct you.
REWRITE_EXAMPLES: list[tuple[list[Message], str]] = [
    (
        [
            Message(role=Role.USER, text="chest hurts when I breathe in, worse lying down, 3 days"),
        ],
        "pleuritic chest pain worse supine acute onset",
    ),
    (
        [
            Message(role=Role.USER, text="I've had a really bad headache since yesterday morning"),
            Message(role=Role.BOT, text="Did it come on suddenly, or build up gradually?"),
            Message(role=Role.USER, text="It hit me all at once. Worst headache of my life, and my neck feels stiff now."),
        ],
        "thunderclap headache sudden onset with neck stiffness",
    ),
    (
        [
            Message(role=Role.USER, text="my stomach's been killing me, kind of down on the lower right side, and I threw up twice"),
        ],
        "right lower quadrant abdominal pain with vomiting",
    ),
    (
        [
            Message(role=Role.USER, text="I keep getting dizzy, it's been a couple of weeks"),
            Message(role=Role.BOT, text="Does it happen at any particular moment?"),
            Message(role=Role.USER, text="mostly when I stand up fast, and sometimes my heart races at the same time"),
        ],
        "orthostatic dizziness with palpitations subacute",
    ),
]


REWRITE_TEMPLATE = """\
{examples}

Conversation:

{conversation}

Clinical search query:"""


def _render_example(conversation: list[Message], query: str) -> str:
    """One few-shot pair, rendered exactly like the real thing.

    Reusing render_transcript here is the point: examples the model sees during
    the shot and the live conversation are formatted by the same code, so the
    two can never drift apart.
    """
    return (
        "Conversation:\n\n"
        f"{render_transcript(conversation)}\n\n"
        f"Clinical search query: {query}"
    )


# Rendered once at import — the shots are static, so there is no reason to pay
# for the join on every turn.
REWRITE_FEWSHOT = "\n\n".join(
    _render_example(conversation, query) for conversation, query in REWRITE_EXAMPLES
)


# Prefixes the model adds back despite being told not to. Lowercased for the
# comparison; matched longest-first so "clinical search query:" is not left
# with a dangling "clinical".
_QUERY_PREFIXES = ("clinical search query:", "search query:", "query:")

# Stripped from both ends until stable. One str.strip() over the whole set
# beats chained strips: models write 'query'. as readily as 'query', and a
# fixed order leaves the closing quote stranded behind the period.
_QUERY_JUNK = "\"'. "


def _clean(raw: str) -> str:
    """Strip the packaging a model puts around a one-line answer.

    Instructions alone do not reliably suppress quotes, a restated label, or a
    trailing period, and every one of those characters is a token the encoder
    then wastes on punctuation instead of clinical signal. Only the first line
    is kept: when the model adds an unrequested explanation it goes below the
    query, so dropping everything after the first newline discards it.
    """
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return ""

    query = lines[0]
    lowered = query.lower()
    for prefix in _QUERY_PREFIXES:
        if lowered.startswith(prefix):
            query = query[len(prefix):].strip()
            break

    return query.strip(_QUERY_JUNK)


def rewrite_query(conversation: list[Message], cfg: Config) -> str:
    """Convert the conversation so far into a short clinical search query.

    Input:  the FULL conversation, not just the last message. Key details
            arrive across turns — onset in turn 1, the red-flag symptom in
            turn 3. Rewriting from the last message alone throws that away.

    Output: a short query in clinical terminology.

        "chest hurts when I breathe in, worse lying down, 3 days"
          -> "pleuritic chest pain worse supine acute onset"

    Assert on the token count before returning. Log a warning above ~50 tokens.
    A silent truncation that quietly degrades retrieval is exactly the kind of
    bug that survives to production because nothing ever raises.
    """
    # TODO(step 4): enforce cfg.max_query_tokens on the way out — warn above
    # ~50, retry once above the limit, then truncate on token boundaries.
    # count_query_tokens() above is the counter to use.
    prompt = REWRITE_TEMPLATE.format(
        examples=REWRITE_FEWSHOT,
        conversation=render_transcript(conversation),
    )
    # effort="low": this is a short translation, not reasoning, and it runs on
    # every turn plus twice per eval question.
    raw = llm.complete(prompt, system=REWRITE_SYSTEM_PROMPT, cfg=cfg, effort="low")
    return _clean(raw)


def rewrite_multi(conversation: list[Message], cfg: Config, n: int = 3) -> list[str]:
    """Optional Day 10 ablation: generate n diverse queries, retrieve for each,
    fuse with RRF. Multi-query retrieval often beats single-query because it
    hedges against one bad rewrite. Free extra row in your results table.
    """
    raise NotImplementedError
