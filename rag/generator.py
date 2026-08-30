"""Day 3 — prompt assembly and the LLM call.

Keep temperature at 0. You cannot evaluate a nondeterministic system on 150
questions and trust the difference between two runs.
"""

from __future__ import annotations

import logging
import re
import warnings

from config import Config
from rag import llm
from schema import Message, RetrievalResult, Role, render_transcript

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an AI doctor for a virtual medical intake service, helping users with questions about their symptoms, general health concerns, and preliminary triage. Your communication style is warm, empathetic, and professional.

Always: 1) Acknowledge user concerns and discomfort with care, 2) Provide clear, grounded explanations using simple layperson language, 3) Conduct multi-turn diagnostic intake by asking targeted follow-up questions to gather clinical details. When users express worry or pain, respond with understanding and focus on supportive guidance.

You can help with: gathering symptom timelines, explaining medical concepts found strictly in your reference materials, and asking intake questions to clarify the user's situation.

You cannot: provide definitive medical diagnoses, prescribe medications, or draw on general medical knowledge outside of your provided reference documents.

For queries where information is missing from your reference materials, ask the user for more details to help gather better information.

For critical emergency symptoms—such as severe chest pain, sudden numbness, or difficulty breathing—immediately instruct the user to call 911 or go to the nearest emergency room.

Avoid complex technical jargon unless you explain it immediately.

End every interaction with the following disclaimer: "Disclaimer: I am an AI assistant simulating a clinical intake, not a licensed human physician. The information provided is for educational purposes based on retrieved reference documents and should not replace professional medical evaluation, diagnosis, or emergency care."
""".strip()


# Placeholder for turn 1. An empty history slot is ambiguous to the model in the
# same way an empty context block is — say the thing explicitly.
NO_PRIOR_TURNS = "(no prior turns)"

# Kept as its own constant, and injected through a template slot, so it can be
# ablated: swap it for "" and re-run Tier 1. Cheapest experiment in the project.
ANSWER_INSTRUCTION = (
    "Answer the patient's current message using only the reference material above. "
    "Cite every substantive claim with its bracketed id, e.g. [1]. "
    "If the reference material does not cover the question, say so explicitly "
    "rather than answering from general knowledge."
)

PROMPT_TEMPLATE = """\
Reference material:

{context}

Conversation so far:

{history}

Patient's current message:

{question}

{instruction}"""


def build_prompt(
    conversation: list[Message],
    retrieval: RetrievalResult,
    cfg: Config,
) -> str:

    if not conversation:
        raise ValueError("build_prompt requires at least one message")

    # Last message must be from the user
    if Role(conversation[-1].role) is not Role.USER:
        raise ValueError(
            "build_prompt expects the last message to be from the user; got "
            f"{conversation[-1].role!r}. Raising rather than warning on purpose: "
            "a silently malformed prompt shows up as an unexplained accuracy "
            "drop days later."
        )

    #get last message and earlier messages
    *earlier, current = conversation

    # retrieve search results as a context block
    context = retrieval.as_context(max_chunks=cfg.max_context_chunks)

    #label the role to earlier messages
    history = render_transcript(earlier)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        history=history or NO_PRIOR_TURNS,
        question=current.text,
        instruction=ANSWER_INSTRUCTION,
    )
    return prompt.strip()


# Matches the [1] markers ANSWER_INSTRUCTION asks the model to emit. Small
# integers on purpose: chunk ids are long and opaque, and models corrupt or
# invent them. An integer either indexes a chunk we actually sent or it does
# not, which makes a bad citation detectable instead of plausible.
# The corpus stores a book slug and nothing finer — no chapter, no section, no
# page (payload fields are text/title/chunk_id/source). A book name is therefore
# the most precise citation this corpus can honestly support, which is what
# makes the deduped one-line footer the right shape rather than a compromise.
# Slugs are mapped because "Obstentrics_Williams" — the corpus's own typo — is
# not something to show a patient.
BOOK_TITLES = {
    "Anatomy_Gray": "Gray's Anatomy for Students",
    "Biochemistry_Lippinco": "Lippincott Illustrated Reviews: Biochemistry",
    "Cell_Biology_Alberts": "Molecular Biology of the Cell",
    "First_Aid_Step1": "First Aid for the USMLE Step 1",
    "First_Aid_Step2": "First Aid for the USMLE Step 2 CK",
    "Gynecology_Novak": "Berek & Novak's Gynecology",
    "Histology_Ross": "Ross Histology: A Text and Atlas",
    "Immunology_Janeway": "Janeway's Immunobiology",
    "InternalMed_Harrison": "Harrison's Principles of Internal Medicine",
    "Neurology_Adams": "Adams and Victor's Principles of Neurology",
    "Obstentrics_Williams": "Williams Obstetrics",
    "Pathology_Robbins": "Robbins & Cotran Pathologic Basis of Disease",
    "Pathoma_Husain": "Pathoma: Fundamentals of Pathology",
    "Pediatrics_Nelson": "Nelson Textbook of Pediatrics",
    "Pharmacology_Katzung": "Katzung Basic and Clinical Pharmacology",
    "Physiology_Levy": "Berne & Levy Physiology",
    "Psichiatry_DSM-5": "Diagnostic and Statistical Manual of Mental Disorders (DSM-5)",
    "Surgery_Schwartz": "Schwartz's Principles of Surgery",
}

# Matches [1] and the grouped form [1, 2, 5] that models write unprompted.
# Missing the grouped form is not a cosmetic bug: an unmatched group is neither
# valid nor invalid, so the source silently never reaches the footer AND no
# warning fires. Keep this able to over-match rather than under-match.
CITATION_PATTERN = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*]")


def _cited_ids(text: str, n_chunks: int) -> tuple[list[int], list[int]]:
    """Split the [i] markers in `text` into resolvable and unresolvable ids.

    Order preserved, duplicates dropped — a claim cited three times is one
    source, but the footer should list sources in the order the reader met them.
    """
    seen: set[int] = set()
    valid: list[int] = []
    invalid: list[int] = []
    for match in CITATION_PATTERN.finditer(text):
        for part in match.group(1).split(","):
            i = int(part.strip())
            if i in seen:
                continue
            seen.add(i)
            (valid if 1 <= i <= n_chunks else invalid).append(i)
    return valid, invalid


def _append_sources(response: str, retrieval: RetrievalResult, cfg: Config) -> str:
    """Renumber citations by BOOK and append a numbered source list.

    The model cites chunk indices, because that is what as_context() labels and
    what makes a fabricated citation mechanically detectable. But a chunk index
    is an internal detail: five markers in the prose imply five sources, and
    dense retrieval routinely draws four of five chunks from one textbook. The
    reader is left counting sources that do not exist.

    So the markers are remapped on the way out — every chunk from Harrison's
    becomes [1], the Williams chunk becomes [2] — and each number now resolves
    to a line the reader can actually see. Grouped citations collapse for free:
    [1, 2] over two chunks of the same book renders as [1].

    Chunk-level precision is not lost, it moves to the log, where the audience
    is you tracing a bad answer rather than a patient reading one.

    The slice MUST match the one as_context() used, or [3] in the answer names a
    different chunk than [3] in the prompt — a citation that looks right, points
    somewhere else, and never raises. cfg.max_context_chunks keeps them aligned;
    slicing is deterministic, so no mapping has to be passed around.
    """
    chunks = retrieval.chunks[: cfg.max_context_chunks]
    valid, invalid = _cited_ids(response, len(chunks))

    if invalid:
        # Not a crash: a fabricated citation is a grounding failure, not a
        # transport failure, and the answer body may still be fine. Counting the
        # rate across an eval run is a cheap metric almost nobody reports.
        warnings.warn(
            f"answer cited {invalid} but only {len(chunks)} chunks were in the "
            f"prompt — those markers resolve to nothing and were dropped."
        )

    if not valid:
        return response

    # Book numbering follows first appearance in the answer, so the reader meets
    # [1] before [2].
    display = {i: BOOK_TITLES.get(chunks[i - 1].title, chunks[i - 1].title) for i in valid}
    books: list[str] = []
    for i in valid:
        if display[i] not in books:
            books.append(display[i])
    book_no = {i: books.index(display[i]) + 1 for i in valid}

    log.info(
        "citations: %s",
        " ".join(f"[{book_no[i]}]<-chunk{i}={chunks[i - 1].id}" for i in valid),
    )

    def _remap(match: re.Match) -> str:
        seen: list[int] = []
        for part in match.group(1).split(","):
            n = book_no.get(int(part.strip()))
            if n is not None and n not in seen:
                seen.append(n)
        return "".join(f"[{n}]" for n in seen)

    text = CITATION_PATTERN.sub(_remap, response)
    # An unresolvable marker renders as "", which strands the space in front of
    # it: "claim [9]." would become "claim ." Tidy both that and any doubled
    # space the substitution leaves behind.
    text = re.sub(r" +([.,;:])", lambda m: m.group(1), text)
    text = re.sub(r"  +", " ", text)

    footer = "\n".join(f"[{n}] {name}" for n, name in enumerate(books, start=1))
    return f"{text}\n\nSources:\n{footer}"


def answer(
    conversation: list[Message],
    retrieval: RetrievalResult,
    cfg: Config,
) -> str:
    """Generate the user-facing answer.

    Three stages, deliberately separable: assemble (pure), call (network),
    resolve citations (pure). Only the middle one can fail in an interesting
    way, and the two pure halves are unit-testable without a key.

    SYSTEM_PROMPT goes in the system slot rather than the returned string —
    stable across every call, so it is the natural prompt-cache prefix, and
    instructions there carry more weight than the same text pasted into a user
    turn.
    """
    prompt = build_prompt(conversation, retrieval, cfg)
    response = llm.complete(prompt, system=SYSTEM_PROMPT, cfg=cfg)
    return _append_sources(response, retrieval, cfg)


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
