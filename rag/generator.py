"""Day 3 — prompt assembly and the LLM call.

Keep temperature at 0. You cannot evaluate a nondeterministic system on 150
questions and trust the difference between two runs.
"""

from __future__ import annotations

from config import Config
from schema import Message, RetrievalResult, Role, render_transcript


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
