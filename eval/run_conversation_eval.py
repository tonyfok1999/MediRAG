"""Tier 2 — conversational vignette eval. Day 11.

Tier 1 measures the pipeline. This measures the product.

The mismatch is the point: MedQA is single-turn multiple choice, your bot is
multi-turn conversational advice. Solving that visibly — two tiers, each with
stated limitations — is worth more in an interview than solving it silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import Config

VIGNETTES = Path("eval/data/vignettes.jsonl")


def load_vignettes(path: Path = VIGNETTES) -> list[dict]:
    """20 hand-written cases. Schema per line:

        {
          "id": "v01",
          "opening_message": "what the patient says first",
          "hidden_facts": {"onset": "3 days", "fever": "38.9C", ...},
          "expected_conditions": ["meningitis", "..."],
          "expected_urgency": "emergency",     # self_care|gp|urgent|emergency
          "should_refuse": false,
          "is_red_flag": true
        }

    Include at least 4 adversarial cases: an out-of-scope drug dosing request,
    a red-flag emergency, a pediatric case, and a vague description that MUST
    trigger a clarifying question rather than a guess.

    Draw presentations from textbook case examples or public sources. Never
    use anything from a real person.
    """
    raise NotImplementedError


def simulate_patient(vignette: dict, bot_message: str, history: list) -> str:
    """LLM plays the patient.

    Critical constraint: it may ONLY reveal facts from vignette["hidden_facts"]
    when actually asked about them. If it volunteers everything upfront you are
    not testing the agent loop at all — you're testing single-turn RAG with
    extra steps, and your Day 8-9 work becomes unmeasurable.

    Instruct it explicitly: answer only what was asked, briefly, in lay
    language, like a patient rather than a textbook.
    """
    raise NotImplementedError


def run_conversation(vignette: dict, system, cfg: Config, max_turns: int = 8) -> list:
    """Drive one full simulated conversation. Returns the transcript."""
    raise NotImplementedError


def judge(vignette: dict, transcript: list, cfg: Config) -> dict:
    """Rubric scoring. Returns:

        condition_in_differential: bool
        urgency_correct:           bool
        urgency_direction:         'over' | 'under' | 'correct'
        asked_key_question:        bool
        correctly_refused:         bool | None

    Weight under-triage far more heavily than over-triage. Telling someone to
    see a doctor unnecessarily is a minor cost; missing a red flag is not.
    Make that asymmetry explicit in your scoring and say so in the README.

    Use a different model than your generator where you can. A model judging
    its own output is a known bias and an interviewer may well ask whether you
    controlled for it.

    Hand-check at least 5 judgments yourself and report judge-human agreement.
    "I validated my LLM judge" puts you ahead of most people who use one.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("write your vignettes first (Day 5), then wire this up (Day 11)")
