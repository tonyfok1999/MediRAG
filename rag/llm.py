"""The single boundary between this project and the LLM API.

Everything that talks to a model goes through here — the rewriter, the
generator, the safety screen's stage 2, the agent loop. One boundary means one
place to add retries, latency logging, or a provider swap, and one place to look
when the bill is wrong.

This module knows nothing about medicine, retrieval, or citations. That is the
point: it takes a string and returns a string.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from config import Config

# Built once per process, on first use. Constructing the client per call opens a
# fresh connection pool every time — the same mistake as loading a transformer
# per query, just cheaper to miss.
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # The SDK looks for ANTHROPIC_API_KEY; .env in this project calls it
        # LLM_API_KEY, so bridge it explicitly rather than relying on the
        # SDK's own resolution and wondering why it can't find a key.
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY is empty or unset. Fill it in .env — config.py "
                "calls load_dotenv() at import, so any entry point picks it up."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _extract_text(response) -> str:
    """Pull the text out of a response, refusing to read a refused turn.

    stop_reason == "refusal" comes back as HTTP 200 with no usable content, so
    checking it before touching .content is the difference between a clear
    error and a confusing empty string.
    """
    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        category = getattr(detail, "category", None)
        raise RuntimeError(f"model declined to answer (category={category})")
    return "".join(b.text for b in response.content if b.type == "text").strip()


def complete(
    prompt: str,
    system: str,
    cfg: Config,
    effort: str = "high",
    max_tokens: int = 16000,
) -> str:
    """One prompt in, one string out.

    Note what is NOT passed: temperature. It is removed on this model
    generation and returns a 400. Run-to-run reproducibility for the eval comes
    from cached_llm's prompt-hash cache, which is exact rather than merely
    low-variance.
    """
    response = _get_client().messages.create(
        model=cfg.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": effort},
    )
    return _extract_text(response)


def complete_json(
    prompt: str,
    system: str,
    cfg: Config,
    schema: dict[str, Any],
    effort: str = "high",
    max_tokens: int = 16000,
) -> dict[str, Any]:
    """Same call, but the model fills in a form instead of writing prose.

    `schema` is a JSON Schema object. The API constrains generation to match it,
    so there is no "Sure! Here's your query:" preamble to strip and no parsing
    heuristic to break the first time the model phrases things differently.

    Used by the rewriter now; the agent loop's ASK/ANSWER Decision is the next
    caller, which is why this takes an arbitrary schema rather than hardcoding
    one shape.
    """
    response = _get_client().messages.create(
        model=cfg.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": schema},
        },
    )
    return json.loads(_extract_text(response))
