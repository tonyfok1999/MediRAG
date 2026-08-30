"""The one boundary between this project and an LLM API.

Exists because rewriter.py and generator.py both need to call a model, and
neither should import the other — the rewrite runs before generation, so
rewriter -> generator points backwards down the pipeline and generator ->
rewriter is simply wrong. Two modules needing the same thing, unable to depend
on each other, is what a third module is for. Same argument schema.py makes for
the shared dataclasses.

The vendor lives here and nowhere else. cfg.llm_provider picks the backend and
complete() keeps the same signature either way, so nothing upstream ever learns
which company answered. Provider is a config flag rather than an edit for the
same reason use_hybrid is: it makes "which model" a row in the results table
instead of an assumption.
"""

from __future__ import annotations

import functools
import os

from dotenv import load_dotenv

from config import Config

# bot/main.py calls this too; load_dotenv is idempotent. Repeated here so that a
# plain `python -m eval.run_pipeline_eval` — which never imports the bot — still
# sees the key. The tidier fix is to call it once at config.py import time and
# delete both copies.
load_dotenv()


# Each provider reads its own conventional variable first, then falls back to
# the generic one. Keeping all three keys in .env at once is what makes
# cfg.llm_provider a genuine one-line switch — otherwise flipping the provider
# also means editing .env, and the two drift apart the first time you forget.
_API_KEY_VARS = {
    "openai": ("OPENAI_API_KEY", "LLM_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "LLM_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY"),
}


def _api_key(provider: str) -> str:
    for var in _API_KEY_VARS[provider]:
        key = os.getenv(var)
        if key and key.strip():
            return key.strip()
    tried = " or ".join(_API_KEY_VARS[provider])
    raise RuntimeError(f"no key for provider {provider!r} — set {tried} in .env")


# Both clients are built lazily and cached: one connection pool per process, and
# the import only happens for the provider actually in use, so you do not need
# both SDKs installed to run either path.
@functools.lru_cache(maxsize=1)
def _openai_client():
    import openai

    return openai.OpenAI(api_key=_api_key("openai"))


@functools.lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic(api_key=_api_key("anthropic"))


@functools.lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    return genai.Client(api_key=_api_key("gemini"))


def complete(
    prompt: str,
    system: str,
    cfg: Config,
    effort: str = "high",
    max_tokens: int = 16000,
) -> str:
    """Send one prompt, get the text back. No conversation state, no tools.

    `effort` is honoured on the Anthropic path only — it is that API's
    cost/quality dial and has no cross-vendor equivalent worth faking. The
    OpenAI path ignores it and uses cfg.temperature instead. Kept in the shared
    signature so callers can express intent once and not re-check the provider.
    """
    if cfg.llm_provider == "openai":
        return _complete_openai(prompt, system, cfg, max_tokens)
    if cfg.llm_provider == "gemini":
        return _complete_gemini(prompt, system, cfg, max_tokens)
    if cfg.llm_provider == "anthropic":
        return _complete_anthropic(prompt, system, cfg, effort, max_tokens)
    raise ValueError(
        f"unknown llm_provider {cfg.llm_provider!r} — expected 'openai', 'gemini' or 'anthropic'"
    )


def _complete_openai(prompt: str, system: str, cfg: Config, max_tokens: int) -> str:
    """Chat Completions.

    cfg.temperature is finally load-bearing here: OpenAI accepts it, so the
    "keep temperature at 0" instruction in generator.py's module docstring is
    actually enforced on this path. (Reasoning models are the exception — they
    accept only the default temperature and reject an explicit 0.)
    """
    response = _openai_client().chat.completions.create(
        model=cfg.llm_model,
        # max_completion_tokens, not the deprecated max_tokens alias — reasoning
        # models reject the old name outright.
        max_completion_tokens=max_tokens,
        temperature=cfg.temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    choice = response.choices[0]
    # Check before reading content: a filtered or length-capped response still
    # returns HTTP 200, and content can be None. Treating that as a normal reply
    # yields "" and a silently wrong answer rather than an error.
    if choice.finish_reason == "content_filter":
        raise RuntimeError("OpenAI content filter blocked the response")
    return (choice.message.content or "").strip()


def _complete_anthropic(
    prompt: str, system: str, cfg: Config, effort: str, max_tokens: int
) -> str:
    """Messages API.

    Note what is NOT passed: temperature. It is removed on the current model
    generation and returns a 400. Eval reproducibility on this path comes from
    cached_llm in eval/run_pipeline_eval.py, which replays byte-identical
    responses — a stronger guarantee than temperature ever was.
    """
    response = _anthropic_client().messages.create(
        model=cfg.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": effort},
    )

    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        raise RuntimeError(f"model refused the request (category={category})")

    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def _complete_gemini(prompt: str, system: str, cfg: Config, max_tokens: int) -> str:
    """Google GenAI.

    The system prompt is `system_instruction` on the config object, not a
    message with a role — Gemini has no system role in `contents`. Pinning an
    exact model id matters more here than elsewhere: aliases like
    "gemini-flash-latest" float, and a floating model turns your eval table into
    a comparison against a moving target.
    """
    from google.genai import types

    response = _gemini_client().models.generate_content(
        model=cfg.llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=cfg.temperature,
            max_output_tokens=max_tokens,
        ),
    )

    # .text is None when the candidate carried no text part — a safety block, or
    # thinking tokens having eaten the whole output budget. Both return HTTP 200,
    # so an unchecked read yields "" and a silently empty query.
    text = response.text
    if not text:
        finish = None
        if response.candidates:
            finish = getattr(response.candidates[0], "finish_reason", None)
        raise RuntimeError(f"gemini returned no text (finish_reason={finish})")
    return text.strip()
