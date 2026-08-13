"""Tier 1 — MedQA pipeline eval. Scaffolding given in full.

Run:  python -m eval.run_pipeline_eval

The scoring logic is yours. The reproducibility machinery is here because
getting it subtly wrong invalidates every number you produce, silently.

Day 4 is the most important day of the project. Two numbers go in your README
today: RAG accuracy, and the no-retrieval control. If RAG doesn't beat the bare
LLM, retrieval is actively hurting and you need to know that on Day 4, not 14.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from config import Config

CACHE = Path("eval/.llm_cache")
RESULTS = Path("eval/results")
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def cached_llm(prompt: str, model: str, call_fn: Callable[[str], str]) -> str:
    """Cache LLM responses by prompt hash.

    Re-running an eval after a non-LLM change should cost close to nothing.
    Without this you will start avoiding eval runs because they cost money,
    then stop measuring, and the project loses the thing that made it worth
    building.
    """
    key = hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    response = call_fn(prompt)
    path.write_text(
        json.dumps({"prompt": prompt, "response": response}), encoding="utf-8"
    )
    return response


def load_questions(path: str = "eval/data/medqa_sample.jsonl") -> list[dict]:
    """Load the fixed 150-question sample.

    Sample once with a fixed seed and COMMIT the file. Re-sampling between runs
    means your numbers aren't comparable and you won't notice.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found — build it first (Day 4). Sample 150 MedQA test "
            f"questions with a fixed seed and commit the file."
        )
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


def run_eval(cfg: Config, questions: list[dict], system, label: str = "") -> dict:
    """Run the eval and write a timestamped result file.

    `system` needs one method: answer_mcq(question, options) -> (pred, context)
    """
    correct = 0
    retrieval_hits = 0
    records = []

    for q in questions:
        t0 = time.time()
        pred, context = system.answer_mcq(q["question"], q["options"])
        latency = time.time() - t0

        is_correct = pred == q["answer"]
        # Retrieval proxy: MedQA has no gold passages, so we check whether the
        # correct answer text appears in the retrieved context. Imperfect — it
        # rewards lexical overlap and misses paraphrase entirely. Name that
        # limitation in your README; stating a metric's weakness reads as
        # competence, not weakness.
        hit = q["answer_text"].lower() in context.lower()

        correct += is_correct
        retrieval_hits += hit
        records.append(
            {
                "qid": q.get("id"),
                "pred": pred,
                "gold": q["answer"],
                "correct": is_correct,
                "retrieval_hit": hit,
                "latency": latency,
            }
        )

    n = len(questions)
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "config": asdict(cfg),
        "config_hash": cfg.hash(),
        "n": n,
        "accuracy": correct / n,
        "retrieval_proxy": retrieval_hits / n,
        "mean_latency": sum(r["latency"] for r in records) / n,
        "records": records,
    }

    out = RESULTS / f"{result['timestamp']}_{cfg.hash()}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[{label or cfg.hash()}]  acc={result['accuracy']:.3f}  "
        f"retr={result['retrieval_proxy']:.3f}  "
        f"lat={result['mean_latency']:.2f}s  -> {out}"
    )
    return result


if __name__ == "__main__":
    cfg = Config()
    questions = load_questions()

    # TODO(day 4): construct your system and run both of these.
    #   run_eval(cfg, questions, rag_system,        label="rag-baseline")
    #   run_eval(cfg, questions, no_retrieval_sys,  label="control-no-retrieval")
    #
    # Commit eval/results/. Those files are the evidence behind your README
    # table — a reviewer who sees timestamped JSONs with embedded configs knows
    # immediately the numbers weren't invented.
    raise SystemExit("wire up your system first (Day 4)")
