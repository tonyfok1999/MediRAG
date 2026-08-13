# MediRAG

A Telegram bot that takes an informal symptom description, asks clarifying
questions like a clinician taking a history, retrieves from a medical corpus,
and explains likely explanations with sources and an explicit scope boundary.

> **Status:** skeleton. Sections marked TODO get filled as the build progresses.

---

## Try it

TODO(day 13) — `t.me/yourbot`

TODO(day 15) — demo GIF: normal case, clarifying question, red-flag
short-circuit, out-of-scope refusal.

---

## Architecture

```
Telegram user
     │
     ▼
bot/main.py          long polling, per-chat_id session state
     │
     ▼
rag/safety.py        red-flag screen (deterministic → LLM)
     │               └─► EMERGENCY, short-circuit, no RAG
     ▼
agent/loop.py        decide: ASK a clarifying question, or ANSWER
     │               └─► ASK ──► back to user, update slots
     ▼
rag/rewriter.py      patient language → clinical terminology
     │
     ▼
rag/retriever.py     MedCPT query encoder → Qdrant top-k
     │
     ▼
rag/generator.py     API LLM: context + history + scope rules
     │
     ▼
  response + citations
```

Two deliberate choices:

1. **The red-flag screener is deterministic and runs first.** It never depends
   on the LLM choosing to be careful.
2. **Query rewriting sits between the conversation and the retriever.** MedCPT's
   query encoder truncates at 64 tokens, and real patient messages exceed that.
   Without the rewrite step, retrieval runs on a truncated fragment.

---

## Stack

| Layer | Choice |
|---|---|
| Corpus | `MedRAG/textbooks` — 18 books, ~125.8k snippets, ~182 chars avg |
| Embeddings | MedCPT dual encoder (`ncbi/MedCPT-Query-Encoder` + `-Article-Encoder`) |
| Vector DB | Qdrant |
| Lexical | `rank-bm25` + Reciprocal Rank Fusion (Day 10) |
| Generation | API LLM, `temperature=0` |
| Agent | Plain Python — no LangChain, see below |
| Interface | `python-telegram-bot` v21+ (async, long polling) |
| Eval | MedQA sample + hand-written vignettes, custom harness |

**Why no LangChain.** The retrieval and agent logic here is a couple hundred
lines. Writing it directly means the behaviour is inspectable and explainable;
the framework would abstract away exactly the layers worth understanding.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in TELEGRAM_TOKEN and LLM_API_KEY

python -m ingest.download   # inspect the corpus
python -m ingest.embed      # build the index — 1-3 hours on CPU
python -m bot.main          # run the bot
```

Get a bot token from [@BotFather](https://t.me/botfather) → `/newbot`.

---

## Results

TODO(day 4+). One row per measured change. Negative results stay in the table.

| Config | Accuracy | Retrieval proxy | Δ vs baseline |
|---|---|---|---|
| No retrieval (control) | — | — | — |
| Baseline: textbooks, MedCPT, top_k=5 | — | — | — |

Generated from `eval/results/*.json`.

---

## Evaluation

Two tiers, because the available benchmark doesn't match the product.

**Tier 1 — pipeline.** 150 MedQA questions, fixed seed, committed. Automatic,
cheap, runs on every change. Measures retrieval quality; does *not* measure
conversational ability. The retrieval proxy (correct answer string present in
context) substitutes for Recall@k since MedQA has no gold passages — it rewards
lexical overlap and misses paraphrase.

**Tier 2 — conversational.** 20 hand-written vignettes with a scripted patient
simulator and an LLM judge. Measures differential accuracy, urgency calibration
(under-triage weighted far more heavily than over-triage), whether the key
discriminating question got asked, and refusal correctness.

---

## Safety

See [`scope.md`](scope.md). Enforced in `rag/safety.py`, tested by the Tier 2
adversarial vignettes.

This bot does not diagnose. It refuses drug dosing, pediatric, pregnancy, and
medication-change questions, and routes red flags straight to emergency
guidance without running retrieval.

---

## Known limitations

TODO(day 14). Candidates: in-memory session store, English only, textbook
corpus has a US-exam bias, LLM judge is unvalidated at low n, no clinician
review.

---

## License

See [LICENSE](LICENSE).
