# Scope card

Written on Day 1, before any RAG code. This is a product requirement, not a
legal footnote — `rag/safety.py` enforces it and the Tier 2 eval tests it.

## In scope

- Explaining what a symptom pattern commonly indicates
- Describing conditions, mechanisms, typical presentations
- Explaining what a test result or diagnosis means in general terms
- Advising on urgency level: self-care / see a GP / urgent care / emergency

## Out of scope — refuse, with a redirect

- Drug dosing, drug interactions, prescriptions
- Pediatric cases (under 18)
- Pregnancy-related presentations
- Mental health crisis / self-harm → crisis resources, not triage
- "Should I stop taking X?" — medication change decisions
- Interpreting specific lab values or imaging for a named individual

## Always

- Disclaimer on `/start` and in the first substantive response of each session
- Red-flag screening runs before retrieval, before the agent loop, before
  anything else
- Cite the source of every substantive claim
- When retrieved context is insufficient, say so rather than filling the gap

## Why this exists

Most portfolio medical bots will happily hand out a drug dosage. Refusing —
deliberately, with the reasoning documented — is a differentiator that costs a
paragraph. Demonstrating you thought about deployment safety unprompted is a
stronger hiring signal than the RAG pipeline itself.
