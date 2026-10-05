# ContextCrunch — Complete Definition Document
**Agent Context Compaction Service using TypeSafe Jev (System One Model)**

*Version 1.0 · September 2026 · Internal Definition Phase*

---

## 1. Executive Summary

### 1.1 What This Is

ContextCrunch is a **generic middleware service** that sits between any agentic AI system and its LLM, continuously compressing the agent's conversation history by making fast, typed decisions on every history item: **KEEP** (verbatim), **TRUNCATE** (shrink), or **DROP** (remove with tombstone).

It uses **TypeSafe Jev** — a newly released "System One" model (September 2026) that returns calibrated, type-safe probabilistic judgements in 70–500ms at ~1/100th the cost of LLM calls — instead of the traditional approach of asking an LLM to summarize history.

### 1.2 The Core Insight

> **Summarization rewrites evidence into lossy prose. Compaction filters evidence, keeping originals verbatim.**

Traditional approach: When context grows, ask an LLM to "summarize the conversation so far." Problems: slow (3–30s), expensive (output tokens ~5x input), hallucination-prone (new prose invents/paraphrases), coarse (emergency all-or-nothing at window limit).

ContextCrunch approach: For each history item, ask Jev five focused semantic questions about its *relation to the goal and other items*. Jev returns probabilities. Code applies confidence-gated policies. What survives is the **original content**, not a retelling. No hallucination can enter kept context.

### 1.3 Expected Impact

| Metric | Target |
|---|---|
| Context token reduction | 75–90% |
| Input token cost per agent run | ~6x reduction |
| Latency added per compaction | < 500ms (parallel Jev calls) |
| Dangerous drop rate (human-would-keep) | < 1% |
| Deployment scope | One gateway deployment → all internal agents |

---

## 2. Problem Definition

### 2.1 The Universal Agent Context Problem

Every long-running agentic system exhibits the same pattern:

```
Agent Step 1:  fetch 5000 tokens of data  →  extract 50-token insight
Agent Step 2:  fetch 3000 tokens of data  →  extract 40-token insight
Agent Step 3:  fetch 8000 tokens of data  →  extract 60-token insight
...
Agent Step N:  LLM receives 50,000+ tokens of history, 95% stale noise
```

**Concrete example from a real coding agent trace (8,465 tokens):**

| Content Type | Tokens | % of Total |
|---|---|---|
| Tool outputs (file dumps, logs, grep, test runs) | 7,300 | 86% |
| Actual insights (goal, findings, fix, verification) | 250 | 3% |
| System prompt + user messages | 915 | 11% |

The agent re-pays for all 8,465 tokens on **every subsequent LLM call**. At 100 steps, that's ~800k input tokens — mostly re-reading its own noise.

### 2.2 Why Existing Solutions Fail

| Approach | Fatal Flaw |
|---|---|
| **LLM Summarization** | Generates new prose → paraphrases/hallucinates exact evidence (stack traces, IDs, numbers, citations). Once summarized, verbatim original is gone. |
| **Sliding Window** | Drops old context blindly — loses the goal, key findings, commitments. |
| **Embedding-based Retrieval** | Adds infrastructure complexity; still sends full context to LLM; semantic search ≠ "is this consumed?" |
| **Manual Context Management** | Requires agent developers to instrument every agent; inconsistent; doesn't scale across teams. |

### 2.3 Why This Is a Gateway Problem, Not an Agent Problem

Every agent — regardless of framework (LangGraph, CrewAI, Claude Code, custom), language, or domain — ultimately sends messages to an LLM over standard protocols (OpenAI Chat Completions, Anthropic Messages, Responses API). **Intercepting at the protocol boundary means one deployment covers all agents with zero agent-side changes.** This is exactly how LiteLLM's `jev-compaction` guardrail already works in production.

---

## 3. Why Jev — The Technical Rationale

### 3.1 What Jev Is (and Isn't)

| Property | Jev (System One) | Traditional LLM |
|---|---|---|
| **Interface** | State in → typed decisions out | Prompt in → prose out |
| **Output** | Noul (probability), Choice (option + distribution), Score (level + distribution) | Free-form text / JSON |
| **Latency** | 70–500ms | 3–30s+ |
| **Cost** | $0.042/MTok input, output free | $0.20–10/MTok input, output ~5x |
| **Hallucination** | Schema-safe by design (bounded output space) | Can invent values, malformed JSON |
| **Confidence** | First-class calibrated output | Unreliable, overconfident |
| **Parallelism** | Questions evaluated in parallel | Sequential token generation |

**Key quote from TypeSafe:** *"Code calculates. Jev judges. LLMs reason."*

### 3.2 The Jev Suitability Test — Why Compaction Scores 6/6

| Criterion | Compaction | Score |
|---|---|---|
| **Judgement** — Is AI deciding rather than creating? | Yes — keep/truncate/drop is a decision | ✓ |
| **Bounded** — Can answer space be defined beforehand? | Yes — exactly 3 options | ✓ |
| **Atomic** — One focused judgement per item? | Yes — per history item | ✓ |
| **Context-contained** — All info in state? | Yes — item + goal + later findings | ✓ |
| **Fast-human** — Could expert judge in 5 seconds? | Yes — "glance at this step, still needed?" | ✓ |
| **Machine-consumed** — Software uses result directly? | Yes — code applies verdict | ✓ |

**This is arguably the most Jev-shaped task that exists.**

### 3.3 Why Not an LLM Classifier?

You *could* prompt an LLM to output `{"verdict": "drop", "confidence": 0.9}`. But:
- **No calibration guarantee** — LLM confidence is notoriously uncalibrated
- **Schema violations** — LLMs still emit malformed JSON, invented values
- **Cost/latency** — 100x more expensive, 50x slower
- **No parallel evaluation** — sequential generation
- **Prompt brittleness** — small prompt changes flip behaviour unpredictably

Jev's RLCD (Reinforcement Learning for Calibrated Decisions) training explicitly optimizes for **epistemically honest probabilities** — groups of predictions at 0.9 confidence are correct ~90% of the time. This is what makes confidence gates trustworthy.

---

## 4. Architecture Overview

### 4.1 High-Level Data Flow

```mermaid
flowchart LR
    A[Agent] -->|1. Request + History| B[ContextCrunch Gateway]
    B -->|2. Split: pinned + reviewable| C[State Builder per item]
    C -->|3. Fan-out Jev calls (parallel)| D[Jev API]
    D -->|4. Typed answers + probabilities| E[Policy Engine]
    E -->|5. KEEP/TRUNCATE/DROP + gates| F[Rebuild Compacted History]
    F -->|6. Forward to LLM| G[LLM]
    G -->|7. Response| B
    B -->|8. Response| A
    E -.->|Side-car log| H[(Side-car Store)]
    H -.->|Recovery on failure| E
```

### 4.2 Deployment Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    ORG LLM GATEWAY                           │
│  (existing proxy: LiteLLM / custom / cloud provider)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  CONTEXTCRUNCH MIDDLEWARE                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Screening  │  │  Compaction │  │      Routing        │  │
│  │  (future)   │  │   (THIS)    │  │    (future)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │      LLM PROVIDERS     │
              │  OpenAI / Anthropic /  │
              │  Internal / Azure      │
              └────────────────────────┘
```

**Integration point:** A single pre-call hook in the existing LLM gateway. No agent changes. No new infrastructure beyond the Jev API client.

### 4.3 Components

| Component | Responsibility | Domain-Specific? |
|---|---|---|
| **State Builder** | Packages each trace item into Jev state | No — pure protocol packaging |
| **Question Set** | The 5 constant questions sent to Jev | No — universal relations |
| **Policy Engine** | Applies confidence gates, thresholds, profiles | No — config only |
| **TRUNCATE Refiner** | Second-pass recursion for large items | No — same questions, finer granularity |
| **Side-car Store** | Logs every verdict + original; enables recovery | No — generic key-value |
| **Profile Registry** | Maps agent/team → policy profile | Yes — 5 numbers + 1 flag per archetype |

---

## 5. The Question Set — The Constant Engine

### 5.1 The Five Questions (Never Change Across Domains)

```python
QUESTION_SET = [
    # 1. Holistic recommendation (advisory — code makes final call)
    {"id": "verdict", "type": "choice",
     "question": "What should happen to this item for the ongoing goal?",
     "options": ["keep", "truncate", "drop"]},

    # 2. Safety veto: can this be re-obtained?
    {"id": "essential", "type": "noul",
     "question": "Does this item contain information the agent cannot recover if removed?"},

    # 3. Primary junk detector: insight already extracted?
    {"id": "consumed", "type": "noul",
     "question": "Has the useful information in this item already been extracted "
                 "into one of the later findings shown in the state?"},

    # 4. Secondary junk detector: made obsolete by later item?
    {"id": "superseded", "type": "noul",
     "question": "Has this item been made obsolete by a later item or finding?"},

    # 5. Graded importance — also ranks truncation priority
    {"id": "relevance", "type": "score",
     "question": "How relevant is this item to the current goal?",
     "levels": [
        "0: Noise or boilerplate with no future use",
        "1: Background — unlikely to be needed again",
        "2: Supporting — may be referenced later",
        "3: Critical — the agent will need this to finish",
     ]},
]
```

### 5.2 Why These Five Questions

| Question | Relation Probed | Why Universal |
|---|---|---|
| `verdict` | item ↔ goal | Every agent has a goal |
| `essential` | item ↔ its source tool | Every tool call is re-issuable (or not) |
| `consumed` | item ↔ later assistant messages | Every agent extracts small insights from big fetches |
| `superseded` | item ↔ later items | Every agent re-runs things (tests, queries, searches) |
| `relevance` | item ↔ goal (graded) | Importance grading needs no domain vocabulary |

**None of these questions mention code, customers, finance, or any domain.** They probe *structural relations in any agent trace*.

### 5.3 The State Schema (What Goes Into Each Jev Call)

```jsonc
{
  "goal": "Assess whether Acme's gross-margin expansion is sustainable...",
  "next_step": "Check supplier concentration risk, then finalize memo draft.",
  "item": {
    "index": 5,
    "role": "tool",
    "tool_call": "fetch_filing('ACME', '10-K', 'FY2025')",
    "content": "UNITED STATES SECURITIES AND EXCHANGE COMMISSION ... 38 pages ...",
    "tokens": 14000,
    "age_in_turns": 10
  },
  "later_findings": [
    "Extracted: GM 42%→47% over 3 yrs; driver = component-cost renegotiation.",
    "Competitors' GM flat; Acme's gap is procurement-driven, not price-driven.",
    "Two suppliers = 61% of COGS. Renegotiation credible, but concentration is a real risk."
  ]
}
```

| Field | Source | Purpose |
|---|---|---|
| `goal` | Pinned user message(s) | Anchor for relevance |
| `next_step` | Last assistant message | Future-need signal |
| `item` | The trace item on trial | The evidence being judged |
| `item.tool_call` | Tool metadata | Recoverability signal |
| `later_findings` | Assistant messages AFTER this item | **Critical** — enables "consumed" judgement |

**`later_findings` is the secret sauce.** It's not computed — it's just copied: all assistant messages that appear *after* the item in the trace. This lets Jev see that the 14,000-token 10-K already had its key facts extracted into message #7, #10, #13.

---

## 6. Policy Engine — Where Thresholds Live

### 6.1 The Decision Logic (Pure Code, No Model)

```python
PROFILES = {
    # Profile name: thresholds for DROP gate
    # drop if: essential < ess AND relevance <= rel AND consumed >= con AND confidence > conf
    "aggressive":   dict(ess=0.50, rel=1.5, con=0.80, conf=0.70, K=2, anchors=False),
    "balanced":     dict(ess=0.40, rel=1.0, con=0.85, conf=0.80, K=4, anchors=False),
    "conservative": dict(ess=0.30, rel=1.0, con=0.90, conf=0.90, K=3, anchors=True),
}

def decide(item, answers, profile):
    # --- Hard code rules (never consult model) ---
    if item["role"] in ("system", "user"):
        return "KEEP"                    # system pinned; human turns irrecoverable
    if item["age_in_turns"] <= profile["K"]:
        return "KEEP"                    # recency protection
    if item["tokens"] < 40:
        return "KEEP"                    # not worth a decision

    # --- DROP requires ALL gates to pass ---
    droppable = (
        answers["verdict"]["choice"] == "drop"
        and answers["essential"]["probability"]  < profile["ess"]
        and answers["relevance"]["score"]        <= profile["rel"]
        and answers["consumed"]["probability"]   >= profile["con"]
        and answers["verdict"]["confidence"]     > profile["conf"]
    )
    if droppable:
        return "DROP"

    # --- TRUNCATE on model recommendation or budget override ---
    if answers["verdict"]["choice"] == "truncate" or item["tokens"] > 800:
        return "TRUNCATE"

    return "KEEP"
```

### 6.2 The Gate Philosophy

```
MODEL PROPOSES          CODE DISPOSES
────────────────────────────────────────
verdict = "drop"    →   but essential=0.6?  → VETO (keep)
confidence = 0.99   →   but consumed=0.3?   → VETO (keep)
verdict = "keep"    →   but tokens=5000?    → OVERRIDE (truncate)
```

**The model never has the final word on anything irreversible.** This is what makes it deployable.

### 6.3 Policy Profiles — The Only Domain Configuration

| Knob | Aggressive (Coding) | Balanced (Support) | Conservative (Research) |
|---|---|---|---|
| `ess` (essential ceiling) | 0.50 | 0.40 | 0.30 |
| `rel` (relevance ceiling) | 1.5 | 1.0 | 1.0 |
| `con` (consumed floor) | 0.80 | 0.85 | 0.90 |
| `conf` (confidence floor) | 0.70 | 0.80 | 0.90 |
| `K` (protected recent turns) | 2 | 4 | 3 |
| `anchors` (preserve citations) | False | False | **True** |

**That's it. Five numbers and one flag per archetype.** No new questions, no new code paths.

---

## 7. The TRUNCATE Recursion — Second Pass at Finer Granularity

When an item is large (>800 tokens) or Jev recommends `truncate`, we don't just blindly clip — we **recurse the same questions at sub-item granularity**.

### 7.1 How It Works

```python
async def refine(item, goal, profile):
    """Split large item → judge each sub-item with SAME questions → rebuild."""
    sub_items = split_by_structure(item)  # deterministic, no model
    
    # Fan out: same QUESTION_SET per sub-item
    states = [build_sub_state(sub, goal, item["later_findings"]) for sub in sub_items]
    results = await asyncio.gather(*[jev.decide(s, QUESTION_SET) for s in states])
    
    kept = []
    for sub, ans in zip(sub_items, results):
        action = decide(sub, ans, profile)
        if action == "KEEP" or (profile["anchors"] and ans["relevance"]["score"] >= 2.0):
            kept.append(prefix_with_anchor(sub))  # keep verbatim + page/section ref
        elif action == "TRUNCATE":
            kept.append(clip_head_tail(sub))      # deterministic clip
        else:
            kept.append(tombstone(sub))           # "[removed: judged irrelevant]"
    
    return "\n---\n".join(kept)
```

### 7.2 Splitting Rules (Deterministic, No Model)

| Item Type | Split Strategy |
|---|---|
| JSON array (invoices, search results) | One element per sub-item |
| Markdown/HTML document | By heading level (## sections) |
| Source code file | By function/class (tree-sitter) |
| Log file | By timestamp chunks (e.g., 500 lines) |
| Plain text | By paragraph (blank-line split) |

**The same five questions apply at any granularity** because they probe relations, not content. A "sub-item" is just an item.

### 7.3 Anchor Preservation (Conservative Profile Only)

When `anchors=True` (research/legal/finance), any sub-item with `relevance >= 2.0` or `essential > 0.5` is kept **verbatim with its page/section reference prefixed**. The memo can still cite "p. 22" and "p. 87" exactly — the evidence survives even though 94% of the document is gone.

---

## 8. Worked Example — End-to-End (Financial Research Agent)

### 8.1 The Scenario

**Goal:** "Assess whether Acme's gross-margin expansion is sustainable; draft memo section with citations."

**Trace (abbreviated):**

| # | Role | Content | Tokens |
|---|---|---|---|
| 1 | SYSTEM | Analyst persona, citation rules | 350 |
| 2 | USER | Assess Acme margin sustainability... | 45 |
| 3 | ASSISTANT | Plan: 10-K → transcript → competitors → draft | 60 |
| 4 | TOOL | Web search "Acme gross margin" | 700 |
| 5 | TOOL | **Fetch Acme 10-K (38 pages)** | **14,000** ← *judged now* |
| 6 | TOOL | Earnings call transcript | 7,500 |
| 7 | ASSISTANT | Extracted: GM 42%→47% over 3 yrs; driver = renegotiation | 60 |
| 8 | TOOL | Market data API | 1,200 |
| 9 | TOOL | Competitor 10-K | 11,000 |
| 10 | ASSISTANT | Competitors flat; gap is procurement-driven | 55 |
| 11 | USER | Also check supplier concentration risk | 30 |
| 12 | TOOL | Supplier disclosures | 2,800 |
| 13 | ASSISTANT | Two suppliers = 61% COGS; concentration risk real | 55 |
| 14 | TOOL | Citation verification | 600 |
| 15 | ASSISTANT | Draft memo section (with citations) | 900 |

### 8.2 Judging Item #5 (the 10-K) — Request

```jsonc
{
  "state": {
    "goal": "Assess whether Acme's gross-margin expansion is sustainable; draft the memo section with citations.",
    "next_step": "Check supplier concentration risk, then finalize the memo draft.",
    "item": {
      "index": 5,
      "role": "tool",
      "tool_call": "fetch_filing('ACME', '10-K', 'FY2025')",
      "content": "UNITED STATES SECURITIES AND EXCHANGE COMMISSION ... 38 pages ...",
      "tokens": 14000,
      "age_in_turns": 10
    },
    "later_findings": [
      "Extracted: GM 42%→47% over 3 yrs; driver = component-cost renegotiation.",
      "Competitors' GM flat; Acme's gap is procurement-driven, not price-driven.",
      "Two suppliers = 61% of COGS. Renegotiation credible, but concentration is a real risk."
    ]
  },
  "questions": [ /* QUESTION_SET — identical for all domains */ ]
}
```

### 8.3 Jev Response

```jsonc
{
  "answers": {
    "verdict":    { "choice": "truncate", "confidence": 0.81,
                    "probabilities": { "keep": 0.31, "truncate": 0.58, "drop": 0.11 } },
    "essential":  { "probability": 0.55 },
    "consumed":   { "probability": 0.87 },
    "superseded": { "probability": 0.19 },
    "relevance":  { "score": 2.4, "confidence": 0.77,
                    "probabilities": { "0": 0.03, "1": 0.14, "2": 0.61, "3": 0.22 } }
  }
}
```

### 8.4 Policy Math (Conservative Profile)

```
DROP gates:
  verdict.choice == "truncate"      → not "drop"        ✗ VETO
  essential 0.55 — needs < 0.30     → fails             ✗ VETO
  (citable passages must survive — Jev raised essential appropriately)
─────────────────────────────────────────────
DROP blocked → TRUNCATE, and profile.anchors=True → anchor-preserving second pass
```

### 8.5 Second Pass — Per-Passage Judgements

The 10-K splits into ~60 passages (by section). Same `QUESTION_SET` per passage.

| Passage | Relevance | Essential | Outcome |
|---|---|---|---|
| MD&A: "Gross margin improved 500bps, driven primarily by renegotiated component pricing…" (p. 22) | 2.9 | 0.71 | **KEEP verbatim + "p. 22"** |
| Supplier note: "Our two largest suppliers accounted for 61% of cost of goods sold…" (p. 87) | 3.0 | 0.82 | **KEEP verbatim + "p. 87"** |
| Risk-factor boilerplate (p. 12) | 0.6 | 0.09 | Tombstone |
| Executive compensation tables (p. 45) | 0.3 | 0.05 | Tombstone |

**Result: 14,000 tokens → ~800 tokens of citable anchors. 94% reduction.**

### 8.6 Full Trace Compaction Result

| Item | Original | Compacted | Action |
|---|---|---|---|
| 1 SYSTEM | 350 | 350 | PINNED |
| 2 USER | 45 | 45 | KEEP (human) |
| 3 ASSISTANT | 60 | 60 | KEEP (plan) |
| 4 TOOL (search) | 700 | 0 | DROP (superseded by targeted fetches) |
| 5 TOOL (10-K) | 14,000 | 800 | TRUNCATE (anchors) |
| 6 TOOL (transcript) | 7,500 | 400 | TRUNCATE (CFO quotes only) |
| 7 ASSISTANT | 60 | 60 | KEEP (extraction) |
| 8 TOOL (market data) | 1,200 | 200 | TRUNCATE (annual rows only) |
| 9 TOOL (competitor) | 11,000 | 600 | TRUNCATE (comparison passages) |
| 10 ASSISTANT | 55 | 55 | KEEP (finding) |
| 11 USER | 30 | 30 | KEEP (human) |
| 12 TOOL (suppliers) | 2,800 | 300 | TRUNCATE (concentration figures) |
| 13 ASSISTANT | 55 | 55 | KEEP (risk finding) |
| 14 TOOL (citations) | 600 | 0 | DROP (process artifact) |
| 15 ASSISTANT | 900 | 900 | KEEP (artifact being built) |
| **TOTAL** | **39,855** | **~3,855** | **~90% reduction** |

---

## 9. Genericity Proof — Three Domains, One Engine

The same engine (questions, code, policy structure) runs on three maximally different agents:

| Domain | Agent Job | Profile | Reduction | Key Difference |
|---|---|---|---|---|
| **Coding** | Fix rounding bug in billing service | `aggressive` | ~90% | Re-readable files → low `essential` |
| **Support** | Resolve double-charge + discount dispute | `balanced` | ~75% | Human in loop → wider `K`, keep commitments |
| **Research** | Draft margin-sustainability memo | `conservative` | ~91% | Citations required → `anchors=True` |

**What changes between domains:** only the profile row (5 numbers + 1 flag).  
**What never changes:** the 5 questions, the state builder, the gate logic, the refiner.

### 8.1 Coding Agent (Aggressive Profile) — Key Verdicts

| Item | Verdict | Why |
|---|---|---|
| `read_file` dump (2,400 tok) | **DROP** | Consumed into finding; re-readable |
| `pip install` log (1,600 tok) | **DROP** | Process boilerplate |
| `pytest` run 1 (1,100 tok) | **DROP** | Superseded by run 2 |
| `pytest` run 2 (1,300 tok) | **TRUNCATE** | Keep "147/147 pass" only |
| Human USER messages | **KEEP** | Universal rule |

### 8.2 Support Agent (Balanced Profile) — Key Verdicts

| Item | Verdict | Why |
|---|---|---|
| Invoice history (14 invoices) | **TRUNCATE** → 2 kept, 12 dropped | Keep June + current month only |
| KB search results | **DROP** | Consumed into finding; re-fetchable |
| Billing ledger (6 months) | **DROP** | Consumed into action taken |
| KB article (discount policy) | **TRUNCATE** | Kept for justification if disputed |
| Human turns (customer) | **KEEP** | Universal rule |

### 8.3 Research Agent (Conservative Profile) — Key Verdicts

| Item | Verdict | Why |
|---|---|---|
| 10-K (14,000 tok) | **TRUNCATE + anchors** | Keep MD&A p.22 + supplier note p.87 verbatim |
| Transcript (7,500 tok) | **TRUNCATE** | CFO quotes only |
| Competitor 10-K (11,000 tok) | **TRUNCATE** | Comparison passages with page refs |
| Supplier disclosures (2,800 tok) | **TRUNCATE** | Concentration figures + refs |
| Citation verification report | **DROP** | Process artifact; memo has quotes |

---

## 10. Implementation Details

### 10.1 Core Module Structure

```
contextcrunch/
├── __init__.py
├── config.py              # Profiles, constants, Jev client config
├── state_builder.py       # build_state(), build_sub_state()
├── questions.py           # QUESTION_SET constant
├── policy.py              # decide(), PROFILES, gate logic
├── refiner.py             # refine(), split_by_structure()
├── sidecar.py             # DecisionLog, SidecarStore
├── gateway_hook.py        # Pre-call hook for LLM gateway
├── jev_client.py          # Thin wrapper around Jev HTTP/SDK
└── main.py                # Compact() orchestration
```

### 10.2 Jev Client Wrapper (Isolates API Drift)

```python
# jev_client.py
import httpx
from dataclasses import dataclass

@dataclass
class JevAnswer:
    choice: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    probability: float | None = None
    score: float | None = None

@dataclass
class JevResponse:
    answers: dict[str, JevAnswer]

class JevClient:
    def __init__(self, api_key: str, base_url: str = "https://api.typesafe.ai/v1"):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )
    
    async def decide(self, state: dict, questions: list[dict]) -> JevResponse:
        resp = await self.client.post("/decisions", json={
            "state": state,
            "questions": questions,
            "schema_version": "v1"
        })
        resp.raise_for_status()
        return self._parse(resp.json())
    
    def _parse(self, data: dict) -> JevResponse:
        answers = {}
        for qid, raw in data.get("answers", {}).items():
            answers[qid] = JevAnswer(
                choice=raw.get("choice"),
                confidence=raw.get("confidence"),
                probabilities=raw.get("probabilities"),
                probability=raw.get("probability"),
                score=raw.get("score"),
            )
        return JevResponse(answers=answers)
```

### 10.3 State Builder

```python
# state_builder.py
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def token_count(text: str) -> int:
    return len(enc.encode(text))

def build_state(trace: list[dict], idx: int, goal: str) -> dict:
    item = trace[idx]
    return {
        "goal": goal,
        "next_step": trace[-1]["content"] if trace[-1]["role"] == "assistant" else "",
        "item": {
            "index": idx,
            "role": item["role"],
            "tool_call": item.get("tool_call"),
            "content": item["content"],
            "tokens": token_count(item["content"]),
            "age_in_turns": len(trace) - idx,
        },
        "later_findings": [
            m["content"] for m in trace[idx+1:] if m["role"] == "assistant"
        ],
    }

def reviewable_indices(trace: list[dict]) -> list[int]:
    return [i for i, m in enumerate(trace) if m["role"] not in ("system",) and i > 0]
```

### 10.4 Policy Engine

```python
# policy.py
from dataclasses import dataclass
from jev_client import JevAnswer

@dataclass
class Profile:
    ess: float
    rel: float
    con: float
    conf: float
    K: int
    anchors: bool

PROFILES = {
    "aggressive":   Profile(0.50, 1.5, 0.80, 0.70, 2, False),
    "balanced":     Profile(0.40, 1.0, 0.85, 0.80, 4, False),
    "conservative": Profile(0.30, 1.0, 0.90, 0.90, 3, True),
}

def decide(item: dict, answers: dict[str, JevAnswer], profile: Profile) -> str:
    if item["role"] in ("system", "user"):
        return "KEEP"
    if item["age_in_turns"] <= profile.K:
        return "KEEP"
    if item["tokens"] < 40:
        return "KEEP"
    
    v = answers["verdict"]
    if (v.choice == "drop"
        and answers["essential"].probability < profile.ess
        and answers["relevance"].score <= profile.rel
        and answers["consumed"].probability >= profile.con
        and v.confidence > profile.conf):
        return "DROP"
    
    if v.choice == "truncate" or item["tokens"] > 800:
        return "TRUNCATE"
    
    return "KEEP"
```

### 10.5 Main Orchestration

```python
# main.py
import asyncio
from jev_client import JevClient
from state_builder import build_state, reviewable_indices
from policy import PROFILES, decide
from refiner import refine
from sidecar import SidecarStore

class ContextCrunch:
    def __init__(self, jev_client: JevClient, sidecar: SidecarStore):
        self.jev = jev_client
        self.sidecar = sidecar
    
    async def compact(
        self,
        trace: list[dict],
        goal: str,
        profile_name: str = "balanced"
    ) -> list[dict]:
        profile = PROFILES[profile_name]
        reviewable = reviewable_indices(trace)
        
        states = [build_state(trace, i, goal) for i in reviewable]
        results = await asyncio.gather(*[
            self.jev.decide(state, QUESTION_SET) for state in states
        ])
        verdicts = dict(zip(reviewable, results))
        
        compacted = []
        for i, item in enumerate(trace):
            if i not in verdicts:
                compacted.append(item)
                continue
            
            action = decide(item, verdicts[i].answers, profile)
            
            if action == "DROP":
                compacted.append(self._tombstone(item))
            elif action == "TRUNCATE":
                compacted.append(await refine(item, goal, profile))
            else:
                compacted.append(item)
            
            self.sidecar.log(i, item, verdicts[i].answers, action, profile_name)
        
        return compacted
    
    def _tombstone(self, item: dict) -> dict:
        return {
            "role": "system",
            "content": f"[Removed by ContextCrunch: {item.get('tool_call', 'item')} — judged no longer relevant]",
            "tool_call_id": item.get("tool_call_id"),
        }
```

### 10.6 Gateway Integration Hook

```python
# gateway_hook.py
async def pre_call_hook(request: dict, context: dict) -> dict:
    crunch = context["crunch_instance"]
    profile = context.get("profile", "balanced")
    goal = extract_goal(request["messages"])
    
    compacted = await crunch.compact(request["messages"], goal, profile)
    request["messages"] = compacted
    return request
```

---

## 11. Side-Car Store & Recovery

### 11.1 What Gets Logged (Every Decision)

```python
@dataclass
class DecisionLogEntry:
    trace_id: str
    item_index: int
    item_role: str
    item_tokens: int
    item_tool_call: str | None
    action: Literal["KEEP", "TRUNCATE", "DROP"]
    verdict_choice: str
    verdict_confidence: float
    essential_prob: float
    consumed_prob: float
    superseded_prob: float
    relevance_score: float
    relevance_confidence: float
    profile: str
    timestamp: datetime
    original_ref: str  # key in object store (S3/blob/DB)
```

### 11.2 Recovery on Failure

```python
async def recover_and_retry(trace_id: str, failure_step: int, sidecar: SidecarStore):
    dropped = sidecar.get_dropped(trace_id)
    dropped.sort(key=lambda e: e.relevance_score, reverse=True)
    restored = [sidecar.fetch_original(e.original_ref) for e in dropped[:3]]
    return inject_restored(restored, failure_step)
```

**Nothing is ever truly lost** — the side-car is the safety net.

---

## 12. Evaluation & Definition-Phase Deliverables

### 12.1 Week 1: Offline Validation (No Integration)

1. **Collect traces:** 5–10 real traces per archetype from existing internal agents (or Claude Code / Copilot sessions).
2. **Human labeling:** For ~100 items per trace, a human marks the "correct" verdict (keep/truncate/drop) — ~30 min/trace.
3. **Replay:** Run ContextCrunch offline on all traces.
4. **Diagnose by question, not verdict:** For every disagreement, check which gate caused it:
   - Dangerous drop (human=keep, engine=drop) → tighten `ess`/`conf`
   - Missed savings (human=drop, engine=keep) → usually `consumed` wording or `rel` threshold
   - Systematic misfire on content type → adjust question wording
5. **Converge:** Target dangerous-drop rate ≈ 0 at cost of some missed savings.

### 12.2 Week 2: Definition Package

| Deliverable | Format |
|---|---|
| Tuned `PROFILES` table + final `QUESTION_SET` | JSON + Python |
| Disagreement analysis per archetype | Spreadsheet |
| Benchmark slide: "X% reduction, Y dangerous drops, $Z saved per 1k runs" | 1-pager |
| 6-week PoC roadmap | Document |
| Risk register | Document |

---

## 13. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Jev API changes** (9 days old) | Medium | High | Thin `jev_client.py` wrapper; verify against `docs.typesafe.ai` weekly |
| **Calibration drift** | Low | High | Continuous eval on production traces; alert if dangerous-drop rate > 1% |
| **Data governance** (external API) | High | High | Synthetic/anonymized traces for PoC; security sign-off before prod; fail-open |
| **Agents with internal state** (LangGraph checkpoints) | Medium | Medium | L1 SDK adapters for major frameworks; gateway covers the rest |
| **Multimodal traces** (images, charts) | Low | Low | Out of scope v1; protocol-level question |
| **Over-aggressive drops in new domain** | Medium | Medium | Start conservative; profile tuning is the only domain work |

---

## 14. Competitive Landscape & Differentiation

| Competitor | What They Do | Our Differentiation |
|---|---|---|
| **LiteLLM `jev-compaction`** | Gateway guardrail, threshold 0.2, permanent drops | **Reversible side-car + recovery**, confidence-gated profiles, AgentGuard bundle (guard+compact+route) |
| **LLM Summarization** | Rewrite history into prose | **Filter, don't rewrite** — originals verbatim, no hallucination |
| **Sliding Window / RAG** | Drop old / retrieve relevant | **Semantic judgement per item** — knows what's consumed/superseded, not just "similar" |
| **Custom per-agent logic** | Each team builds their own | **One gateway, all agents** — zero agent changes, org-wide ROI |

---

## 15. Open Questions for Definition Review

1. **Jev API access:** Early access status, quota limits, SLA — need confirmation from TypeSafe.
2. **Data residency:** Does Jev API meet org data residency requirements? (Current: US West Coast)
3. **Pricing at scale:** $0.042/MTok input is launch pricing — confirm long-term commitment.
4. **Profile discovery:** Can we auto-suggest profile from agent metadata (tool types, avg trace length)?
5. **Observability integration:** Emit metrics to existing stack (Datadog/Prometheus) — token savings, drop rate, recovery rate per team.

---

## 16. Next Steps (If Approved)

| Week | Activity |
|---|---|
| 1–2 | Offline validation on real traces (Section 12.1) |
| 3 | Gateway integration prototype (LiteLLM hook or custom proxy) |
| 4 | Shadow mode: run compaction in parallel, compare LLM outputs |
| 5 | Canary: 1 team, 1 agent archetype, full logging |
| 6 | Rollout: profile registry, fleet dashboards, recovery drills |

---

## 17. Appendix: The "Why This Works" Mental Model

### The Three Universal Junk Patterns

Everything the engine drops falls into exactly three patterns — **in every domain**:

| Pattern | Coding | Support | Research |
|---|---|---|---|
| **Consumed verbosity** — big tool output whose insight was extracted into a small assistant message | `read_file` dump → finding | invoice history → duplicate identified | 10-K → margin figures |
| **Process boilerplate** — artifacts of *doing*, not *knowing* | pip install log | ticket-update confirmation | citation-check report |
| **Superseded intermediates** — earlier result made obsolete by later one | pytest run 1 (after run 2 passed) | old ledger (after credit action) | web-search results (after targeted fetches) |

These patterns are consequences of **how agents work**, not of **what domain the agent works in**. That is the deep reason the engine transfers.

### The Safety Argument

**A tool call is by definition re-issuable.** The agent can re-read the file, re-query the CRM, re-fetch the filing. The cost of a wrong drop is one re-fetch + one turn — never lost information. Plus the side-car keeps every original item + verdict, so nothing is ever truly gone.

### The One-Line Pitch

> **"We build the engine once. Domains are just configuration."**

> **"Three agents that could not be more different — same engine, same five questions. What changed: five numbers and one flag."**

---

*End of definition document. This is the single source of truth for ContextCrunch.*