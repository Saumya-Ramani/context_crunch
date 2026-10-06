# ContextCrunch — Architecture Diagram Prompt

A ready-to-paste prompt for generating the project architecture diagram. Paste the whole
block into ChatGPT (with image generation), ChatGPT-4o image gen, Gemini image generation,
Midjourney, Midlibrary / Miro AI diagram, or similar tools.

---

## PART 1 — The prompt (paste this whole block)

```
Create a clean, professional system architecture diagram for a software project called
"ContextCrunch". The diagram must communicate a SAFETY-CRITICAL design philosophy clearly,
because this is not an ordinary data pipeline.

===============================================================
SECTION A — OVERALL STYLE AND FORMAT
===============================================================

Canvas: wide landscape orientation, approximately 16:9 aspect ratio. High resolution,
crisp and sharp. White or very light off-white background.

Visual style: modern flat-design technical architecture diagram, similar to a polished
engineering documentation figure or a Cloud/Kubernetes architecture poster. Use clean
rounded rectangles for components with subtle drop shadows. Use thin, crisp connector
lines with small arrowheads. Use a calm, professional color palette: slate blue, teal,
soft indigo, warm amber for warnings/safety, muted red only for destructive operations.

Typography: modern sans-serif (Inter, Segoe UI, or Helvetica). Clear readable labels.
Every component box must have a bold title and a smaller one-line description below it.

CRITICAL LAYOUT RULE: The diagram must be organized into clear horizontal bands or
layers, with each layer given a label on the left edge. Read left to right and top to
bottom. Do not let any arrows cross each other. Do not let boxes overlap. Leave clear
white space between groups. Make sure NO text is cut off, clipped, or overlapping.

===============================================================
SECTION B — THE SYSTEM'S CORE IDEA (shown as a title/caption area)
===============================================================

Title at the top center, large and bold:

    "ContextCrunch — Agent Context Compaction Service"

Subtitle directly below, smaller:

    "Sits in front of an LLM call. Never rewrites text. Nothing is destroyed.
     Always fails open."

===============================================================
SECTION C — LAYER 1: THE HOST / CLIENT
===============================================================

Leftmost band, labeled "HOST APPLICATION":

Box: "Host Agent / Client"
  - A long AI agent session running for many turns
  - Accumulates a large message history: tool outputs, file dumps, install logs,
    search results
  - Keeps its own complete copy of the full history at all times

Draw a visual representation of a growing message history next to this box:
a tall vertical stack of small horizontal message bars, color-coded by type:
  - light grey = system messages
  - blue = user messages
  - green = assistant messages
  - amber = tool messages (these grow huge and are the target of compaction)

Emphasize visually that the amber "tool" bars become enormous — far larger than the
others — because tool outputs are the bulk of the context bloat.

===============================================================
SECTION D — LAYER 2: THE HTTP SERVICE
===============================================================

Band labeled "HTTP SERVICE":

Box: "API Layer — FastAPI  (api/main.py)"
  - POST /v1/compact   -> the single main endpoint
  - GET  /health       -> liveness check
  - GET  /v1/sessions/{id}/stats -> per-session decision history
  - Compiles components ONCE at startup and holds them for the process lifetime
  - This layer owns the FAIL-OPEN contract

Make this box stand out. Give it a distinct border (for example a thicker amber border)
to signal that it owns the fail-open guarantee.

===============================================================
SECTION E — LAYER 3: THE THREE AGENTS (this is the heart of the diagram)
===============================================================

Band labeled "CONTEXT CRUNCH AGENTS" — this band must be visually the largest and most
prominent band in the entire diagram. It is the core of the system.

Draw the three agents in a LEFT-TO-RIGHT SEQUENCE with numbered circular badges (1), (2),
(3) attached to each agent box.

--- AGENT 1 ---

Box title: "1. SCOUT"  (file: agents/scout.py)
Subtitle: "Rule-based. No model calls. Deterministic."
Contents listed inside the box, as bullet points:
  - PICK PROFILE  -> aggressive | balanced | conservative
  - EXTRACT GOAL  -> what is the agent trying to do
  - FIND PINNED TOOLS -> side-effecting tools that must never be compacted

Below the box, add a small callout note:
  "Chooses one of three threshold profiles based on keyword hits in the tool names
   present in the history. No signal, or a tie, falls back to conservative."

Add a small side annotation: "calls no LLM"

--- AGENT 2 ---

Box title: "2. GUARDIAN"  (file: agents/guardian.py)
Subtitle: "Rule-based. No model calls. Paranoid by design."
Bullet points inside:
  - REGRET RULE  -> an item removed before is needed again -> log mistake; after 2, force conservative
  - STUCK RULE   -> same error repeated 3+ times -> restore 3 most relevant recent removals
  - May ONLY make things SAFER (restore items, escalate profile) — never loosen thresholds

Below the box, add a small callout note:
  "Reads the side-car Store. Requires the same session_id as the Worker, otherwise it
   silently does nothing."

Add a small side annotation: "calls no LLM"

--- AGENT 3 ---

Box title: "3. WORKER / COMPACTOR"  (file: agents/worker.py)
Subtitle: "The ONLY component that edits the history. The only one that calls the model."
Make this box visually the largest box in the whole diagram.

Below the three agent boxes, draw a three-part horizontal sub-flow inside a light
container labelled "WORKER INTERNAL FLOW":

Sub-box A: "PLAN"
  - Count tokens. Below the trigger threshold -> return untouched.
  - Determine which messages are ELIGIBLE. Only "tool" messages are ever judged.
  - Auto-KEEP without any model call: system/user/assistant roles; items the Guardian
    asked back for; pinned tools; items within the last K turns

Sub-box B: "ACT"
  - Eligible items that fit the size limit are all judged in ONE parallel batch
  - Items too large to send whole are SPLIT into pieces first, each piece judged
    separately, then whatever survives is rejoined
  - EVERY reviewed item's ORIGINAL text is written to the side-car Store BEFORE
    it is dropped

Sub-box B2 (narrow, connected from ACT with a small downward arrow):
  "SPLIT-FIRST PATH"
    - deterministic, no model consulted
    - JSON arrays split one element per piece (lets a 14-invoice ledger drop 12,
      keep 2)
    - plain text split on paragraph boundaries
    - each piece carries an "anchor" naming where it came from

Sub-box C: "VERIFY"
  - Check the compaction's OWN output, do not trust it:
      same message count | same roles in same order | every tool_call_id still
      attached | no empty content
  - ANY failure -> return the ORIGINALS. A long history is recoverable;
    a malformed one is not.

===============================================================
SECTION F — THE FIVE QUESTIONS AND THE POLICY ENGINE
===============================================================

Place these as a distinct band or as a prominent side-section.

Heading: "THE FIVE QUESTIONS ASKED ABOUT EVERY ITEM"
Subtitle: "Defined once, used everywhere (core/questions.py)"
Draw as a vertical list of five uniform question cards, or as a small table with columns
[question id] [type] [what it asks]:

  1. verdict     | choice | keep / truncate / drop
  2. essential   | noul   | is it essential
  3. consumed    | noul   | has its content already been extracted
  4. superseded  | noul   | has a later result made it obsolete
  5. relevance   | score  | how relevant, 0-3

Important callout, placed right next to this list:
  "consumed is only answerable because the state sent to the model includes
   later_findings: assistant messages AFTER this item, nearest first. That lets the
   model see that a huge fetch was already distilled into a short later message.
   This is the most important input in the whole system."

--- THE POLICY ENGINE — make this visually critical ---

Box title: "POLICY ENGINE  (core/policy.py)"
Subtitle: "The model proposes. CODE disposes. Every drop gate is a VETO."

Draw four gate cards in sequence, each looking like a locked gate / barrier:

  GATE 1 — RECOMMENDATION
    "The model actually said 'drop'"

  GATE 2 — CONFIDENCE
    "The model was confident about that drop"
    Add a small annotation: "floor 0.70-0.90 depending on profile"

  GATE 2B — RELEVANCE (put this as a separate gate card)
    "Not essential, AND not relevant"
    Add annotation: "meaning it cannot be re-obtained if it goes"

  GATE 3 — JUNK REASON
    "At least one positive junk reason exists"
    Show the three reasons as three small chips inside this gate:
      consumed | superseded | noise

Very important annotation attached to the gates, in amber/amber-orange text:
  "ANY SINGLE GATE FAILING BLOCKS THE DROP AND THE ITEM IS KEPT.
   Missing answer = veto = keep. The model never gets the final word on anything
   it cannot take back."

Then draw a small three-way output fan-out from the policy engine:

Three outcome boxes side by side:
  - "KEEP"      -> original text, byte for byte
  - "DROP"      -> replaced by a short TOMBSTONE
  - "TRUNCATE"  -> surviving pieces rejoined

--- THE TOMBSTONE BOX ---

Box title: "TOMBSTONE  (core/tombstone.py)"
Subtitle: "What a dropped message turns into"
Show the literal text:
  "[Removed by ContextCrunch: <tool>. judged no longer relevant.
    Recoverable via sidecar.]"

Add a CRITICAL annotation in amber:
  "CRITICAL: the tombstone keeps role, name and tool_call_id. An assistant message
   carries a tool_calls entry pointing at a tool_call_id; if the matching tool result
   disappears or changes role, THE PROVIDER REJECTS THE WHOLE REQUEST."

===============================================================
SECTION G — THE MODEL LAYER (EXTERNAL, DEPENDENCY)
===============================================================

Band labeled "MODEL LAYER (external dependency)":

Box: "Laya — Structured-Output LLM"
Subtitle: "Runs the five questions over one item's state"
Add side annotation: "reached only via LayaEngine, which batches items and caps
in-flight concurrency"

Side annotation next to the engine:
  "The engine holds NO fail-open behaviour on purpose. The Worker lets exceptions
   escape and the API layer decides what to do. An engine that quietly returned a
   default answer would hide a real failure behind a plausible-looking verdict."

--- WHAT THE MODEL ACTUALLY SEES ---

Draw a small panel showing the four-field state handed to the model for ONE item:
  goal           -> what the agent is trying to do
  next_step      -> most recent assistant text (a hint for future need)
  item           -> the message on trial
  later_findings -> assistant messages after it, nearest first

Add a note:
  "item.tokens reports the FULL untruncated size while item.content is only what the
   model is shown, so it can judge 'this is enormous' without reading all of it."

===============================================================
SECTION H — LAYER 5: THE SIDE-CAR STORE (PERSISTENCE)
===============================================================

Band labeled "PERSISTENCE — SIDE-CAR STORE":

Draw as a cylindrical database symbol for SQLite, combined with a file-folder tree.

Box A: "Store  (storage/store.py)"
  - SQLite database + directory of original messages
  - Decisions table: session_id, item_index, fingerprint, action, relevance
  - Mistakes table: session_id, fingerprint, item_index
  - Session ids are stored under a HASH of the id, never the raw id, because ids
    come from an agent and are not trusted to be path-safe

Box B: "SPLIT AND REBUILD PATH (core/splitter.py + core/refiner.py)"
  - Deterministic splitting — no model is consulted
  - Pieces carry an anchor naming their origin
  - If a rebuild is not genuinely shorter than the original, it is DISCARDED
    (a compaction that reports a saving it did not achieve is worse than one that
    admits it saved nothing)

===============================================================
SECTION I — LAYER 6: OUTPUT, AND THE HOST'S OWN LLM CALL
===============================================================

Band labeled "OUTPUT AND DOWNSTREAM LLM CALL":

Box: "Compacted History (or originals + fail_open=true)"
  - Same number of messages, same roles, same tool_call_ids
  - Same number of messages even when items are dropped, because DROP produces a
    tombstone, not a deletion

Drop an annotation showing token economics:
  "Offline evaluation: 0 dangerous drops, all planted facts survived, ~60% tokens saved.
   Against the real hosted model: only ~4% saved — the model answers 'drop' with
   confidence 0.4-0.5 while every profile sets a 0.7-0.9 confidence floor, so the
   gate vetoes almost every drop. This is a threshold problem, not a bug."

===============================================================
SECTION J — GLOBAL GUARANTEES STRIP
===============================================================

Along the very bottom of the diagram, run a full-width horizontal strip divided into three
equal panels, divided by vertical divider lines:

Panel 1: "NEVER REWRITES TEXT"
  "What is kept is byte-for-byte original. What is removed becomes a tombstone that keeps
   role, name and tool_call_id."

Panel 2: "NOTHING IS DESTROYED"
  "Every original is written to a side-car before it is dropped, so any removal can be
   undone."

Panel 3: "ALWAYS FAILS OPEN"
  "If anything goes wrong, the caller gets its own history back untouched and the LLM call
   proceeds. tokens_before == tokens_after, so the host can see nothing changed."

===============================================================
SECTION K — FINAL QUALITY REQUIREMENTS
===============================================================

1. NO TEXT MAY BE CLIPPED, OVERLAPPED, OR RUN OFF THE EDGE OF THE CANVAS. Leave generous
   margins. Image generators very commonly clip text, so keep every annotation short.
   Prefer short phrases over long sentences. Prefer symbols and icons over long prose.

2. Do not invent components, services, databases, clouds, or third-party vendors that are
   NOT explicitly named in this prompt. Accuracy matters more than visual richness.

3. All arrowheads must point in the direction of data flow. The main pipeline flows
   left to right. The Guardian and Store read from the flow and feed corrections back
   into it.

4. Use a consistent, restrained palette. Amber/orange should be reserved for SAFETY and
   WARNING semantics. Use red ONLY for destructive operations (DROP, destructive tools).
   Use teal/indigo for normal processing.

5. Make the "three agents" band unmistakably the visual center of gravity of the whole
   diagram. That is the story of this system.

6. Include small monospaced file paths as secondary labels (e.g. agents/worker.py) in a
   smaller, lighter grey font.
   Do NOT render them as if they were class names or endpoints.

7. Include, in the top-left corner, a small legend explaining the colors used.

===============================================================
END OF PROMPT
===============================================================
```

---

## PART 2 — Tips for getting a better result

| Tip | Why |
|---|---|
| Generate at the **largest size** the tool allows | Text clipping is the #1 failure mode of image generators |
| If text is garbled, ask for a **regeneration** rather than editing | Image models cannot reliably edit text in an image |
| For a pixel-perfect result, use a **diagram tool** instead | AI image generators cannot guarantee legible text |
| Alternative tools: Mermaid, PlantUML, Miro AI, diagrams.net, Figma, Excalidraw | These render exact text |

### If you need guaranteed-correct text

AI image generators are unreliable with text. If the diagram is for something important,
generate the visual layout from this prompt, then recreate the exact boxes and labels in
one of the tools above so the labels are guaranteed correct.

### Ready-made Mermaid source you can render directly

If you'd rather skip image generation entirely and get a guaranteed-correct diagram,
append this to the prompt block above, or render it in VS Code (VS Code's Mermaid support
renders `.md` files with mermaid blocks directly):

VS Code renders Mermaid automatically in Markdown preview, and
[UNDERSTANDING.md](contextcrunch/md/UNDERSTANDING.md) already contains a Mermaid block —
preview that file to see it render.

---

## PART 2b — Full detailed Mermaid source for the complete architecture

Full architecture in one diagram — scopes: host → API → 3 agents → engine → model → store → output.

```mermaid
flowchart TB
    subgraph HOST["HOST APPLICATION"]
        H1["Long agent session<br/>accumulates message history"]
        H2["Keeps its own FULL copy<br/>always"]
    end

    subgraph API["HTTP SERVICE — api/main.py"]
        A1["POST /v1/compact<br/>GET /health<br/>GET /v1/sessions/{id}/stats"]
        A1n["Builds components ONCE at startup<br/>owns FAIL-OPEN contract"]
    end

    subgraph AGENTS["CONTEXT CRUNCH AGENTS"]
        Scout["1. SCOUT — agents/scout.py<br/>rule-based, no LLM"]
        ScoutD["profile: aggressive / balanced / conservative<br/>goal extraction<br/>pinned side-effecting tools"]
        Guardian["2. GUARDIAN — agents/guardian.py<br/>rule-based, no LLM, paranoid"]
        GuardianD["REGRET: removed-before item needed again<br/>-> log mistake; after 2 force conservative<br/>STUCK: same error 3+ times<br/>-> restore 3 recent removals"]
        Worker["3. WORKER / COMPACTOR — agents/worker.py<br/>the ONLY editor of history"]
        WorkerF["PLAN -> ACT -> VERIFY<br/>PLAN: token count, trigger threshold, eligible items<br/>ACT: parallel batch judge, split-first for big items<br/>VERIFY: count / roles / tool_call_id / empty content<br/>ANY failure -> ORIGINALS"]
    end

    subgraph POL["POLICY ENGINE — core/policy.py"]
        G["Every drop gate is a VETO"]
        G1["verdict==drop"]
        G2["confidence > floor 0.70-0.90"]
        G2B["not essential AND not relevant"]
        G3["at least one junk reason<br/>consumed / superseded / noise"]
    end

    subgraph ML["MODEL LAYER — external dependency"]
        Laya["Laya structured-output LLM<br/>runs the five questions"]
        E1["LayaEngine: batched, concurrency-capped<br/>no fail-open inside"]
    end

    subgraph CORE["CORE FILES"]
        Q["core/questions.py<br/>the five questions, defined once"]
        S["core/state.py<br/>goal / next_step / item / later_findings"]
        SP["core/splitter.py + refiner.py<br/>deterministic split, then rebuild"]
    end

    subgraph OUT["OUTPUT"]
        O1["Compacted history — or<br/>ORIGINALS + fail_open=true"]
        O1n["same count, same roles, same tool_call_id"]
    end

    subgraph PERSIST["PERSISTENCE — side-car store.py"]
        St1["SQLite decisions + mistakes tables"]
        St2["originals as files under HASH(session_id)<br/>never raw id"]
    end

    H1 -->|"full history"| A1
    A1 --> Scout
        Scout --> ScoutD
        Scout --> Guardian
        A1 --> Guardian
        Guardian --> GuardianD
        GuardianD -->|"Overrides(profile, restore)"| Worker
        ScoutD -->|"SessionContext(profile, goal, pinned)"| Worker

        Worker --> WorkerF
        WorkerF --> Q
        WorkerF --> S
        S -->|"4-field state"| E1
        E1 -->|"batched, capped"| Laya
        Laya -->|"verdict + essential + consumed + superseded + relevance"| E1
        E1 -->|"LayaResult per item"| Worker

        Worker -->|"decide()"| G
        G --> G1
        G1 --> G2
        G2 --> G2B
        G2B --> G3
        G3 -->|"all gates pass"| O1
        G3 -.->|"ANY gate fails -> KEEP"| O1

        Worker -->|"too large to send whole"| SP
        SP -->|"pieces, each judged"| E1
        SP -->|"surviving pieces rejoined"| Worker

        Worker -->|"EVERY original, before dropping"| St1
        St1 --> St2
        GuardianD -.->|"reads decisions"| St1
        GuardianD -.->|"logs mistakes"| St1

        WorkerF -->|"VERIFY ok"| O1
        WorkerF -.->|"VERIFY failed -> ORIGINALS"| O1
        O1 -->|"compacted history"| H1
        O1 -.->|"fail_open=true, tokens_before==tokens_after"| A1

        classDef host fill:#eef2f7,stroke:#64748b,color:#0f172a
        classDef api fill:#fef3c7,stroke:#d97706,color:#0f172a
        classDef agent fill:#e0e7ff,stroke:#4f46e5,color:#0f172a
        classDef core fill:#ccfbf1,stroke:#0d9488,color:#0f172a
        classDef model fill:#fce7f3,stroke:#db2777,color:#0f172a
        classDef persist fill:#e2e8f0,stroke:#475569,color:#0f172a
        classDef warn fill:#fee2e2,stroke:#dc2626,color:#0f172a

        class H1,H2 host
        class A1,A1n api
        class Scout,ScoutD,Guardian,GuardianD,Worker,WorkerF,G,G1,G2,G2B,G3 agent
        class Q,S,SP core
        class Laya,E1 model
        class St1,St2,O1,O1n persist
    ```