# Medical Device Complaint Handling — Agentic Workflow
## Overall Architecture Plan

> **Positioning:** Decision support, not decision automation. A Quality Manager (QM) reviews and approves every output before it enters the QMS.
> **Regulatory framing:** ISO 14971:2019 (risk), ISO 13485:2016 §8.2.2 (complaint handling), IEC 62304 §9 (software problem resolution).

---

## 1. System Overview

Medical device companies receive complaints daily — narratives describing device failures, patient harm events, software malfunctions, and diagnostic errors. Manually triaging each complaint against FDA evidence, assessing risk per ISO 14971, drafting CAPA recommendations, and formatting a QRB-ready report takes a QM **4–8 hours per complaint**.

This system reduces that to **under 30 minutes of QM review time** by automating evidence gathering, risk assessment, and report drafting through a 6-agent pipeline. The QM receives a structured draft with traceable citations, reviews it, and approves or revises before it enters the QMS.

**What the system does NOT do:**
- Autonomously submit reports to regulators
- Make final risk acceptability decisions (ISO 14971 §7.2/§7.3/§8 — these are management decisions)
- Replace the PRRC or Quality Officer in regulatory sign-off

---

## 2. Architecture

```
Raw Complaint (text)
        ↓
[extraction_agent]  ──→  Gate 1: confidence < 0.5 → END (escalate)
        ↓  writes: state.extraction
        ├──────────────────────────────────────────┐
[retrieval_agent]                    [similarity_module]   ← parallel
        ↓  writes: state.retrieval        ↓  writes: state.similarity
        └──────────────── join ───────────┘
        ↓  (both in state now)
[risk_analysis_agent]  ──→  Gate 3: UNACCEPTABLE + 0 citations → END (escalate)
        ↓  writes: state.risk_assessment, state.capa, state.escalation_flags
[report_agent]  (self-critique loop, max 2 rounds / 20s)
        ↓  writes: state.report
QM Review → Approve / Revise → QMS Record

[complaint_simulation_agent]  ← offline only, populates test bench
```

**Key principle from `option_a_memory.py`:** Agents do not call each other. Each agent is a plain function that reads from `ComplaintState` and returns a dict of new fields. LangGraph writes those fields back into shared state and routes to the next node. The Risk Analysis Agent is the clearest example — it reads `state.extraction`, `state.retrieval`, and `state.similarity` that previous agents already wrote, then writes `state.risk_assessment` and `state.capa` that the Report Agent will read. No direct coupling between agents.

---

## 3. Code Structure

### 3.1 Shared State (the single source of truth)

Mirrors `BaristaStateWithMemory` from `option_a_memory.py`. Every agent reads from and writes to this TypedDict. No agent receives another agent's output as a direct argument — everything flows through state.

```python
# pipeline/state.py
from typing import TypedDict, Optional

class ComplaintState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    complaint_text: str          # raw narrative from QM / intake form
    trace_id: str                # UUID — propagated to all agents for observability

    # ── Agent 3: Extraction writes these ───────────────────────────
    modality: Optional[str]
    manufacturer: Optional[str]
    device_model: Optional[str]
    software_version: Optional[str]
    component: Optional[str]
    failure_mode: Optional[str]
    severity_indicator: Optional[str]   # S1–S5
    qms_complaint_category: Optional[str]  # SW-FUNC, IMG-QUAL, etc.
    software_related: Optional[bool]
    is_safety_related: Optional[bool]
    confidence: Optional[float]
    extraction_cot: Optional[str]       # chain-of-thought reasoning

    # ── Similarity Module writes these (non-LLM, parallel) ─────────
    cluster_id: Optional[int]
    cluster_label: Optional[str]
    cluster_size: Optional[int]
    trend_flag: Optional[str]           # emerging | stable | declining
    growth_rate_30d: Optional[float]
    similar_event_ids: Optional[list]

    # ── Agent 4: Retrieval writes these ────────────────────────────
    matching_events: Optional[list]     # top-5 MAUDE events with relevance scores
    matching_recalls: Optional[list]    # top-5 recalls with root cause + action
    regulatory_context: Optional[str]
    low_confidence_retrieval: Optional[bool]

    # ── Agent 2: Risk Analysis writes these ← THE CORE AGENT ───────
    hazardous_situation: Optional[str]
    harm: Optional[str]
    severity_level: Optional[str]       # S1–S5
    severity_rationale: Optional[str]
    probability_level: Optional[str]    # P1–P5
    probability_rationale: Optional[str]
    risk_level: Optional[str]           # ACCEPTABLE | ALARP | UNACCEPTABLE
    evidence_basis: Optional[list]      # [{source, id, relevance}]
    uncertainty: Optional[str]
    capa_immediate: Optional[str]
    capa_investigation: Optional[str]
    capa_corrective: Optional[str]
    capa_preventive: Optional[str]
    capa_verification: Optional[str]
    capa_effectiveness: Optional[str]
    capa_precedent: Optional[str]       # real recall ID cited
    escalation_required: Optional[bool]
    prrc_notification_required: Optional[bool]
    fsca_required: Optional[bool]

    # ── Agent 6: Report Generation writes these ─────────────────────
    document_id: Optional[str]          # SR-YYYY-NNNN
    report_markdown: Optional[str]
    citation_count: Optional[int]
    unsupported_claims: Optional[int]
    self_score: Optional[float]
    review_needed_flag: Optional[bool]
    approval_status: Optional[str]      # DRAFT | PENDING_QM_REVIEW

    # ── Pipeline control (written by gates, read by conditional edges) ──
    gate1_passed: Optional[bool]
    gate3_passed: Optional[bool]
    escalation_notice: Optional[str]

    # ── Conversation history (for multi-turn LLM agents) ────────────
    messages: list
```

---

### 3.2 External Data Stores as Tools

Mirrors the `MEMORY_TOOLS` pattern from `option_a_memory.py`. Agents access data stores through tool definitions, not direct imports. Swap the backing store without touching agent code.

```python
# pipeline/tools.py

# ── ChromaDB (vector store for recall + event embeddings) ──────────────────
def query_chromadb(query_text: str, collection: str, n_results: int = 10) -> list:
    """Semantic search over embedded FDA recalls and events."""
    ...

# ── openFDA API ─────────────────────────────────────────────────────────────
def query_openfda_events(product_code: str, keyword: str, limit: int = 10) -> list:
    """Live query to openFDA device adverse events endpoint."""
    ...

def query_openfda_recalls(product_code: str, root_cause: str = None) -> list:
    """Live query to openFDA device recall endpoint."""
    ...

# ── NetworkX knowledge graph ─────────────────────────────────────────────────
def traverse_device_graph(manufacturer: str, product_code: str, hops: int = 2) -> list:
    """Graph traversal: manufacturer → product code → linked recalls."""
    ...

# ── SQLite (episodic memory: past signal reports) ────────────────────────────
def load_past_reports(failure_mode: str, modality: str, limit: int = 5) -> list:
    """Load similar past reports for Risk Analysis Agent context."""
    ...

def save_signal_report(document_id: str, trace_id: str, report_data: dict):
    """Persist completed report to signal_reports table."""
    ...

# Tool definitions for LLM tool-calling (mirrors MEMORY_TOOLS structure)
RETRIEVAL_TOOLS = [
    {
        "name": "query_chromadb",
        "description": "Search embedded FDA recalls and events by semantic similarity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "collection": {"type": "string", "enum": ["events", "recalls", "combined"]},
                "n_results": {"type": "integer", "default": 10},
            },
            "required": ["query_text", "collection"],
        },
    },
    {
        "name": "query_openfda_recalls",
        "description": "Fetch real-time recall records from openFDA by product code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_code": {"type": "string"},
                "root_cause": {"type": "string"},
            },
            "required": ["product_code"],
        },
    },
    {
        "name": "load_past_reports",
        "description": "Load similar past signal reports from episodic memory (SQLite).",
        "input_schema": {
            "type": "object",
            "properties": {
                "failure_mode": {"type": "string"},
                "modality": {"type": "string"},
            },
            "required": ["failure_mode"],
        },
    },
]
```

---

### 3.3 Agent Functions (each reads state, returns a dict of updates)

```python
# agents/extraction.py
def extraction_agent(state: ComplaintState) -> dict:
    """
    Reads:  state.complaint_text
    Writes: state.modality, .failure_mode, .severity_indicator,
            .qms_complaint_category, .confidence, .is_safety_related, ...

    Wraps raw text in <user_narrative> tags (prompt-injection defense).
    Runs 1 self-reflection pass to fix enum mismatches.
    Gate 1 result written to state.gate1_passed — LangGraph reads it for routing.
    """
    ...
    return {
        "modality": ...,
        "failure_mode": ...,
        "severity_indicator": ...,
        "qms_complaint_category": ...,
        "confidence": ...,
        "is_safety_related": ...,
        "gate1_passed": confidence >= 0.5 and failure_mode is not None,
    }


# agents/retrieval.py
def retrieval_agent(state: ComplaintState) -> dict:
    """
    Reads:  state.failure_mode, state.modality, state.manufacturer
    Writes: state.matching_events, state.matching_recalls,
            state.regulatory_context, state.low_confidence_retrieval

    Queries ChromaDB + openFDA + knowledge graph. Top-5 per category.
    Drops items below relevance 0.3. Never passes raw payloads downstream.
    """
    ...
    return {
        "matching_events": ...,
        "matching_recalls": ...,
        "regulatory_context": ...,
        "low_confidence_retrieval": all_scores_below_threshold,
    }


# agents/similarity.py  (non-LLM — deterministic ML pipeline)
def similarity_module(state: ComplaintState) -> dict:
    """
    Reads:  state.complaint_text, state.modality
    Writes: state.cluster_id, state.cluster_label, state.trend_flag,
            state.growth_rate_30d, state.similar_event_ids

    sentence-transformers embedding → HDBSCAN cluster assignment →
    UMAP projection → z-score anomaly on cluster growth rate.
    No LLM calls. Runs in parallel with retrieval_agent.
    """
    ...
    return {
        "cluster_id": ...,
        "cluster_label": ...,
        "trend_flag": ...,
        "growth_rate_30d": ...,
        "similar_event_ids": ...,
    }


# agents/risk_analysis.py  ← THE CORE AGENT
def risk_analysis_agent(state: ComplaintState) -> dict:
    """
    Reads:  state.failure_mode, state.severity_indicator,      ← from extraction_agent
            state.matching_events, state.matching_recalls,      ← from retrieval_agent
            state.cluster_id, state.trend_flag, state.growth_rate_30d  ← from similarity_module

    Writes: state.risk_level, state.hazardous_situation, state.harm,
            state.severity_level, state.probability_level,
            state.evidence_basis, state.uncertainty,
            state.capa_*, state.escalation_required,
            state.prrc_notification_required, state.fsca_required,
            state.gate3_passed

    ISO 14971 §7.3–7.6:
      1. Load past signal reports from SQLite for similar cases (tool call)
      2. Synthesize retrieval evidence — cite real MAUDE/recall IDs
      3. Apply 5×5 S×P matrix → risk_level (ACCEPTABLE | ALARP | UNACCEPTABLE)
      4. Constitutional guardrail: refuse ALARP/UNACCEPTABLE with 0 citations
      5. Generate structured CAPA from recall precedents
      6. Compute escalation flags deterministically (not by LLM)
      7. 1 self-reflection round to verify citation coverage

    Communication back to pipeline: writes results to state fields.
    Report Agent reads these fields from state — no direct call between agents.
    """
    past_reports = load_past_reports(state["failure_mode"], state["modality"])

    # Build context for LLM from state fields already populated upstream
    evidence_context = {
        "events":   state.get("matching_events", []),
        "recalls":  state.get("matching_recalls", []),
        "cluster":  {"trend": state.get("trend_flag"), "growth": state.get("growth_rate_30d")},
        "history":  past_reports,
    }

    # ... LLM call with ISO 14971 CoT prompt ...

    # Escalation flags computed in Python — not by LLM
    escalation_required = risk_level in ("ALARP", "UNACCEPTABLE")
    prrc_notification   = risk_level == "UNACCEPTABLE"
    gate3_passed        = not (risk_level == "UNACCEPTABLE" and len(evidence_basis) == 0)

    return {
        "risk_level":               risk_level,
        "hazardous_situation":      ...,
        "harm":                     ...,
        "severity_level":           ...,
        "probability_level":        ...,
        "evidence_basis":           evidence_basis,
        "uncertainty":              ...,
        "capa_immediate":           ...,
        "capa_corrective":          ...,
        "capa_preventive":          ...,
        "capa_verification":        ...,
        "capa_precedent":           ...,   # real recall ID from state.matching_recalls
        "escalation_required":      escalation_required,
        "prrc_notification_required": prrc_notification,
        "fsca_required":            False,  # requires confirmed root cause — human decision
        "gate3_passed":             gate3_passed,
    }


# agents/report.py
def report_agent(state: ComplaintState) -> dict:
    """
    Reads:  ALL state fields written by upstream agents
    Writes: state.document_id, state.report_markdown, state.citation_count,
            state.unsupported_claims, state.self_score, state.review_needed_flag

    Fills ISO 13485 §4.2.4 controlled document template.
    Self-critique loop (max 2 rounds / 20s):
      rubric checks citation coverage, schema completeness,
      uncertainty disclosure, risk/CAPA consistency.
    Persists to SQLite signal_reports (episodic memory).
    """
    ...
    save_signal_report(document_id, state["trace_id"], report_data)
    return {
        "document_id":         document_id,
        "report_markdown":     ...,
        "citation_count":      ...,
        "unsupported_claims":  ...,
        "self_score":          ...,
        "review_needed_flag":  ...,
        "approval_status":     "DRAFT",
    }
```

---

### 3.4 Graph Construction

```python
# pipeline/graph.py
from langgraph.graph import StateGraph, END

def gate1_router(state: ComplaintState) -> str:
    return "proceed" if state.get("gate1_passed") else "halt"

def gate3_router(state: ComplaintState) -> str:
    return "proceed" if state.get("gate3_passed") else "halt"

def build_complaint_graph():
    graph = StateGraph(ComplaintState)

    # Register nodes (each is an agent function)
    graph.add_node("extract",       extraction_agent)
    graph.add_node("retrieve",      retrieval_agent)
    graph.add_node("cluster",       similarity_module)   # non-LLM
    graph.add_node("risk_analysis", risk_analysis_agent) # core agent
    graph.add_node("report",        report_agent)

    # Entry point
    graph.set_entry_point("extract")

    # Gate 1: halt if extraction confidence too low
    graph.add_conditional_edges("extract", gate1_router, {
        "proceed": "retrieve",
        "halt":    END,
    })

    # Parallel fan-out: retrieve + cluster run simultaneously after extraction
    graph.add_edge("extract",  "cluster")   # LangGraph runs both after Gate 1 pass

    # Join: risk_analysis waits for both retrieve and cluster to complete
    graph.add_edge("retrieve",      "risk_analysis")
    graph.add_edge("cluster",       "risk_analysis")

    # Gate 3: halt if UNACCEPTABLE risk with no evidence citations
    graph.add_conditional_edges("risk_analysis", gate3_router, {
        "proceed": "report",
        "halt":    END,
    })

    graph.add_edge("report", END)
    return graph.compile()


# Entry point
if __name__ == "__main__":
    import uuid
    app = build_complaint_graph()

    result = app.invoke({
        "complaint_text": "During cardiac MRI on Philips Achieva 1.5T, banding "
                          "artifacts appeared in SSFP sequence. Images non-diagnostic. "
                          "Repeat scan required. SW version 5.7.1.",
        "trace_id": str(uuid.uuid4()),
        "messages": [],
        # All other fields start as None — each agent populates its own
    })

    print(f"\nRisk Level:  {result['risk_level']}")
    print(f"Escalation:  {result['escalation_required']}")
    print(f"Document ID: {result['document_id']}")
    print(f"Review Flag: {result['review_needed_flag']}")
```

---

## 4. Agent Specifications

### Agent 1: Orchestrator

| Field | Value |
|---|---|
| Autonomy | L2 Workflow — fixed assembly-line, no LLM routing |
| Tech | LangGraph `StateGraph` |
| Spec | [specs/agent-01-orchestrator.md](specs/agent-01-orchestrator.md) |

**Input:** raw complaint text + `trace_id`
**Output:** `ReportOutput` or partial report with `REVIEW NEEDED` flag

**Responsibilities:**
- Dispatch complaint to Agent 3 (Extraction)
- After Gate 1 pass: fan out Agent 4 and Similarity Module in parallel
- Join parallel results; dispatch combined context to Agent 2
- After Gate 3 pass: dispatch to Agent 6 (Report)
- Enforce loop caps and the 120s total pipeline timeout
- Propagate `trace_id` to all agents for observability and audit trail

**Validation Gates:**

| Gate | Trigger | Action |
|---|---|---|
| Gate 1 (post-extraction) | `confidence < 0.5` OR missing `failure_mode` / `severity_indicator` | Flag for human; do not proceed |
| Gate 2 (post-retrieval) | 0 results returned | Inject "no FDA evidence found" warning into Agent 2 context |
| Gate 2 (post-retrieval) | All relevance scores < 0.3 | Mark retrieval as low-confidence; drop items below 0.3 |
| Gate 3 (post-risk) | `UNACCEPTABLE` risk + 0 evidence citations | Reject; escalate for human review |
| Gate 3 (post-risk) | `ACCEPTABLE` risk + Death events in retrieved evidence | Override to ALARP; escalate |

---

### Agent 2: Risk Analysis Agent

| Field | Value |
|---|---|
| Autonomy | L1 Augmented LLM — single-pass CoT + 1 self-reflection round |
| Spec | [specs/agent-02-risk-analysis.md](specs/agent-02-risk-analysis.md) |

**Input:** `ExtractionOutput` + `RetrievalOutput` + cluster context (from Similarity Module)
**Output:** `RiskCapaOutput`

**Responsibilities:**
1. Query internal knowledge base / past CAPA records for similar historical cases
2. Synthesize evidence from retrieval results; cite real FDA report numbers / recall IDs
3. Apply ISO 14971 5×5 risk matrix: Severity (S1–S5) × Probability (P1–P5) → risk level
4. Identify hazardous situation using ISO 14971 Annex C hazard categories
5. Generate structured CAPA: immediate containment → root cause investigation → corrective action → preventive action → verification method → effectiveness criteria
6. Set escalation flags based on risk level and evidence state
7. Return `RiskCapaOutput` to Orchestrator

**ISO 14971 Risk Matrix:**

|  | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| **S5** | ALARP | UNACCEPTABLE | UNACCEPTABLE | UNACCEPTABLE | UNACCEPTABLE |
| **S4** | ACCEPTABLE | ALARP | UNACCEPTABLE | UNACCEPTABLE | UNACCEPTABLE |
| **S3** | ACCEPTABLE | ACCEPTABLE | ALARP | UNACCEPTABLE | UNACCEPTABLE |
| **S2** | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE | ALARP | ALARP |
| **S1** | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE | ALARP |

**Escalation logic (deterministic rules):**
- `escalation_required = true` iff `risk_level ∈ {ALARP, UNACCEPTABLE}`
- `prrc_notification_required = true` iff `risk_level == UNACCEPTABLE`
- `fsca_required = true` iff `(confirmed_root_cause AND UNACCEPTABLE AND active_distribution)`

**Constitutional guardrail:** Refuse to output `ALARP` or `UNACCEPTABLE` without ≥1 evidence citation in `evidence_basis[]`.

**Targets:** Expert rubric > 3.0/5; uncited claims < 15%

---

### Agent 3: Extraction/Taxonomy Agent

| Field | Value |
|---|---|
| Autonomy | L1 Augmented LLM — CoT extraction + 1 self-reflection round |
| Spec | [specs/agent-03-extraction-taxonomy.md](specs/agent-03-extraction-taxonomy.md) |

**Input:** raw complaint text (string)
**Output:** `ExtractionOutput`

**Responsibilities:**
1. Normalize text: fix encoding, strip non-relevant metadata
2. CoT extraction: device → manufacturer → failure mode → component → symptom → patient impact → discovery phase → software version → affected countries (ISO 3166-1)
3. Classify into ISO 13485 §8.2.2 taxonomy (13 categories — see spec)
4. Assign severity indicator (S1–S5) from patient impact language
5. Score confidence (0.0–1.0); flag `is_safety_related`, `usability_concern`, `security_concern`
6. Wrap raw narrative in `<user_narrative>` delimiters (prompt-injection defense)
7. One self-reflection pass to fix enum mismatches and missing required fields

**ISO 13485 §8.2.2 Complaint Categories:**

| Code | Meaning |
|---|---|
| `SW-FUNC` | Software functional failure (crash, freeze, unexpected reboot) |
| `SW-ALGO` | Algorithm / calculation error (wrong values, reconstruction error) |
| `SW-UI` | User interface issue (display error, incorrect labeling) |
| `SW-DATA` | Data integrity / loss (corruption, DICOM failure, transfer error) |
| `SW-CYBER` | Cybersecurity concern |
| `IMG-QUAL` | Image quality degradation (artifact, noise, poor resolution) |
| `IMG-PROC` | Image processing error (registration failure) |
| `PERF-ACC` | Performance / accuracy issue (false positive, false negative, drift) |
| `SAFE-PAT` | Patient safety concern (injury, adverse event, harm) |
| `SAFE-USR` | User / operator safety (radiation exposure, electrical shock) |
| `HW-MECH` | Hardware mechanical failure |
| `HW-ELEC` | Hardware electrical failure |
| `DOC-LABEL` | Labeling / documentation issue (IFU error, missing warning) |

**Target:** Field-level F1 > 0.80 vs gold benchmark

---

### Agent 4: Retrieval Agent

| Field | Value |
|---|---|
| Autonomy | L1 single-pass RAG → upgrade to L2 ReAct only if Precision@5 < 0.65 on gold set |
| Spec | [specs/agent-04-retrieval.md](specs/agent-04-retrieval.md) |

**Input:** `ExtractionOutput` (device, manufacturer, modality, failure_mode)
**Output:** `RetrievalOutput`

**Responsibilities:**
1. Embed complaint with `all-MiniLM-L6-v2`; query ChromaDB collections (past events, recalls, combined)
2. Query openFDA API: adverse events (MAUDE) + recalls, filtered by product code and modality
3. Graph traversal: device → manufacturer → product code → linked events → linked recalls (NetworkX)
4. Hybrid re-ranking: combine vector similarity + graph proximity score
5. Discard results with relevance < 0.3
6. Summarize top-5 results before handoff (token budget discipline — never pass raw payloads)
7. If upgraded to ReAct: Thought → Action(query) → Observation loop; cap 5 iters / 30s; fallback = best-so-far results

**Resilience:** openFDA HTTP 429 / timeout → exponential backoff (3 retries) → degrade to ChromaDB local-only, never hard-fail.

**Target:** Precision@5 > 0.65 vs gold relevance judgments

---

### Agent 5: Complaint Simulation (Synthetic Data Generator)

| Field | Value |
|---|---|
| Autonomy | L1 Augmented LLM — single-pass generation |
| Role | Offline — test bench only, not in live pipeline |
| Spec | [specs/agent-05-complaint-simulation.md](specs/agent-05-complaint-simulation.md) |

**Input:** generation parameters — `modality`, `failure_type`, `severity_level` (S1–S5), `manufacturer`, `product_code`, optional `seed_narrative`
**Output:** `SimulatedComplaint`

**Responsibilities:**
1. Sample stylistic patterns from real MAUDE narrative templates (avg 836 chars, clinical language, device terminology)
2. Generate realistic complaint narrative with: device name + model, software version, symptom, patient/operator impact, discovery context
3. Vary difficulty: `clean` (unambiguous) | `ambiguous` (multiple possible categories) | `incomplete` (missing fields) | `edge_case` (multi-device, injection attempt)
4. Attach ground-truth metadata: expected modality, expected `qms_complaint_category`, expected severity, expected risk_level — these become gold-set labels
5. Quality gate: reject and regenerate if narrative < 100 chars or lacks a failure mode description

**Use cases:**
- Pipeline regression testing without real patient data
- DPO preference pair generation (paired good/bad report drafts from same complaint)
- Ablation benchmark construction (50 labeled examples for evaluation)
- Edge-case stress testing (injection attempts, multi-device, ambiguous severity)

---

### Agent 6: Report Generation Agent

| Field | Value |
|---|---|
| Autonomy | L2 Evaluator-Optimizer — self-critique loop, max 2 rounds / 20s |
| Spec | [specs/agent-06-report-generation.md](specs/agent-06-report-generation.md) |

**Input:** `ExtractionOutput` + `SimilarityOutput` + `RetrievalOutput` + `RiskCapaOutput`
**Output:** `ReportOutput` (ISO 13485 §4.2.4 controlled document)

**Responsibilities:**
1. Fill controlled document template with upstream JSON data
2. Generate 6 report sections: extracted fields | pattern match + cluster trend | FDA evidence citations | risk assessment | CAPA recommendations | report quality metadata
3. Document header: `SR-YYYY-NNNN` ID, revision 1.0, `generated_by: Signal Intelligence System v1.0`, `approval_status: DRAFT`, `Reviewed by: [blank — human only]`
4. Self-critique against 4-point rubric (max 2 rounds):
   - Citation coverage: every factual claim cites a specific FDA record ID
   - Schema completeness: all required fields populated
   - Uncertainty disclosure: non-confirmable facts flagged
   - Risk/CAPA consistency: risk level and CAPA severity align
5. If rubric fails after 2 rounds → accept with `REVIEW NEEDED` flag
6. AI transparency footer (mandatory): *"AI-assisted draft — requires human review and approval per AIMS policy"*
7. Persist report + quality block to `signal_reports` SQLite table (episodic memory for Similarity Module)

**Targets:** Hallucination rate < 15%; LLM-Judge / human rubric kappa > 0.60

---

## 4. Shared JSON Schema Contracts

Freeze in `schemas.py` before any agent is built. This is the single most important integration risk control.

```python
# Key schemas (Pydantic v2)

class ExtractionOutput(BaseModel):      # Agent 3 → Agent 4, Agent 2
class RetrievalOutput(BaseModel):       # Agent 4 → Agent 2
class SimilarityOutput(BaseModel):      # Similarity Module → Agent 2, Agent 6
class RiskCapaOutput(BaseModel):        # Agent 2 → Agent 6
class ReportOutput(BaseModel):          # Agent 6 → Orchestrator → QM
class SimulatedComplaint(BaseModel):    # Agent 5 → test bench
```

**Enforced enums (must not use alternatives):**
- `risk_level`: `ACCEPTABLE | ALARP | UNACCEPTABLE` — never `HIGH / MEDIUM / LOW`
- `severity_indicator`: `S1_negligible | S2_minor | S3_serious | S4_critical | S5_catastrophic`
- `probability`: `P1_incredible | P2_improbable | P3_remote | P4_occasional | P5_frequent`
- `qms_complaint_category`: 13-code enum per ISO 13485 §8.2.2 (see Agent 3 table above)
- `approval_status`: `DRAFT | PENDING_QM_REVIEW | APPROVED | REJECTED`

---

## 5. Regulatory Alignment

| Standard | Clause | Implementation |
|---|---|---|
| ISO 14971:2019 | §7.3 Hazard identification | Agent 2 — Annex C hazard category lookup |
| ISO 14971:2019 | §7.4 Risk estimation | Agent 2 — S×P matrix with evidence-calibrated probability |
| ISO 14971:2019 | §7.5 Risk evaluation | Agent 2 — 5×5 acceptability matrix |
| ISO 14971:2019 | §7.6 Risk control | Agent 2 CAPA — immediate containment + corrective + preventive |
| ISO 14971:2019 | §10 Post-production info | Full pipeline — PMS surveillance |
| ISO 14971:2019 | Annex C | Agent 2 — hazardous situation categories |
| ISO 14971:2019 | Annex D | Agent 2 — severity/probability scale definitions |
| ISO 13485:2016 | §8.2.2 | Agent 3 — 13-code complaint taxonomy |
| ISO 13485:2016 | §8.5.2 | Agent 2 CAPA — corrective action |
| ISO 13485:2016 | §8.5.3 | Agent 2 CAPA — preventive action |
| ISO 13485:2016 | §4.2.4 | Agent 6 — controlled document fields (ID, revision, author, reviewer) |
| ISO 13485:2016 | §8.2.1 | Similarity Module — post-market surveillance trending |
| IEC 62304 | §9.1 | Agent 3 — prepare structured problem report |
| IEC 62304 | §9.2 | Agent 4 + Agent 2 — investigate, determine cause |
| IEC 62304 | §9.5 | Observability layer — maintain records (LangSmith traces) |
| IEC 62304 | §9.6 | Similarity Module — trend analysis |
| IEC 62304 | §9.7 | Agent 2 CAPA — verify resolution effectiveness |

---

## 6. Implementation Order

Build in this sequence. Each step is testable before the next begins.

| Step | What to Build | Why This Order |
|---|---|---|
| 0 | `schemas.py` — all Pydantic contracts + mock factories | Unblocks all agents; every agent can run standalone with `mock_extraction()` etc. |
| 1 | Agent 5 (Simulation) | Generates the 50-example gold benchmark before any live agent is built |
| 2 | Agent 3 (Extraction) | First live component; also builds the single-LLM-call baseline (ablation #5) |
| 3 | Agent 4 (Retrieval) | ChromaDB index built from downloaded data; openFDA API integration |
| 4 | Agent 2 (Risk Analysis) | Depends on both Agent 3 and Agent 4 outputs |
| 5 | Agent 6 (Report Generation) | Depends on all upstream outputs |
| 6 | Agent 1 (Orchestrator) | Wire all agents with LangGraph, validation gates, loop caps |

---

## 7. Evaluation Targets

| Metric | Target | Measured By |
|---|---|---|
| Extraction field-level F1 | > 0.80 | vs gold benchmark (Agent 5 output) |
| Retrieval Precision@5 | > 0.65 | vs manual relevance judgments |
| Risk rubric score | > 3.0 / 5.0 | LLM-as-Judge on 20 gold examples |
| Hallucination rate | < 15% uncited claims | Count per report |
| DPO win-rate | > 60% vs baseline | Preference eval on held-out pairs |
| LLM-Judge / human kappa | > 0.60 | Cohen's kappa on shared subset |
| End-to-end latency | < 120s | Wall-clock timing |
| Cost per report | < $0.50 | Token count × GPT-4.1 pricing |

---

## 8. Ablation Studies

| ID | What Is Removed | Metric Affected |
|---|---|---|
| A1 | Agent 4 retrieval (no RAG) | Retrieval Precision@5; grounding score |
| A2 | Agent 6 self-critique pass | Hallucination rate; citation coverage |
| A3 | DPO adapter on Agent 6 | Report win-rate vs baseline |
| A4 | Similarity Module temporal scoring | Trend detection; cluster growth flags |
| A5 | Everything → single LLM call baseline | Overall quality; cost; latency |

A5 (single LLM call) is built first (Step 2 above) and serves as the control condition for all other ablations.
