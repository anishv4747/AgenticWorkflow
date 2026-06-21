# MedDevComplaintSystem — Pipeline

Agentic workflow for medical device complaint triage. Takes a raw complaint narrative and produces an ISO 14971 risk assessment + CAPA recommendation.

**Current status:** Risk Analysis Agent is fully implemented (live LLM). All other agents are mocks returning hardcoded data.

---

## Prerequisites

- Python 3.12+
- An Anthropic API key

---

## Setup

**1. Install dependencies**

From the repo root:

```bash
pip install langgraph anthropic python-dotenv
```

**2. Add your API key**

Edit `.env` in this directory:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the pipeline

From the `Project/MedDevComplaintSystem/` directory:

```bash
# Run with the built-in default complaint
python pipeline.py

# Run with your own complaint text
python pipeline.py --complaint "During a CT scan on a Siemens SOMATOM, the reconstruction software crashed mid-scan. Patient had to be rescanned. SW version 7.2.1."
```

---

## What each agent does

| Agent | Status | What it does |
|---|---|---|
| `extraction_agent` | MOCK | Returns hardcoded Philips MRI extraction fields |
| `retrieval_agent` | MOCK | Returns 3 MAUDE events + 1 Class II recall (Z-2024-00423) |
| `similarity_module` | MOCK | Returns cluster 12 (MRI_SSFP_artifact_cluster, emerging trend) |
| `risk_analysis_agent` | **LIVE** | Calls `claude-sonnet-4-6` — ISO 14971 autonomous tool-calling assessment |
| `report_agent` | MOCK | Prints a stub document with the risk level from state |

**Gates:**
- **Gate 1** (after extraction): halts if `confidence < 0.5` or `failure_mode` is missing
- **Gate 3** (after risk analysis): halts if risk is `UNACCEPTABLE` with zero evidence citations

---

## What to expect in the output

The terminal prints each agent's inputs and outputs as it runs:

```
────────────────────────────────────────────────────────────────
  EXTRACTION AGENT  [MOCK]
────────────────────────────────────────────────────────────────
  ← reading from state:  complaint_text
  → writing to state:    modality, failure_mode, severity_indicator, ...

  GATE 1  (post-extraction)
  ✓ PASS → proceeding to retrieve + cluster

  RETRIEVAL AGENT  [MOCK]
  → writing to state:    3 events, 1 recall

  SIMILARITY MODULE  [MOCK]
  → writing to state:    cluster_id, trend_flag, growth_rate_30d

────────────────────────────────────────────────────────────────
  RISK ANALYSIS AGENT  [autonomous tool-calling | claude-sonnet-4-6]
────────────────────────────────────────────────────────────────
  [LOOP] starting tool-calling loop (cap 6 iters / 30s)...
  [TOOL] turn 1: score_severity(...) -> ok
  [TOOL] turn 1: compute_probability(...) -> ok
  [TOOL] turn 1: load_past_reports(...) -> ok
  [TOOL] turn 2: apply_risk_matrix(...) -> ok
  [TOOL] turn 3: lookup_capa_requirements(...) -> ok
  [TOOL] turn 4: submit_risk_narrative(...) -> ok
  [RESULT] severity_level=S3 probability_level=P3 risk_level=ALARP

  → writing to state:    risk_level, severity, probability, CAPA, escalation flags

  GATE 3  (post-risk-analysis)
  ✓ PASS → proceeding to report

  REPORT AGENT  [MOCK]
  ← reading from state:  risk_level, escalation flags, CAPA ...
  → writing to state:    document_id SR-2026-XXXX
```

Followed by a final summary block with all risk assessment fields.

---

## File structure

```
MedDevComplaintSystem/
├── pipeline.py               # LangGraph graph + mock agents — entry point
├── agents/
│   ├── risk_analysis.py      # Live Risk Analysis Agent (ISO 14971, autonomous tool-calling loop)
│   └── risk_tools.py         # Deterministic tools (severity/probability/risk_matrix/CAPA/escalation)
├── data/
│   └── signal_reports.db     # SQLite episodic memory (auto-created on first run)
├── specs/                    # Agent design specs (agent-01 through agent-06)
├── PLAN.md                   # Architecture + ComplaintState TypedDict + graph wiring
├── PLAN_v2.md                # Risk Analysis Agent v2: autonomous tool-calling design
└── .env                      # ANTHROPIC_API_KEY (never committed)
```

---

## Episodic memory

`signal_reports.db` is created automatically on first run. It stores completed risk assessments so the Risk Analysis Agent can reference prior similar cases in future runs. On the first run, it will be empty.
