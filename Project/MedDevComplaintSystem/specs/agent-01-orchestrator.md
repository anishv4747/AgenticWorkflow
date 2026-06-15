# Agent 1: Orchestrator

| Field | Value |
|---|---|
| Epic | Pipeline Integration |
| Owner | M2 |
| Autonomy Level | L2 Workflow — fixed assembly-line, no LLM routing |
| Priority | P0 |
| Estimate | 3 days |
| Status | Not started |
| Depends On | schemas.py, all 5 other agents |

## User Story

As a **Quality Manager**, I want complaints to flow automatically through the pipeline with validation at each step, so that I receive a structured draft report without having to manually coordinate between analysis tools.

## Context

The Orchestrator is the control layer of the entire pipeline. It is NOT an LLM agent — it is a deterministic workflow manager (LangGraph `StateGraph`) that routes data between agents, enforces validation gates, manages timeouts, and ensures that a failing intermediate step halts the pipeline rather than silently corrupting downstream outputs. See `PLAN.md §2` for the full architecture diagram.

## Scope

- **In:** Receiving raw complaint, dispatching to agents in order, enforcing gates, propagating `trace_id`, managing loop caps and fallbacks, returning final report or escalation notice.
- **Out:** Any LLM-based routing decisions, business logic (lives in each agent), report formatting (Agent 6), risk methodology (Agent 2).

## Input / Output

- **Input:** `complaint_text: str`, `trace_id: str` (UUID generated at pipeline entry)
- **Output:** `ReportOutput` (complete) OR `EscalationNotice` (if any gate blocks progression)

## Pipeline Topology

```
complaint_text + trace_id
        ↓
  [Agent 3: Extraction]
        ↓
  Gate 1 check
        ↓ (pass)
  [Agent 4: Retrieval] ‖ [Similarity Module]   ← parallel fan-out
        ↓ (join after both complete)
  Gate 2 check + evidence synthesis
        ↓ (pass)
  [Agent 2: Risk Analysis]
        ↓
  Gate 3 check
        ↓ (pass)
  [Agent 6: Report Generation]
        ↓
  Return ReportOutput to caller (QM interface)
```

## Validation Gates

| Gate | Location | Trigger Condition | Action |
|---|---|---|---|
| Gate 1 | Post-extraction | `confidence < 0.5` | Flag for human review; return `EscalationNotice`; do not proceed |
| Gate 1 | Post-extraction | Missing `failure_mode` or `severity_indicator` | Retry extraction once; if still missing, escalate |
| Gate 1 | Post-extraction | `modality` not in known set | Reject and log |
| Gate 2 | Post-retrieval | 0 results returned from Agent 4 | Inject `"No FDA evidence found"` warning into Agent 2 context; continue |
| Gate 2 | Post-retrieval | All relevance scores < 0.3 | Mark as `low_confidence_retrieval`; discard all; continue with warning |
| Gate 3 | Post-risk | `risk_level == UNACCEPTABLE` AND `evidence_citations == []` | Reject; return `EscalationNotice` |
| Gate 3 | Post-risk | `risk_level == ACCEPTABLE` AND Death events in retrieved evidence | Override to ALARP; escalate to human |
| Gate 3 | Post-risk | CAPA references recall ID not in local DB | Strip reference; flag in report |

## Loop Caps and Timeouts

| Component | Cap | Timeout | Fallback |
|---|---|---|---|
| Agent 3 self-reflection | 1 round | — | Accept extraction as-is; flag uncertainty |
| Agent 4 ReAct (if upgraded) | 5 iterations | 30s | Return best-so-far retrieval results |
| Agent 6 self-critique | 2 rounds | 20s | Accept report with `REVIEW NEEDED` flag |
| Full pipeline | — | 120s total | Return partial report with available outputs |

## Acceptance Criteria

- [ ] Given a valid complaint, when pipeline runs, then all agents execute in the correct order with parallel fan-out at Agent 4 / Similarity Module.
- [ ] Given `confidence < 0.5` from Agent 3, when Gate 1 is evaluated, then an `EscalationNotice` is returned and no further agents are called.
- [ ] Given 0 retrieval results from Agent 4, when Gate 2 is evaluated, then a "no FDA evidence" warning is injected into Agent 2's context and the pipeline continues.
- [ ] Given `UNACCEPTABLE` risk with 0 citations from Agent 2, when Gate 3 is evaluated, then the pipeline halts and an `EscalationNotice` is returned.
- [ ] Given a slow Agent 4 exceeding 30s on ReAct, when timeout fires, then the best-so-far retrieval results are used and the pipeline continues.
- [ ] Given the full pipeline exceeding 120s, when the orchestrator timeout fires, then a partial report is returned with available outputs.
- [ ] Given any handoff payload, when schema validation runs, then a schema mismatch raises a loud error (not a silent pass-through).
- [ ] Given a `trace_id`, when any agent logs, then that same `trace_id` appears in all log entries for that run.

## Technical Approach

1. Implement as LangGraph `StateGraph` with typed state: `PipelineState(complaint_text, trace_id, extraction, retrieval, similarity, risk_capa, report, gate_flags, errors)`.
2. Each agent is a node; edges encode gate logic (conditional branching).
3. Parallel fan-out: use LangGraph `send_message` / parallel node execution for Agent 4 ‖ Similarity Module; join with `wait_for_all`.
4. Gate logic: pure Python functions with no LLM calls. Gate result written to `gate_flags` in state.
5. Inject `trace_id` at pipeline entry; propagate through state to every node.
6. Schema-validate every handoff using `validate_handoff(stage_name, payload)` from `schemas.py`.
7. Wrap each node in a try/except with timeout enforcement using `asyncio.wait_for`.
8. Log every state transition to LangSmith (or fallback JSON logger to `logs/`).

## Dependencies

- **Blocked by:** `schemas.py` (US-06 equivalent), Agent 3, Agent 4, Agent 2, Agent 6, Similarity Module
- **Blocks:** End-to-end pipeline tests, ablation study runs (A5 baseline needs orchestrator to run in `--baseline-mode`)

## Test Plan

- **Unit:** Each gate function tested in isolation with fixture inputs (low-confidence extraction → halts; zero retrieval → continues with warning; uncited UNACCEPTABLE → halts).
- **Integration:** 5 gold complaints run end-to-end → schema-valid `ReportOutput` produced for all 5.
- **Fault injection:** Inject known-bad Agent 3 output (confidence=0.2) → verify pipeline halts at Gate 1, not at Agent 2.
- **Timeout test:** Simulate Agent 4 hanging → verify pipeline returns partial report within 125s.
- **Cascading failure test:** Inject bad extraction → verify downstream agents never receive it.

## Definition of Done

- [ ] Pipeline runs end-to-end on 5 real complaints producing `ReportOutput`
- [ ] All 3 gate behaviours verified by fault-injection tests
- [ ] Loop caps and timeouts enforced with verified fallback behaviour
- [ ] `trace_id` appears in all log entries for a given run
- [ ] Schema validation fails loudly on malformed payloads (not silently)
