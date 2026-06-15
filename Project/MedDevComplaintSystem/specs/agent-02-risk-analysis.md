# Agent 2: Risk Analysis Agent

| Field | Value |
|---|---|
| Epic | Risk Assessment |
| Owner | M5 |
| Autonomy Level | L1 Augmented LLM — single-pass CoT + 1 self-reflection round |
| Priority | P0 |
| Estimate | 5 days |
| Status | Not started |
| Regulatory Basis | ISO 14971:2019 §7.3–7.6, Annex C, Annex D; ISO 13485 §8.5.2/8.5.3; IEC 62304 §9.7 |

## User Story

As a **Quality Manager**, I want a structured risk assessment grounded in real FDA evidence and ISO 14971 methodology, so that I can make an informed decision about whether to escalate a complaint, initiate a CAPA, or accept the current risk level — without spending 60–90 minutes researching it myself.

## Context

Agent 2 is the regulatory reasoning core of the pipeline. It receives the structured extraction (Agent 3) and FDA evidence (Agent 4 + cluster context from Similarity Module), then applies ISO 14971:2019 risk methodology to produce a draft risk assessment and CAPA recommendation. The output is evidence-grounded — the agent is constitutionally blocked from asserting ALARP or UNACCEPTABLE risk without citing at least one specific FDA record. Risk and CAPA are merged into a single LLM call to avoid over-orchestration (see CLAUDE.md rationale). Results are returned to Agent 1 (Orchestrator).

## Scope

- **In:** ISO 14971 §7.3 hazard identification, §7.4 risk estimation (S×P matrix), §7.5 risk evaluation, §7.6 risk control (CAPA), escalation flag determination.
- **Out:** Final risk acceptability determination (§7.2/§7.3/§8 — human judgment per ISO 14971); regulatory submission drafting; residual risk evaluation (human decision).

## Input / Output

**Input:**
```json
{
  "extraction": "ExtractionOutput",
  "retrieval": "RetrievalOutput",
  "cluster_context": "SimilarityOutput"
}
```

**Output:** `RiskCapaOutput`
```json
{
  "iso14971_assessment": {
    "hazardous_situation": "string — Annex C category",
    "harm": "string",
    "severity": { "level": "S1–S5", "label": "string", "rationale": "string" },
    "probability": { "level": "P1–P5", "label": "string", "rationale": "string (cite event counts)" },
    "risk_level": "ACCEPTABLE | ALARP | UNACCEPTABLE",
    "risk_control_needed": "boolean",
    "annex_c_hazard_category": "string"
  },
  "evidence_basis": [
    { "source": "MAUDE | Recall | Cluster", "id": "string", "relevance": "string" }
  ],
  "uncertainty": "string — what cannot be confirmed from available evidence",
  "iec62304_classification": "A | B | C",
  "capa_recommendation": {
    "immediate_containment": "string",
    "root_cause_investigation": "string",
    "corrective_action": "string — per ISO 13485 §8.5.2",
    "preventive_action": "string — per ISO 13485 §8.5.3",
    "verification_method": "string",
    "effectiveness_criteria": "string",
    "timeline": "string",
    "precedent_basis": "string — cite real recall ID or MAUDE report number",
    "iso13485_clause": "§8.5.2 | §8.5.3 | both"
  },
  "escalation_flags": {
    "escalation_required": "boolean",
    "prrc_notification_required": "boolean",
    "fsca_required": "boolean",
    "rationale": "string"
  }
}
```

## ISO 14971 Risk Methodology

### Severity Scale (Annex D)

| Level | Code | Definition | Medical Device Example |
|---|---|---|---|
| S1 | Negligible | Inconvenience, no injury | UI freeze requiring restart, no data loss |
| S2 | Minor | Temporary injury, no intervention required | Image artifact caught before clinical use |
| S3 | Serious | Injury requiring medical intervention | Missed diagnosis; repeat procedure with radiation |
| S4 | Critical | Permanent impairment or life-threatening | False negative leading to delayed cancer treatment |
| S5 | Catastrophic | Death | Device malfunction during life-critical intervention |

### Probability Scale (calibrated to openFDA dataset)

| Level | Code | Definition | Dataset Calibration |
|---|---|---|---|
| P1 | Incredible | < 1 in 100,000 uses | 0 events in working dataset |
| P2 | Improbable | 1 in 10,000–100,000 | 1–2 events per product code |
| P3 | Remote | 1 in 1,000–10,000 | 3–20 events per product code |
| P4 | Occasional | 1 in 100–1,000 | 20–200 events per product code |
| P5 | Frequent | > 1 in 100 | > 200 events per product code |

### Risk Acceptability Matrix (§7.5)

|  | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| **S5** | ALARP | UNACCEPTABLE | UNACCEPTABLE | UNACCEPTABLE | UNACCEPTABLE |
| **S4** | ACCEPTABLE | ALARP | UNACCEPTABLE | UNACCEPTABLE | UNACCEPTABLE |
| **S3** | ACCEPTABLE | ACCEPTABLE | ALARP | UNACCEPTABLE | UNACCEPTABLE |
| **S2** | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE | ALARP | ALARP |
| **S1** | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE | ALARP |

### Escalation Logic (deterministic — not LLM-assessed)

```
escalation_required      = risk_level in {ALARP, UNACCEPTABLE}
prrc_notification_required = risk_level == UNACCEPTABLE
fsca_required            = confirmed_root_cause AND risk_level == UNACCEPTABLE AND active_distribution
```

## Constitutional Guardrails

1. **Citation requirement:** The agent MUST NOT output `risk_level = ALARP` or `risk_level = UNACCEPTABLE` if `evidence_basis[]` is empty. If no evidence is available, risk_level must be `ACCEPTABLE` and uncertainty must explain the lack of evidence.
2. **Enum enforcement:** `risk_level` must be one of `{ACCEPTABLE, ALARP, UNACCEPTABLE}`. The strings `HIGH`, `MEDIUM`, `LOW` are invalid and must be rejected at Gate 3.
3. **Uncertainty honesty:** The `uncertainty` field must describe what cannot be confirmed (e.g., "Cannot confirm if same software version is affected — narrative lacks version information").
4. **Precedent basis:** CAPA `precedent_basis` must cite a real recall ID or MAUDE report number from the retrieval results. If none is available, field is `"No direct precedent found in evidence"` — not a fabricated reference.

## Acceptance Criteria

- [ ] Given S3 severity and P4 probability, when risk matrix is applied, then `risk_level == UNACCEPTABLE` (verify specific cell from matrix above).
- [ ] Given `evidence_basis == []`, when agent tries to output `ALARP`, then constitutional guardrail rejects this and forces `ACCEPTABLE` with uncertainty explanation.
- [ ] Given `risk_level == UNACCEPTABLE`, when escalation flags are set, then `escalation_required == true` AND `prrc_notification_required == true`.
- [ ] Given `risk_level == ALARP`, when escalation flags are set, then `escalation_required == true` AND `prrc_notification_required == false`.
- [ ] Given CAPA output, when `precedent_basis` is checked, then it references a real recall ID present in `retrieval.matching_recalls[]` — not a fabricated ID.
- [ ] Given expert rubric evaluation on 20 gold examples, then average score > 3.0 / 5.0.
- [ ] Given self-reflection pass, when output contains a claim without a citation, then it is either removed or tagged in `uncertainty`.

## Technical Approach

1. System prompt encodes ISO 14971 methodology: severity table + probability table + 5×5 matrix as literal lookup, not free-form reasoning.
2. CoT structure enforced in prompt: `evidence → hazardous situation → harm → severity (cite evidence) → probability (cite event counts) → matrix lookup → risk_level → CAPA`.
3. Constitutional guardrail implemented as a post-processing check before schema validation: if `risk_level != ACCEPTABLE` and `len(evidence_basis) == 0`, override risk_level to ACCEPTABLE and append to uncertainty.
4. Escalation flags computed deterministically in Python after LLM output is parsed — not by the LLM.
5. CAPA RAG: retrieve top recall `action` fields from `RetrievalOutput.matching_recalls` and include in prompt as precedent examples.
6. Self-reflection pass (1 round): critique prompt checks each claim for a citation; adds missing citations to uncertainty or removes unsupported claim.
7. Output validated against `RiskCapaOutput` Pydantic schema before returning to Orchestrator.

## Dependencies

- **Blocked by:** `schemas.py`, Agent 3 (ExtractionOutput), Agent 4 (RetrievalOutput), Similarity Module (SimilarityOutput)
- **Blocks:** Agent 6 (Report Generation), Agent 1 Gate 3 check, ablation studies A1/A2/A5

## Test Plan

- **Unit:**
  - Matrix consistency: inject S3 × P4 → assert `UNACCEPTABLE`
  - Constitutional guardrail: inject empty evidence → assert `risk_level == ACCEPTABLE`
  - Escalation truth table: 5 combinations of risk_level × evidence state
  - Enum rejection: assert `HIGH` / `MEDIUM` / `LOW` fail schema validation
  - Fabricated recall ID: inject ID not in `matching_recalls[]` → assert stripped from CAPA
- **Eval:** Rubric scoring on 20 gold examples (human + LLM-Judge); target > 3.0/5; citation coverage rate
- **Resilience:** Inject retrieval output with 0 recalls → agent proceeds with warning, not crash

## Definition of Done

- [ ] ISO 14971 5×5 matrix correctly implemented and tested for all 25 cells
- [ ] Constitutional guardrail blocks uncited ALARP/UNACCEPTABLE
- [ ] Escalation flags match the three-rule truth table in all test cases
- [ ] CAPA output always cites a real recall ID or states "No direct precedent found"
- [ ] Expert rubric > 3.0/5 on gold set
- [ ] `RiskCapaOutput` Pydantic schema validates cleanly on all test outputs
