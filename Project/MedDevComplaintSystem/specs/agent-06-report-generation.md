# Agent 6: Report Generation Agent

| Field | Value |
|---|---|
| Epic | Report Assembly |
| Owner | M3 |
| Autonomy Level | L2 Evaluator-Optimizer — self-critique loop, max 2 rounds / 20s |
| Priority | P0 |
| Estimate | 4 days |
| Status | Not started |
| Regulatory Basis | ISO 13485:2016 §4.2.4 (document control); ISO 13485:2016 §8.2.1 (PMS); IEC 62304 §9.3 (advise relevant parties) |

## User Story

As a **Quality Manager**, I want a structured, citation-rich signal report in my organization's controlled document format — ready for Quality Review Board (QRB) review — so that I can make an informed approval decision in under 30 minutes instead of spending the afternoon writing it from scratch.

## Context

Agent 6 is the final step in the live pipeline. It receives all upstream outputs and assembles them into a complete ISO 13485 §4.2.4 controlled document. Unlike the other agents, it uses a self-critique loop (Evaluator-Optimizer pattern) — it generates a draft, critiques it against a 4-point rubric, and revises if needed. The loop is hard-capped at 2 rounds and 20 seconds. After generation, the report is persisted to `signal_reports` SQLite table so the Similarity Module can use past reports as episodic memory when processing future complaints. The QM is the final human gate before the report enters the QMS.

## Scope

- **In:** Filling controlled document template, generating 6 report sections, self-critique and revision (max 2 rounds), AI transparency footer, report persistence to SQLite, returning `ReportOutput` to Orchestrator.
- **Out:** Making any risk acceptability decisions (Agent 2 + human responsibility), QM review (human), QRB decision (human), QMS record entry (human after approval).

## Input / Output

**Input:** All upstream outputs combined
```json
{
  "extraction": "ExtractionOutput",
  "similarity": "SimilarityOutput",
  "retrieval": "RetrievalOutput",
  "risk_capa": "RiskCapaOutput",
  "trace_id": "string"
}
```

**Output:** `ReportOutput`
```json
{
  "document_id": "SR-YYYY-NNNN",
  "revision": "1.0",
  "generated_by": "Signal Intelligence System v1.0",
  "generated_at": "ISO 8601 timestamp",
  "approval_status": "DRAFT",
  "reviewed_by": null,
  "markdown_report": "string — full formatted report",
  "quality": {
    "citation_count": "int — total traceable source references",
    "unsupported_claims": "int — claims without citation",
    "self_score": "float 1.0–5.0 — rubric score from self-critique",
    "review_needed_flag": "boolean — true if rubric still failing after 2 rounds",
    "reflection_rounds_used": "int — 0, 1, or 2"
  },
  "ai_transparency_notice": "AI-assisted draft — requires human review and approval per AIMS policy"
}
```

## Report Structure (6 Sections)

The markdown report must contain exactly these sections, in this order:

### Section 1: Extracted Fields
Structured summary of Agent 3 output: device, manufacturer, modality, component, failure mode, symptom, severity indicator, discovery phase, complaint category.

### Section 2: Pattern Match + Cluster Trend
Similarity Module output: cluster ID and label, cluster size, trend flag (emerging / stable / declining), growth rate over 30 days, similar complaint IDs.

### Section 3: FDA Evidence Citations
Agent 4 output: list of matching adverse events (MAUDE report numbers, relevance scores, snippets) and matching recalls (recall IDs, reason for recall, root cause, corrective action). Every item must show its source reference.

### Section 4: Risk Assessment
Agent 2 ISO 14971 output: hazardous situation, harm, severity × probability → risk level, Annex C hazard category, evidence basis (with citations), uncertainty statement.

### Section 5: CAPA Recommendations
Agent 2 CAPA output: immediate containment, root cause investigation, corrective action (§8.5.2), preventive action (§8.5.3), verification method, effectiveness criteria, timeline, precedent basis (real recall ID cited).

### Section 6: Report Quality Metadata
Self-critique scores, citation count, unsupported claims count, review_needed_flag, and the mandatory AI transparency notice.

## Document Header (ISO 13485 §4.2.4)

```markdown
| Field | Value |
|---|---|
| Document ID | SR-YYYY-NNNN |
| Revision | 1.0 |
| Prepared by | Signal Intelligence System v1.0 |
| Reviewed by | [blank — human approval only] |
| Approval status | DRAFT |
| Generated at | ISO 8601 timestamp |
| Complaint category | [from ExtractionOutput.qms_complaint_category] |
| Risk classification | [from RiskCapaOutput.iso14971_assessment.risk_level] |
| Retention | Lifetime of device + 2 years (per ISO 13485 §4.2.5) |
```

## Self-Critique Rubric (4 Points, 1–5 Scale Each)

| Check | What Is Assessed | Pass Condition |
|---|---|---|
| Citation coverage | Every factual claim in §3, §4, §5 cites a specific FDA record ID | ≥ 90% of claims cited |
| Schema completeness | All 6 sections present; all required header fields populated | All fields present, none null except `reviewed_by` |
| Uncertainty disclosure | Non-confirmable facts are explicitly flagged | At least 1 `uncertainty` statement from Agent 2 is visible in §4 |
| Risk / CAPA consistency | Risk level severity matches CAPA urgency (UNACCEPTABLE → immediate containment stated) | No contradiction between §4 and §5 |

**Stopping rule:**
- All 4 checks pass after draft → accept (0 revision rounds used)
- Any check fails → revise (round 1)
- Any check still fails → revise (round 2)
- Any check still fails after round 2 → accept with `review_needed_flag = true`

**Hard caps:** 2 revision rounds maximum; 20s total for the self-critique loop; if timeout fires → accept with `review_needed_flag = true`.

## Mandatory AI Transparency Footer

Every report must end with this exact text, in a clearly visible section:

> *"AI-assisted draft — requires human review and approval per AIMS policy. This report was generated by Signal Intelligence System v1.0 using GPT-4.1. All factual claims should be verified against source records before QRB presentation. Risk acceptability determination is the responsibility of the Quality Manager per ISO 14971 §6."*

This footer must NOT be modified by the self-critique loop.

## Acceptance Criteria

- [ ] Given all upstream outputs, when report is generated, then all 6 sections are present and the ISO 13485 header is complete (document_id, revision, generated_by, generated_at, approval_status=DRAFT, reviewed_by=null).
- [ ] Given a citation check, when self-critique runs, then every factual claim in §3, §4, §5 has a source reference — or `unsupported_claims` count is incremented.
- [ ] Given a rubric failure on round 1, when round 2 revision runs, then the report is improved (at minimum: one previously uncited claim now has a citation).
- [ ] Given rubric still failing after round 2, when the loop exits, then `review_needed_flag == true` and the report is still returned (not discarded).
- [ ] Given `review_needed_flag == true`, when the Orchestrator returns the report, then the QM interface highlights this flag prominently.
- [ ] Given hallucination rate measurement on 20 gold examples, then percentage of uncited claims < 15%.
- [ ] Given `reviewed_by` field, when any run completes, then `reviewed_by == null` — it is never auto-populated by the system.
- [ ] Given the AI transparency footer, then it appears verbatim at the end of every generated report.
- [ ] Given the self-critique loop running past 20s, when the timeout fires, then the current draft is accepted with `review_needed_flag = true` — the loop does not hang.

## Technical Approach

1. Fill a Markdown template (`configs/report_template.md`) by substituting upstream JSON fields into each section programmatically — minimize LLM calls for structural content.
2. LLM call 1 (draft): pass all upstream summaries; instruct to write §4 (risk narrative) and §5 (CAPA narrative) as prose — these are the sections that require generation, not just data formatting.
3. Self-critique call: pass full draft + rubric; ask for scores and revision instructions.
4. If revision needed: LLM call 2 — pass draft + critique notes; instruct to fix specific issues only.
5. Second self-critique call (if round 2 needed): same as above.
6. Post-process: count citations (regex: `[A-Z]{2}\d+` for MAUDE numbers, `Z-\d{4}-\d{4}` for recalls); increment `unsupported_claims` for any uncited factual claim.
7. Assign `SR-YYYY-NNNN` document ID; compute quality block; append AI transparency footer.
8. Persist to SQLite `signal_reports` table: `(document_id, trace_id, generated_at, approval_status, quality_json, report_markdown)`.
9. Return `ReportOutput` to Orchestrator.

**Token budget:** ~5K tokens total for this component (3K input context + 2K output). Never pass raw upstream payloads — use summarized versions from Orchestrator state.

## Dependencies

- **Blocked by:** `schemas.py`, Agent 3 (ExtractionOutput), Agent 4 (RetrievalOutput), Agent 2 (RiskCapaOutput), Similarity Module (SimilarityOutput), SQLite `signal_reports` table schema
- **Blocks:** End-to-end pipeline tests, DPO training (needs report drafts for preference pairs), ablation A2 (no-reflection ablation), ablation A3 (no-DPO ablation)

## Test Plan

- **Unit:**
  - Mock all 4 upstream inputs → assert schema-valid `ReportOutput` with 6 sections
  - Missing citation in draft → assert self-critique triggers revision (round 1)
  - Rubric still failing after round 2 → assert `review_needed_flag == true`, report still returned
  - 20s timeout simulation → assert loop exits with `review_needed_flag == true`, no hang
  - `reviewed_by` field → assert always `null` after generation
  - AI footer → assert verbatim footer text appears in every report
- **Eval:** Hallucination rate (uncited claims %) on 20 gold examples; target < 15%. Self-score vs human rubric kappa; target > 0.60.
- **Ablation A2:** Run without self-critique → measure hallucination rate increase vs full pipeline.

## Definition of Done

- [ ] Report generated with all 6 sections and ISO 13485 header on 5 real complaints
- [ ] Self-critique loop terminates within 2 rounds and 20s on all test cases
- [ ] `review_needed_flag` correctly set on known-hard cases (incomplete evidence, conflicting risk signals)
- [ ] Hallucination rate < 15% on gold set
- [ ] Report persisted to SQLite `signal_reports` table; retrievable by `trace_id`
- [ ] AI transparency footer present verbatim in every report
- [ ] `reviewed_by == null` in all generated reports (never auto-populated)
