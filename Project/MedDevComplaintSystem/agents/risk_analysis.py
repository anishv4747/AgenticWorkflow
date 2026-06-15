"""
Risk Analysis Agent — full ISO 14971:2019 implementation.

Two-pass design:
    Pass 1  →  generate complete risk assessment + CAPA from complaint context
    Pass 2  →  self-critique against a checklist; return revised assessment

Constitutional guardrail (enforced in Python after Pass 2):
    ALARP or UNACCEPTABLE with zero evidence citations → forced downgrade to ACCEPTABLE.

Escalation flags are computed deterministically in Python, never by the LLM.

Episodic memory:
    load_past_reports() queries SQLite signal_reports table for prior similar cases.
    The table is created by init_db() on first use (idempotent).
    save_report() is called by the report agent, not here — this module only reads.
"""


import json
import sqlite3
import logging
from pathlib import Path
from anthropic import Anthropic

logger = logging.getLogger(__name__)

# ── SQLite setup ──────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "data" / "signal_reports.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS signal_reports (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id           TEXT NOT NULL,
    trace_id              TEXT NOT NULL,
    generated_at          TEXT NOT NULL,
    failure_mode          TEXT,
    modality              TEXT,
    qms_complaint_category TEXT,
    risk_level            TEXT,
    severity_level        TEXT,
    probability_level     TEXT,
    evidence_count        INTEGER DEFAULT 0,
    capa_precedent        TEXT,
    approval_status       TEXT DEFAULT 'DRAFT',
    report_json           TEXT
);
"""

_db_initialized = False


def init_db(db_path: Path = _DB_PATH) -> None:
    """Create signal_reports table if it does not exist. Safe to call multiple times."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()


def _ensure_db() -> None:
    global _db_initialized
    if not _db_initialized:
        init_db()
        _db_initialized = True


def load_past_reports(
    failure_mode: str,
    modality: str,
    limit: int = 5,
    db_path: Path = _DB_PATH,
) -> list[dict]:
    """
    Return up to `limit` past signal reports matching this failure_mode and modality.
    Returns [] if the table is empty or no matches found — caller handles gracefully.
    """
    _ensure_db()
    query = """
        SELECT document_id, generated_at, failure_mode, modality,
               qms_complaint_category, risk_level, severity_level,
               probability_level, evidence_count, capa_precedent
        FROM signal_reports
        WHERE modality = ? OR failure_mode LIKE ?
        ORDER BY generated_at DESC
        LIMIT ?
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (modality, f"%{failure_mode[:30]}%", limit)).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        logger.warning("load_past_reports failed: %s", exc)
        return []


def save_report(
    document_id: str,
    trace_id: str,
    generated_at: str,
    failure_mode: str,
    modality: str,
    qms_complaint_category: str,
    risk_level: str,
    severity_level: str,
    probability_level: str,
    evidence_count: int,
    capa_precedent: str,
    report_json: str,
    db_path: Path = _DB_PATH,
) -> None:
    """Persist a completed report to episodic memory. Called by the report agent."""
    _ensure_db()
    sql = """
        INSERT INTO signal_reports
            (document_id, trace_id, generated_at, failure_mode, modality,
             qms_complaint_category, risk_level, severity_level,
             probability_level, evidence_count, capa_precedent, report_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(sql, (
            document_id, trace_id, generated_at, failure_mode, modality,
            qms_complaint_category, risk_level, severity_level,
            probability_level, evidence_count, capa_precedent, report_json,
        ))
        conn.commit()


# ── Prompts ───────────────────────────────────────────────────────────────────

_PASS1_SYSTEM = """\
You are a medical device risk analyst applying ISO 14971:2019.

Your task: given a device complaint, prior FDA evidence, trend data, and any past similar
signal reports, produce a complete risk assessment and CAPA recommendation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ISO 14971 SEVERITY SCALE (Annex D)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
S1 (Negligible)   – No injury; no clinical impact; inconvenience only.
S2 (Minor)        – Minor, reversible injury; minor delay in diagnosis.
S3 (Serious)      – Serious injury; major delay in diagnosis; repeat procedure required.
S4 (Critical)     – Permanent injury; surgical intervention required.
S5 (Catastrophic) – Death.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ISO 14971 PROBABILITY SCALE (calibrated to ~14,000 FDA imaging device events)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P1 (Improbable) – < 1 per 1,000,000 uses.
P2 (Remote)     – 1 per 100,000 to 1 per 1,000,000.
P3 (Occasional) – 1 per 10,000 to 1 per 100,000.
P4 (Probable)   – 1 per 1,000 to 1 per 10,000.
P5 (Frequent)   – > 1 per 1,000.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK ACCEPTABILITY MATRIX (5×5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCEPTABLE:    S1×(P1–P5)  S2×(P1–P3)  S3×(P1–P2)  S4×P1   S5×P1
ALARP:         S2×(P4–P5)  S3×(P3–P4)  S4×(P2–P3)  S5×(P2–P3)
UNACCEPTABLE:  S3×P5  S4×(P4–P5)  S5×(P4–P5)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE CITATION REQUIREMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEVER assign ALARP or UNACCEPTABLE unless evidence_basis contains at least one specific
FDA record (MAUDE report number or recall ID) that justifies the probability estimate.
If you cannot cite evidence, assign ACCEPTABLE and document the uncertainty.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING APPROACH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Identify the hazardous situation (device failure → patient/operator exposure path).
2. Identify the specific harm (what injury or consequence results).
3. Justify severity S1–S5 from the complaint and evidence.
4. Justify probability P1–P5 using event counts, recall history, and trend data.
5. Apply the matrix to determine risk_level.
6. Cite real FDA record IDs in evidence_basis — do not invent IDs.
7. Generate CAPA proportionate to risk_level:
   UNACCEPTABLE → immediate containment within hours is mandatory.
   ALARP         → immediate action and formal investigation.
   ACCEPTABLE    → investigation and preventive action.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object. No markdown fences, no prose outside the JSON.

{
  "chain_of_thought": "Step-by-step reasoning (not shown to QM, used for audit trail)",
  "hazardous_situation": "Specific situation: device failure → operator/patient exposure path",
  "harm": "Specific harm: injury or consequence to patient or operator",
  "severity_level": "S1|S2|S3|S4|S5",
  "severity_rationale": "1–2 sentences citing complaint language and evidence",
  "probability_level": "P1|P2|P3|P4|P5",
  "probability_rationale": "Cite specific FDA record IDs to justify — e.g. '3 MAUDE events (MW3021547, MW2998341, MW3014892) + 1 Class II recall (Z-2024-00423) for the same failure mode'",
  "risk_level": "ACCEPTABLE|ALARP|UNACCEPTABLE",
  "evidence_basis": [
    {"source": "MAUDE|RECALL", "id": "string", "relevance": "one sentence why this record applies"}
  ],
  "uncertainty": "What is unknown or unconfirmed — e.g. patient outcome not reported, root cause not yet verified",
  "capa_immediate": "Immediate containment action and timeline (e.g. 'Within 24h: notify field service to check SW version at all affected sites')",
  "capa_investigation": "Root cause investigation steps and responsible function",
  "capa_corrective": "Corrective action per ISO 13485 §8.5.2 (fix the root cause)",
  "capa_preventive": "Preventive action per ISO 13485 §8.5.3 (prevent recurrence in other products)",
  "capa_verification": "How to verify the CAPA is effective (test, audit, metric)",
  "capa_effectiveness": "Measurable criteria for effectiveness (e.g. '0 recurrence events in 90 days post-fix')",
  "capa_precedent": "Real MAUDE report number or recall ID used as basis for the CAPA recommendation"
}
"""

_PASS2_USER = """\
You have just produced the above risk assessment. Now review it against this checklist:

CHECKLIST:
1. MATRIX MATH — Does your S×P combination map correctly to the risk_level per the defined matrix?
   Verify cell by cell. Correct if wrong.

2. CITATION COVERAGE — If risk_level is ALARP or UNACCEPTABLE, does evidence_basis contain at
   least one specific FDA record ID? If not, either add citations from the provided evidence or
   downgrade risk_level to ACCEPTABLE.

3. CAPA PROPORTIONALITY —
   • UNACCEPTABLE: capa_immediate must specify a concrete action within hours.
   • ALARP: capa_immediate must be present and specific.
   • ACCEPTABLE: capa_investigation should still be present.
   Fix any mismatch.

4. PRECEDENT LINKAGE — Does capa_precedent cite a real ID from the evidence provided
   (matching_recalls or matching_events)? If you used a fabricated ID, correct it to a real one
   from the input, or set it to null.

5. UNCERTAINTY — Does the uncertainty field capture what would change this assessment
   (e.g. if root cause confirmed, if patient outcome unknown, if trending upward)?

Return ONLY a revised JSON object with the same schema as your draft.
If a field needs no change, carry it forward unchanged.
Do not add commentary outside the JSON.
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

def risk_analysis_agent(state: dict) -> dict:
    """
    Full ISO 14971 risk assessment — two-pass with episodic memory.

    Reads from state (set by upstream agents):
        failure_mode, severity_indicator, modality, manufacturer, device_model,
        software_version, component, qms_complaint_category, is_safety_related
          ← written by extraction_agent

        matching_events, matching_recalls, regulatory_context
          ← written by retrieval_agent

        cluster_label, cluster_size, trend_flag, growth_rate_30d, similar_event_ids
          ← written by similarity_module

    Writes to state:
        risk_level, hazardous_situation, harm,
        severity_level, severity_rationale,
        probability_level, probability_rationale,
        evidence_basis, uncertainty,
        capa_immediate, capa_investigation, capa_corrective, capa_preventive,
        capa_verification, capa_effectiveness, capa_precedent,
        escalation_required, prrc_notification_required, fsca_required,
        gate3_passed, messages
    """
    _ensure_db()
    client = Anthropic()
    W = 64

    print(f"\n{'─' * W}")
    print("  RISK ANALYSIS AGENT  [LIVE — claude-sonnet-4-6]")
    print(f"{'─' * W}")

    # ── 1. Episodic memory: load past similar reports ─────────────────────
    failure_mode = state.get("failure_mode") or ""
    modality = state.get("modality") or ""
    past_reports = load_past_reports(failure_mode, modality)

    print(f"  ← reading from state (upstream agents):")
    print(f"      {'failure_mode':<28} = {failure_mode}")
    print(f"      {'modality':<28} = {modality}")
    print(f"      {'manufacturer':<28} = {state.get('manufacturer')}")
    print(f"      {'device_model':<28} = {state.get('device_model')}")
    print(f"      {'software_version':<28} = {state.get('software_version')}")
    print(f"      {'severity_indicator':<28} = {state.get('severity_indicator')}  (from extraction)")
    print(f"      {'qms_complaint_category':<28} = {state.get('qms_complaint_category')}")
    print(f"      {'matching_events':<28} = {len(state.get('matching_events') or [])} events")
    print(f"      {'matching_recalls':<28} = {len(state.get('matching_recalls') or [])} recall(s)")
    print(f"      {'cluster_label':<28} = {state.get('cluster_label')}")
    print(f"      {'trend_flag':<28} = {state.get('trend_flag')} "
          f"(growth {state.get('growth_rate_30d')})")
    print(f"      {'past_signal_reports (SQLite)':<28} = {len(past_reports)} matching record(s)")

    if past_reports:
        past_context = "\n## Past Similar Signal Reports (from episodic memory)\n"
        for r in past_reports:
            past_context += (
                f"- {r['document_id']} ({r['generated_at'][:10]}): "
                f"{r['failure_mode']} / {r['modality']} → {r['risk_level']} "
                f"[{r['severity_level']}×{r['probability_level']}] "
                f"CAPA precedent: {r.get('capa_precedent', 'N/A')}\n"
            )
    else:
        past_context = "\n## Past Similar Signal Reports\nNone found in episodic memory.\n"

    # ── 2. Build context block from state fields ───────────────────────────
    context = f"""
## Complaint
{state['complaint_text']}

## Extracted Fields
- Modality:            {state.get('modality')}
- Device:              {state.get('device_model')} by {state.get('manufacturer')}
- Software Version:    {state.get('software_version')}
- Component:           {state.get('component')}
- Failure Mode:        {state.get('failure_mode')}
- Severity Indicator:  {state.get('severity_indicator')} (initial estimate from extraction)
- QMS Category:        {state.get('qms_complaint_category')}
- Safety Related:      {state.get('is_safety_related')}

## FDA Evidence
{state.get('regulatory_context', 'No regulatory context available.')}

Matching Adverse Events:
{json.dumps(state.get('matching_events', []), indent=2)}

Matching Recalls:
{json.dumps(state.get('matching_recalls', []), indent=2)}

## Trend Data
- Cluster:       {state.get('cluster_label')} (size: {state.get('cluster_size', 'unknown')})
- Trend:         {state.get('trend_flag')} (30-day growth rate: {state.get('growth_rate_30d')})
- Similar IDs:   {state.get('similar_event_ids', [])}
{past_context}"""

    # ── 3. Pass 1 — generate initial assessment ───────────────────────────
    # Anthropic: system is a top-level param; messages list is user/assistant only.
    # Prefill with "{" forces the model to open a JSON object immediately.
    print(f"\n  [Pass 1] calling claude-sonnet-4-6 → initial ISO 14971 assessment ...")
    p1_user_content = f"Perform ISO 14971 risk assessment:\n{context}"
    p1_turn = [
        {"role": "user",      "content": p1_user_content},
        {"role": "assistant", "content": "{"},          # JSON prefill
    ]
    p1_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=_PASS1_SYSTEM,
        messages=p1_turn,
        temperature=0.2,
    )
    p1_raw = "{" + p1_response.content[0].text         # re-attach prefill
    p1_result = json.loads(p1_raw)

    print(f"  [Pass 1] done.")
    print(f"      risk_level       = {p1_result.get('risk_level')}")
    print(f"      severity         = {p1_result.get('severity_level')} — {p1_result.get('severity_rationale', '')[:60]}")
    print(f"      probability      = {p1_result.get('probability_level')} — {p1_result.get('probability_rationale', '')[:60]}")
    print(f"      evidence_basis   = {len(p1_result.get('evidence_basis', []))} citation(s)")
    for ev in p1_result.get("evidence_basis", []):
        print(f"          [{ev.get('source')}] {ev.get('id')} — {ev.get('relevance', '')[:50]}")

    # ── 4. Pass 2 — self-critique and revision (always runs) ─────────────
    print(f"\n  [Pass 2] self-critique checklist (matrix math, citations, CAPA, precedent) ...")
    p2_turn = [
        {"role": "user",      "content": p1_user_content},
        {"role": "assistant", "content": p1_raw},
        {"role": "user",      "content": _PASS2_USER},
        {"role": "assistant", "content": "{"},          # JSON prefill
    ]
    p2_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=_PASS1_SYSTEM,
        messages=p2_turn,
        temperature=0.1,
    )
    p2_raw = "{" + p2_response.content[0].text
    result = json.loads(p2_raw)

    changed = result.get("risk_level") != p1_result.get("risk_level") or \
              len(result.get("evidence_basis", [])) != len(p1_result.get("evidence_basis", []))
    print(f"  [Pass 2] done. {'(changes made)' if changed else '(no changes from Pass 1)'}")
    print(f"      risk_level       = {result.get('risk_level')}")
    print(f"      evidence_basis   = {len(result.get('evidence_basis', []))} citation(s)")
    for ev in result.get("evidence_basis", []):
        print(f"          [{ev.get('source')}] {ev.get('id')} — {ev.get('relevance', '')[:50]}")

    # ── 5. Hard guardrail — enforced in Python after both passes ─────────
    risk = result.get("risk_level", "ACCEPTABLE")
    evidence = result.get("evidence_basis", [])

    if risk in ("ALARP", "UNACCEPTABLE") and len(evidence) == 0:
        print(f"\n  [Guardrail] ✗ {risk} with 0 citations — force-downgrading to ACCEPTABLE")
        logger.warning(
            "[risk_analysis_agent] Guardrail: %s with 0 citations after Pass 2. "
            "Force-downgrading to ACCEPTABLE.",
            risk,
        )
        result["risk_level"] = "ACCEPTABLE"
        result["uncertainty"] = (
            f"[DOWNGRADED from {risk}] Model assigned elevated risk but provided no "
            "FDA citations after two passes. Assessment requires human review with "
            "direct evidence lookup. " + (result.get("uncertainty") or "")
        )
        risk = "ACCEPTABLE"
        evidence = []

    # ── 6. Escalation flags — deterministic, never by LLM ────────────────
    escalation_required = risk in ("ALARP", "UNACCEPTABLE")
    prrc_notification   = risk == "UNACCEPTABLE"
    # fsca_required needs confirmed root cause + active distribution — human decision only
    fsca_required       = False

    gate3_passed = not (risk == "UNACCEPTABLE" and len(evidence) == 0)

    all_messages = (state.get("messages") or []) + [
        {"role": "user",      "content": p1_user_content},
        {"role": "assistant", "content": p1_raw},
        {"role": "user",      "content": _PASS2_USER},
        {"role": "assistant", "content": p2_raw},
    ]

    print(f"\n  → writing to state:")
    print(f"      {'risk_level':<28} = {risk}")
    print(f"      {'severity_level':<28} = {result.get('severity_level')}")
    print(f"      {'probability_level':<28} = {result.get('probability_level')}")
    print(f"      {'hazardous_situation':<28} = {(result.get('hazardous_situation') or '')[:60]}")
    print(f"      {'harm':<28} = {(result.get('harm') or '')[:60]}")
    print(f"      {'evidence_basis':<28} = {len(evidence)} citation(s)")
    print(f"      {'escalation_required':<28} = {escalation_required}")
    print(f"      {'prrc_notification_required':<28} = {prrc_notification}")
    print(f"      {'fsca_required':<28} = {fsca_required}")
    print(f"      {'gate3_passed':<28} = {gate3_passed}")
    print(f"      {'capa_immediate':<28} = {(result.get('capa_immediate') or '')[:60]}")
    print(f"      {'capa_precedent':<28} = {result.get('capa_precedent')}")
    print(f"      {'uncertainty':<28} = {(result.get('uncertainty') or '')[:60]}")

    return {
        "hazardous_situation":        result.get("hazardous_situation"),
        "harm":                       result.get("harm"),
        "severity_level":             result.get("severity_level"),
        "severity_rationale":         result.get("severity_rationale"),
        "probability_level":          result.get("probability_level"),
        "probability_rationale":      result.get("probability_rationale"),
        "risk_level":                 risk,
        "evidence_basis":             evidence,
        "uncertainty":                result.get("uncertainty"),
        "capa_immediate":             result.get("capa_immediate"),
        "capa_investigation":         result.get("capa_investigation"),
        "capa_corrective":            result.get("capa_corrective"),
        "capa_preventive":            result.get("capa_preventive"),
        "capa_verification":          result.get("capa_verification"),
        "capa_effectiveness":         result.get("capa_effectiveness"),
        "capa_precedent":             result.get("capa_precedent"),
        "escalation_required":        escalation_required,
        "prrc_notification_required": prrc_notification,
        "fsca_required":              fsca_required,
        "gate3_passed":               gate3_passed,
        "messages":                   all_messages,
    }
