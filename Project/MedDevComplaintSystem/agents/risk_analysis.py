"""
Risk Analysis Agent — tool-first ISO 14971:2019 implementation.

Design: severity, probability, risk_level, CAPA timing, and escalation flags
are computed by DETERMINISTIC TOOLS (agents/risk_tools.py) — never generated
freehand by an LLM. The same complaint + same evidence always produces the
same numbers.

Flow:
    1. score_severity(complaint_text)                          — tool
    2. compute_probability(event_count, recall_count, ...)     — tool
    3. apply_risk_matrix(severity_level, probability_level)    — tool
    4. load_past_reports(failure_mode, modality)                — tool (SQLite)
    5. lookup_capa_requirements(risk_level)                     — tool
    6. LLM call — narrative ONLY, given the fixed numbers above
    7. validate_evidence_citations(...)                         — tool (strips hallucinated IDs)
    8. compute_escalation_flags(risk_level, evidence_count)     — tool

The LLM never decides severity_level, probability_level, or risk_level — it
writes the hazardous_situation/harm/rationale/CAPA prose that explains a
result the tools already fixed.

Episodic memory:
    load_past_reports() queries SQLite signal_reports table for prior similar cases.
    The table is created by init_db() on first use (idempotent).
    save_report() is called by the report agent, not here — this module only reads.
"""

import re
import json
import sqlite3
import logging
from pathlib import Path
from anthropic import Anthropic

from agents.risk_tools import (
    score_severity,
    compute_probability,
    apply_risk_matrix,
    validate_evidence_citations,
    lookup_capa_requirements,
    compute_escalation_flags,
)

logger = logging.getLogger(__name__)


def _parse_json_response(text: str) -> dict:
    """
    claude-sonnet-4-6 doesn't support assistant-message prefill, so we can't
    force the response to open with '{'. Strip markdown fences if the model
    wrapped the JSON in them, then parse the first {...} block.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(text[start:end + 1])


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


# ── Narrative prompt — the LLM explains fixed numbers, it does not set them ───

_NARRATIVE_SYSTEM = """\
You are a medical device risk analyst writing the narrative for an ISO 14971:2019 risk
assessment.

CRITICAL: severity_level, probability_level, and risk_level have ALREADY been determined
by deterministic rule-based tools — you are NOT deciding them and you MUST NOT contradict
them. Your job is narrower:

1. Write hazardous_situation and harm — describe the specific failure → exposure → harm
   pathway, grounded in the complaint text.
2. Write severity_rationale and probability_rationale — explain, in plain language, why the
   GIVEN severity_level and probability_level fit this complaint and evidence. Reference the
   matched keywords / evidence counts you are given.
3. Propose evidence_basis citations — ONLY use MAUDE report numbers or recall IDs that
   literally appear in the "Available Evidence" section below. Do not invent IDs. If no
   evidence is available, return an empty list and say so in uncertainty.
4. Write a CAPA narrative consistent with the REQUIRED CAPA PARAMETERS you are given
   (e.g. if containment_timeline_hours=24, capa_immediate must reflect a 24-hour timeline;
   if prrc_notification_required=true, capa_immediate must mention PRRC notification).
5. Write uncertainty — what is unknown or unconfirmed, and what would change the assessment.

Return ONLY a valid JSON object, no markdown fences, no prose outside the JSON:
{
  "hazardous_situation": "string",
  "harm": "string",
  "severity_rationale": "string",
  "probability_rationale": "string",
  "evidence_basis": [
    {"source": "MAUDE|RECALL", "id": "string (must exist in Available Evidence)", "relevance": "one sentence"}
  ],
  "uncertainty": "string",
  "capa_immediate": "string",
  "capa_investigation": "string",
  "capa_corrective": "string",
  "capa_preventive": "string",
  "capa_verification": "string",
  "capa_effectiveness": "string",
  "capa_precedent": "string — real recall or MAUDE ID from Available Evidence, or null"
}
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

def risk_analysis_agent(state: dict) -> dict:
    """
    Tool-first ISO 14971 risk assessment with episodic memory.

    Reads from state (set by upstream agents):
        complaint_text, failure_mode, modality, manufacturer, device_model,
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
    print("  RISK ANALYSIS AGENT  [tools: deterministic | narrative: claude-sonnet-4-6]")
    print(f"{'─' * W}")

    failure_mode = state.get("failure_mode") or ""
    modality = state.get("modality") or ""
    matching_events = state.get("matching_events") or []
    matching_recalls = state.get("matching_recalls") or []
    similar_event_ids = state.get("similar_event_ids") or []

    print(f"  ← reading from state (upstream agents):")
    print(f"      {'failure_mode':<28} = {failure_mode}")
    print(f"      {'modality':<28} = {modality}")
    print(f"      {'matching_events':<28} = {len(matching_events)} events")
    print(f"      {'matching_recalls':<28} = {len(matching_recalls)} recall(s)")
    print(f"      {'cluster_size':<28} = {state.get('cluster_size')}")
    print(f"      {'trend_flag':<28} = {state.get('trend_flag')} (growth {state.get('growth_rate_30d')})")

    # ── TOOL 1: severity ────────────────────────────────────────────────────
    severity = score_severity(state["complaint_text"])
    print(f"\n  [TOOL] score_severity()")
    print(f"      severity_level   = {severity['severity_level']}"
          f"{'  (default — no keyword match)' if severity['default_applied'] else ''}")
    print(f"      matched_phrases  = {severity['matched_phrases']}")

    # ── TOOL 2: probability ─────────────────────────────────────────────────
    probability = compute_probability(
        event_count=len(matching_events),
        recall_count=len(matching_recalls),
        growth_rate_30d=state.get("growth_rate_30d"),
        cluster_size=state.get("cluster_size"),
    )
    print(f"  [TOOL] compute_probability()")
    print(f"      probability_level = {probability['probability_level']}")
    print(f"      basis              = {probability['basis']}")

    # ── TOOL 3: risk matrix lookup (pure dict, no LLM) ──────────────────────
    risk_level = apply_risk_matrix(severity["severity_level"], probability["probability_level"])
    print(f"  [TOOL] apply_risk_matrix({severity['severity_level']}, {probability['probability_level']}) = {risk_level}")

    # ── TOOL 4: episodic memory ──────────────────────────────────────────────
    past_reports = load_past_reports(failure_mode, modality)
    print(f"  [TOOL] load_past_reports() = {len(past_reports)} matching record(s)")

    # ── TOOL 5: CAPA requirements ─────────────────────────────────────────────
    capa_reqs = lookup_capa_requirements(risk_level)
    print(f"  [TOOL] lookup_capa_requirements({risk_level}) = {capa_reqs}")

    if past_reports:
        past_context = "\n## Past Similar Signal Reports (from episodic memory)\n"
        for r in past_reports:
            past_context += (
                f"- {r['document_id']} ({r['generated_at'][:10]}): "
                f"{r['failure_mode']} / {r['modality']} → {r['risk_level']} "
                f"[{r['severity_level']}×{r['probability_level']}]\n"
            )
    else:
        past_context = "\n## Past Similar Signal Reports\nNone found in episodic memory.\n"

    # ── Build context for the narrative-only LLM call ──────────────────────
    context = f"""
## Complaint
{state['complaint_text']}

## Extracted Fields
- Modality:            {state.get('modality')}
- Device:              {state.get('device_model')} by {state.get('manufacturer')}
- Software Version:    {state.get('software_version')}
- Component:           {state.get('component')}
- Failure Mode:        {state.get('failure_mode')}
- QMS Category:        {state.get('qms_complaint_category')}

## FIXED VALUES (already determined by deterministic tools — do not change these)
- severity_level:    {severity['severity_level']}  (matched: {severity['matched_phrases']})
- probability_level: {probability['probability_level']}  (basis: {probability['basis']})
- risk_level:        {risk_level}

## REQUIRED CAPA PARAMETERS (your CAPA narrative must be consistent with these)
{json.dumps(capa_reqs, indent=2)}

## Available Evidence (ONLY cite IDs that appear here)
Matching Adverse Events:
{json.dumps(matching_events, indent=2)}

Matching Recalls:
{json.dumps(matching_recalls, indent=2)}

Similarity cluster member IDs (also citable): {similar_event_ids}

## Trend Data
- Cluster:  {state.get('cluster_label')} (size: {state.get('cluster_size', 'unknown')})
- Trend:    {state.get('trend_flag')} (30-day growth rate: {state.get('growth_rate_30d')})
{past_context}"""

    # ── LLM call — narrative only ────────────────────────────────────────────
    print(f"\n  [LLM] calling claude-sonnet-4-6 → narrative for fixed risk_level={risk_level} ...")
    user_content = f"Write the risk assessment narrative for:\n{context}"
    turn = [{"role": "user", "content": user_content}]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=_NARRATIVE_SYSTEM,
        messages=turn,
        temperature=0.2,
    )
    raw = response.content[0].text
    narrative = _parse_json_response(raw)
    print(f"  [LLM] done. proposed {len(narrative.get('evidence_basis', []))} citation(s)")

    # ── TOOL 6: validate citations — strip anything hallucinated ────────────
    cited_ids = [e.get("id") for e in narrative.get("evidence_basis", []) if e.get("id")]
    validated = validate_evidence_citations(
        cited_ids=cited_ids,
        matching_events=matching_events,
        matching_recalls=matching_recalls,
        similar_event_ids=similar_event_ids,
    )
    evidence_basis = [
        e for e in narrative.get("evidence_basis", [])
        if e.get("id") in validated["valid_ids"]
    ]
    print(f"  [TOOL] validate_evidence_citations() = {validated['valid_count']} valid"
          f"{', ' + str(len(validated['invalid_ids'])) + ' stripped (hallucinated)' if validated['invalid_ids'] else ''}")

    # ── TOOL 7: escalation flags — finalized now that evidence_count is known ─
    flags = compute_escalation_flags(risk_level, evidence_count=len(evidence_basis))
    print(f"  [TOOL] compute_escalation_flags({risk_level}, evidence_count={len(evidence_basis)}) = {flags}")

    all_messages = (state.get("messages") or []) + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": raw},
    ]

    print(f"\n  → writing to state:")
    print(f"      {'severity_level':<28} = {severity['severity_level']}  (TOOL)")
    print(f"      {'probability_level':<28} = {probability['probability_level']}  (TOOL)")
    print(f"      {'risk_level':<28} = {risk_level}  (TOOL)")
    print(f"      {'hazardous_situation':<28} = {(narrative.get('hazardous_situation') or '')[:60]}")
    print(f"      {'harm':<28} = {(narrative.get('harm') or '')[:60]}")
    print(f"      {'evidence_basis':<28} = {len(evidence_basis)} citation(s)")
    print(f"      {'escalation_required':<28} = {flags['escalation_required']}  (TOOL)")
    print(f"      {'prrc_notification_required':<28} = {flags['prrc_notification_required']}  (TOOL)")
    print(f"      {'gate3_passed':<28} = {flags['gate3_passed']}  (TOOL)")
    print(f"      {'capa_immediate':<28} = {(narrative.get('capa_immediate') or '')[:60]}")

    return {
        "severity_level":             severity["severity_level"],
        "probability_level":          probability["probability_level"],
        "risk_level":                 risk_level,
        "hazardous_situation":        narrative.get("hazardous_situation"),
        "harm":                       narrative.get("harm"),
        "severity_rationale":         narrative.get("severity_rationale"),
        "probability_rationale":      narrative.get("probability_rationale"),
        "evidence_basis":             evidence_basis,
        "uncertainty":                narrative.get("uncertainty"),
        "capa_immediate":             narrative.get("capa_immediate"),
        "capa_investigation":         narrative.get("capa_investigation"),
        "capa_corrective":            narrative.get("capa_corrective"),
        "capa_preventive":            narrative.get("capa_preventive"),
        "capa_verification":          narrative.get("capa_verification"),
        "capa_effectiveness":         narrative.get("capa_effectiveness"),
        "capa_precedent":             narrative.get("capa_precedent"),
        "escalation_required":        flags["escalation_required"],
        "prrc_notification_required": flags["prrc_notification_required"],
        "fsca_required":              flags["fsca_required"],
        "gate3_passed":               flags["gate3_passed"],
        "messages":                   all_messages,
    }
