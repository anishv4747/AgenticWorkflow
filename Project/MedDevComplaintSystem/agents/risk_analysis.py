"""
Risk Analysis Agent — autonomous tool-calling ISO 14971:2019 implementation.

Design: severity, probability, risk_level, CAPA timing, and episodic-memory lookup
are computed by DETERMINISTIC TOOLS (agents/risk_tools.py) — never generated
freehand by an LLM. What changed from the v1 design: the LLM now AUTONOMOUSLY
decides which tools to call, in what order, and how many times (a real
tool-calling loop, see PLAN_v2.md), instead of code calling them in a fixed
hardcoded sequence. The LLM still never decides the numbers themselves —
two enforcement checks (presence + value-fidelity, see _dispatch_tool_call)
guarantee severity_level/probability_level/risk_level always come from a real,
correctly-chained tool call, never from LLM invention.

Tool tiers:
    Autonomous, LLM-callable (the agent decides when/whether to call these):
      - score_severity            mandatory, any time
      - compute_probability       mandatory, any time
      - apply_risk_matrix         mandatory, after the two above
      - lookup_capa_requirements  mandatory, after apply_risk_matrix
      - load_past_reports         optional, zero or more calls, adaptive params
      - submit_risk_narrative     finalize — terminates the loop
    Code-only, mandatory, never LLM-callable (constitutional guardrails):
      - validate_evidence_citations  strips hallucinated citation IDs
      - compute_escalation_flags     escalation_required / prrc / fsca / gate3

Loop safety: capped at MAX_ITERS iterations / TIME_BUDGET_S wall-clock. If the
loop exceeds the cap without the LLM finalizing, a deterministic fallback
computes severity/probability/risk_level/CAPA directly (bypassing the LLM) so
these fields are NEVER missing — only the narrative prose is left null and
flagged in `uncertainty` for human authoring.

Episodic memory:
    load_past_reports() queries SQLite signal_reports table for prior similar cases.
    The table is created by init_db() on first use (idempotent).
    save_report() is called by the report agent, not here — this module only reads.

See PLAN_v2.md for the full design rationale (autonomy decisions, enforcement
mechanism, revised reproducibility target).
"""

import json
import time
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

MAX_ITERS = 6
TIME_BUDGET_S = 30
MANDATORY_TOOLS = {"score_severity", "compute_probability", "apply_risk_matrix", "lookup_capa_requirements"}


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


# ── Tool schemas exposed to the LLM ────────────────────────────────────────────
# These live here (not in risk_tools.py) because a JSON Schema is an LLM-interface
# concern, not business logic. risk_tools.py stays pure and untouched.

_TOOLS = [
    {
        "name": "score_severity",
        "description": (
            "Classify the severity of patient harm described in the complaint text, using "
            "a deterministic keyword rule table (S1=negligible .. S5=catastrophic/death). "
            "Mandatory — you must call this before you can finalize the assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "complaint_text": {"type": "string", "description": "The complaint narrative to classify."},
            },
            "required": ["complaint_text"],
        },
    },
    {
        "name": "compute_probability",
        "description": (
            "Compute the probability-of-occurrence level (P1=rare .. P5=frequent) from "
            "deterministic thresholds on event count, recall count, 30-day growth rate, and "
            "cluster size. Mandatory — you must call this before you can finalize the assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_count": {"type": "integer", "description": "Number of matching adverse events in evidence."},
                "recall_count": {"type": "integer", "description": "Number of matching recalls in evidence."},
                "growth_rate_30d": {"type": ["number", "null"], "description": "30-day growth rate of this cluster, if known."},
                "cluster_size": {"type": ["integer", "null"], "description": "Size of the similarity cluster, if known."},
            },
            "required": ["event_count", "recall_count"],
        },
    },
    {
        "name": "apply_risk_matrix",
        "description": (
            "Look up the final ISO 14971 risk_level (ACCEPTABLE | ALARP | UNACCEPTABLE) from "
            "the 5x5 severity x probability matrix. You MUST pass the exact severity_level and "
            "probability_level values that score_severity and compute_probability returned in "
            "this conversation — do not guess or invent them. Mandatory, call after both of those."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "severity_level": {"type": "string", "enum": ["S1", "S2", "S3", "S4", "S5"]},
                "probability_level": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5"]},
            },
            "required": ["severity_level", "probability_level"],
        },
    },
    {
        "name": "lookup_capa_requirements",
        "description": (
            "Look up the mandatory CAPA timing and notification requirements (containment "
            "timeline in hours, PRRC notification / investigation requirements) for a given "
            "risk_level. Use the exact risk_level apply_risk_matrix returned. Mandatory, call "
            "after apply_risk_matrix — your CAPA narrative must be consistent with this result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "risk_level": {"type": "string", "enum": ["ACCEPTABLE", "ALARP", "UNACCEPTABLE"]},
            },
            "required": ["risk_level"],
        },
    },
    {
        "name": "load_past_reports",
        "description": (
            "Search episodic memory (past completed signal reports) for prior similar cases. "
            "Optional — call zero or more times. Retry with a broader modality, a shorter or "
            "different failure_mode keyword, or a higher limit if your first query returns "
            "nothing. Useful context for rationale/capa_precedent/uncertainty, but never "
            "required and never overrides the deterministic classification tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "failure_mode": {"type": "string", "description": "Failure mode keyword(s) to match (substring match)."},
                "modality": {"type": "string", "description": "Device modality to match, e.g. 'MRI', 'CT'."},
                "limit": {"type": "integer", "description": "Max records to return. Default 5.", "minimum": 1, "maximum": 20},
            },
            "required": ["failure_mode", "modality"],
        },
    },
    {
        "name": "submit_risk_narrative",
        "description": (
            "FINAL STEP. Submit the completed narrative for this ISO 14971 risk assessment. "
            "Call this exactly once, only after you have called score_severity, "
            "compute_probability, apply_risk_matrix, and lookup_capa_requirements in this "
            "conversation. If a required tool call is missing or stale, this call will be "
            "rejected with an explanation — call the missing/stale tool(s) and retry. Do NOT "
            "include severity_level, probability_level, or risk_level here — those are not "
            "yours to set; this tool only carries the narrative prose."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hazardous_situation": {"type": "string"},
                "harm": {"type": "string"},
                "severity_rationale": {"type": "string"},
                "probability_rationale": {"type": "string"},
                "evidence_basis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "enum": ["MAUDE", "RECALL"]},
                            "id": {"type": "string", "description": "Must literally appear in Available Evidence."},
                            "relevance": {"type": "string"},
                        },
                        "required": ["source", "id", "relevance"],
                    },
                },
                "uncertainty": {"type": "string"},
                "capa_immediate": {"type": "string"},
                "capa_investigation": {"type": "string"},
                "capa_corrective": {"type": "string"},
                "capa_preventive": {"type": "string"},
                "capa_verification": {"type": "string"},
                "capa_effectiveness": {"type": "string"},
                "capa_precedent": {"type": ["string", "null"]},
            },
            "required": [
                "hazardous_situation", "harm", "severity_rationale", "probability_rationale",
                "evidence_basis", "uncertainty", "capa_immediate", "capa_investigation",
                "capa_corrective", "capa_preventive", "capa_verification", "capa_effectiveness",
            ],
        },
    },
]


# ── Agent system prompt ─────────────────────────────────────────────────────────

_AGENT_SYSTEM = """\
You are a medical device risk analyst agent performing an ISO 14971:2019 risk assessment.
You own the full workflow for this case, via tools — you are not just writing prose.

CRITICAL — numbers are tool-computed, never yours to invent:
- severity_level, probability_level, risk_level, and CAPA timing/notification requirements
  come ONLY from calling score_severity, compute_probability, apply_risk_matrix, and
  lookup_capa_requirements. You must call all four before you can finalize. Use the EXACT
  values each tool returns when calling the next one in the chain — never guess or alter them.
- score_severity and compute_probability do not depend on each other's output, and
  load_past_reports doesn't depend on anything — you may call all three together in your
  first turn. apply_risk_matrix needs the outputs of score_severity + compute_probability;
  lookup_capa_requirements needs apply_risk_matrix's output — these two are sequential.

Episodic memory (load_past_reports) is OPTIONAL and adaptive: call it zero, one, or several
times. If your first query returns no results, consider retrying with a broader modality or a
shorter/different failure_mode keyword before giving up. It only informs your narrative — it
never changes the classification.

Your narrative job, once the numbers are fixed:
1. hazardous_situation / harm — the specific failure -> exposure -> harm pathway, grounded in
   the complaint text.
2. severity_rationale / probability_rationale — explain, in plain language, why the GIVEN
   severity_level and probability_level fit this complaint and evidence.
3. evidence_basis — ONLY cite MAUDE report numbers or recall IDs that literally appear in the
   "Available Evidence" section below. Never invent IDs. If none apply, return an empty list
   and say so in uncertainty.
4. CAPA narrative (capa_immediate/investigation/corrective/preventive/verification/effectiveness)
   consistent with the REQUIRED CAPA PARAMETERS lookup_capa_requirements returns (e.g. if
   containment_timeline_hours=24, capa_immediate must reflect a 24-hour timeline; if
   prrc_notification_required=true, capa_immediate must mention PRRC notification).
5. uncertainty — what is unknown or unconfirmed, and what would change the assessment.

When ready, call submit_risk_narrative exactly once with the complete narrative. Do not include
severity_level, probability_level, or risk_level in that call.
"""


# ── Dispatch helpers ─────────────────────────────────────────────────────────────

def _dispatch_tool_call(block, ctx: dict) -> tuple[dict, bool]:
    """
    Execute one tool_use block. Returns (result_payload, is_error).

    Two enforcement mechanisms live here, not in a separate LLM-visible "checker"
    tool (which the LLM could simply decline to call):
      - value-fidelity: apply_risk_matrix / lookup_capa_requirements reject args
        that don't match the most recently observed upstream tool output.
      - presence + staleness: submit_risk_narrative rejects unless all mandatory
        tools were called AND a re-derivation of apply_risk_matrix from the
        latest severity/probability still matches the recorded risk_level.
    Ground-truth arguments (complaint_text, event_count, recall_count,
    growth_rate_30d, cluster_size) are always taken from state, never trusted
    from the LLM, even though the schema asks the model to supply them — this
    keeps the schema self-documenting while removing an entire class of
    hallucinated-input failures.
    """
    name = block.name
    inp = block.input

    if name == "score_severity":
        result = score_severity(ctx["complaint_text"])
        ctx["called_tools"].add("score_severity")
        ctx["last_severity_level"] = result["severity_level"]
        return result, False

    if name == "compute_probability":
        result = compute_probability(
            event_count=len(ctx["matching_events"]),
            recall_count=len(ctx["matching_recalls"]),
            growth_rate_30d=ctx["state"].get("growth_rate_30d"),
            cluster_size=ctx["state"].get("cluster_size"),
        )
        ctx["called_tools"].add("compute_probability")
        ctx["last_probability_level"] = result["probability_level"]
        return result, False

    if name == "apply_risk_matrix":
        sev = inp.get("severity_level")
        prob = inp.get("probability_level")
        last_sev = ctx["last_severity_level"]
        last_prob = ctx["last_probability_level"]
        if (last_sev and sev != last_sev) or (last_prob and prob != last_prob):
            return {
                "error": (
                    f"apply_risk_matrix was called with severity_level={sev!r}, "
                    f"probability_level={prob!r}, but score_severity/compute_probability "
                    f"most recently returned {last_sev!r}/{last_prob!r} in this conversation. "
                    f"Re-call apply_risk_matrix with those exact values."
                )
            }, True
        try:
            risk_level = apply_risk_matrix(sev, prob)
        except ValueError as exc:
            return {"error": str(exc)}, True
        ctx["called_tools"].add("apply_risk_matrix")
        ctx["last_risk_level"] = risk_level
        return {"risk_level": risk_level}, False

    if name == "lookup_capa_requirements":
        risk_level = inp.get("risk_level")
        last_risk = ctx["last_risk_level"]
        if last_risk and risk_level != last_risk:
            return {
                "error": (
                    f"lookup_capa_requirements was called with risk_level={risk_level!r}, but "
                    f"apply_risk_matrix most recently returned {last_risk!r}. Re-call with that "
                    f"value."
                )
            }, True
        try:
            reqs = lookup_capa_requirements(risk_level)
        except ValueError as exc:
            return {"error": str(exc)}, True
        ctx["called_tools"].add("lookup_capa_requirements")
        ctx["last_capa_reqs"] = reqs
        return reqs, False

    if name == "load_past_reports":
        reports = load_past_reports(
            failure_mode=inp.get("failure_mode") or ctx["failure_mode"],
            modality=inp.get("modality") or ctx["modality"],
            limit=inp.get("limit", 5),
        )
        ctx["past_reports_seen"].extend(reports)
        return {"count": len(reports), "reports": reports}, False

    if name == "submit_risk_narrative":
        missing = MANDATORY_TOOLS - ctx["called_tools"]
        if missing:
            return {
                "error": (
                    f"Cannot finalize: you have not yet called {sorted(missing)} in this "
                    f"conversation. Call them now, then retry submit_risk_narrative."
                )
            }, True
        try:
            recheck = apply_risk_matrix(ctx["last_severity_level"], ctx["last_probability_level"])
        except ValueError:
            recheck = None
        if recheck != ctx["last_risk_level"]:
            return {
                "error": (
                    f"risk_level is stale: apply_risk_matrix({ctx['last_severity_level']!r}, "
                    f"{ctx['last_probability_level']!r}) would now return {recheck!r}, but the "
                    f"last recorded risk_level is {ctx['last_risk_level']!r}. Re-call "
                    f"apply_risk_matrix (and lookup_capa_requirements) with the current "
                    f"severity/probability levels before finalizing."
                )
            }, True
        ctx["narrative"] = dict(inp)
        return {"accepted": True}, False

    return {"error": f"unknown tool {name!r}"}, True


def _deterministic_fallback(ctx: dict) -> None:
    """
    Loop cap exceeded without a finalize — compute the classification directly,
    bypassing the LLM, so severity/probability/risk_level/CAPA are NEVER missing.
    Only narrative prose is left null, flagged for human authoring.
    """
    logger.warning(
        "risk_analysis_agent: tool-call loop exceeded cap (iters/time budget) — "
        "falling back to deterministic completion"
    )
    if ctx["last_severity_level"] is None:
        ctx["last_severity_level"] = score_severity(ctx["complaint_text"])["severity_level"]
    if ctx["last_probability_level"] is None:
        prob = compute_probability(
            len(ctx["matching_events"]), len(ctx["matching_recalls"]),
            ctx["state"].get("growth_rate_30d"), ctx["state"].get("cluster_size"),
        )
        ctx["last_probability_level"] = prob["probability_level"]
    if ctx["last_risk_level"] is None:
        ctx["last_risk_level"] = apply_risk_matrix(ctx["last_severity_level"], ctx["last_probability_level"])
    if ctx["last_capa_reqs"] is None:
        ctx["last_capa_reqs"] = lookup_capa_requirements(ctx["last_risk_level"])

    ctx["narrative"] = {
        "hazardous_situation": None,
        "harm": None,
        "severity_rationale": "AUTO-FALLBACK: narrative generation did not complete within the tool-call budget; severity/probability/risk were computed deterministically and are reliable.",
        "probability_rationale": "AUTO-FALLBACK: see severity_rationale.",
        "evidence_basis": [],
        "uncertainty": "Narrative incomplete — risk analysis loop exceeded its iteration/time cap. The numeric classification (severity_level/probability_level/risk_level) is reliable because it was tool-computed; narrative fields require human authoring.",
        "capa_immediate": None,
        "capa_investigation": None,
        "capa_corrective": None,
        "capa_preventive": None,
        "capa_verification": None,
        "capa_effectiveness": None,
        "capa_precedent": None,
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

def risk_analysis_agent(state: dict) -> dict:
    """
    Autonomous tool-calling ISO 14971 risk assessment with episodic memory.

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
    print("  RISK ANALYSIS AGENT  [autonomous tool-calling | claude-sonnet-4-6]")
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

    ctx = {
        "state": state,
        "complaint_text": state["complaint_text"],
        "failure_mode": failure_mode,
        "modality": modality,
        "matching_events": matching_events,
        "matching_recalls": matching_recalls,
        "called_tools": set(),
        "last_severity_level": None,
        "last_probability_level": None,
        "last_risk_level": None,
        "last_capa_reqs": None,
        "past_reports_seen": [],
        "narrative": None,
    }

    user_content = f"""\
## Complaint
{state['complaint_text']}

## Extracted Fields
- Modality:            {state.get('modality')}
- Device:              {state.get('device_model')} by {state.get('manufacturer')}
- Software Version:    {state.get('software_version')}
- Component:           {state.get('component')}
- Failure Mode:        {state.get('failure_mode')}
- QMS Category:        {state.get('qms_complaint_category')}

## Available Evidence (ONLY cite IDs that appear here)
Matching Adverse Events:
{json.dumps(matching_events, indent=2)}

Matching Recalls:
{json.dumps(matching_recalls, indent=2)}

Similarity cluster member IDs (also citable): {similar_event_ids}

## Trend Data
- Cluster:  {state.get('cluster_label')} (size: {state.get('cluster_size', 'unknown')})
- Trend:    {state.get('trend_flag')} (30-day growth rate: {state.get('growth_rate_30d')})

## Your task
Classify this complaint's severity and probability using your tools, determine risk_level,
look up CAPA requirements, optionally check episodic memory for similar past cases, then
submit the narrative via submit_risk_narrative.
"""

    messages = [{"role": "user", "content": user_content}]

    print(f"\n  [LOOP] starting tool-calling loop (cap {MAX_ITERS} iters / {TIME_BUDGET_S}s)...")
    start_t = time.monotonic()
    iters = 0
    cap_exceeded = False

    while True:
        iters += 1
        elapsed = time.monotonic() - start_t
        if iters > MAX_ITERS or elapsed > TIME_BUDGET_S:
            cap_exceeded = True
            break

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=_AGENT_SYSTEM,
            tools=_TOOLS,
            messages=messages,
            temperature=0.2,
        )

        if response.stop_reason == "end_turn":
            print(f"  [LOOP] turn {iters}: model stopped without finalizing — nudging")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": (
                    "You have not called submit_risk_narrative yet. If you still need "
                    "score_severity, compute_probability, apply_risk_matrix, or "
                    "lookup_capa_requirements, call them now; otherwise call "
                    "submit_risk_narrative to finish."
                ),
            })
            continue

        if response.stop_reason != "tool_use":
            print(f"  [LOOP] turn {iters}: unexpected stop_reason={response.stop_reason!r} — treating as cap")
            cap_exceeded = True
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue
            result, is_error = _dispatch_tool_call(block, ctx)
            # Log the actual result (ground-truth, post-override), not block.input —
            # several tools override LLM-supplied args with state-derived ground truth
            # (see _dispatch_tool_call), so block.input alone would be misleading here.
            status = "ERROR" if is_error else "ok"
            result_preview = json.dumps(result, default=str)
            if len(result_preview) > 160:
                result_preview = result_preview[:157] + "..."
            print(f"  [TOOL] turn {iters}: {block.name}(requested={block.input}) -> {status} {result_preview}")
            tool_result = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            }
            if is_error:
                tool_result["is_error"] = True
            tool_results.append(tool_result)

        messages.append({"role": "user", "content": tool_results})

        if ctx["narrative"] is not None:
            break

    if cap_exceeded and ctx["narrative"] is None:
        elapsed = time.monotonic() - start_t
        print(f"  [CAP EXCEEDED] iters={iters} elapsed={elapsed:.1f}s — applying deterministic fallback")
        _deterministic_fallback(ctx)

    narrative = ctx["narrative"]
    severity_level = ctx["last_severity_level"]
    probability_level = ctx["last_probability_level"]
    risk_level = ctx["last_risk_level"]

    print(f"\n  [RESULT] severity_level={severity_level} probability_level={probability_level} risk_level={risk_level}")
    print(f"  [RESULT] episodic memory queries made: {len(ctx['past_reports_seen'])} record(s) seen across all load_past_reports calls")

    # ── Code-only mandatory guardrails — never LLM-callable ─────────────────
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

    flags = compute_escalation_flags(risk_level, evidence_count=len(evidence_basis))
    print(f"  [TOOL] compute_escalation_flags({risk_level}, evidence_count={len(evidence_basis)}) = {flags}")

    all_messages = (state.get("messages") or []) + messages

    print(f"\n  → writing to state:")
    print(f"      {'severity_level':<28} = {severity_level}  (TOOL, autonomous)")
    print(f"      {'probability_level':<28} = {probability_level}  (TOOL, autonomous)")
    print(f"      {'risk_level':<28} = {risk_level}  (TOOL, autonomous)")
    print(f"      {'hazardous_situation':<28} = {(narrative.get('hazardous_situation') or '')[:60]}")
    print(f"      {'harm':<28} = {(narrative.get('harm') or '')[:60]}")
    print(f"      {'evidence_basis':<28} = {len(evidence_basis)} citation(s)")
    print(f"      {'escalation_required':<28} = {flags['escalation_required']}  (TOOL)")
    print(f"      {'prrc_notification_required':<28} = {flags['prrc_notification_required']}  (TOOL)")
    print(f"      {'gate3_passed':<28} = {flags['gate3_passed']}  (TOOL)")
    print(f"      {'capa_immediate':<28} = {(narrative.get('capa_immediate') or '')[:60]}")

    return {
        "severity_level":             severity_level,
        "probability_level":          probability_level,
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
