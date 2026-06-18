"""
Deterministic tools for the Risk Analysis Agent.

Severity, probability, risk_level, CAPA timing, and escalation flags are
computed HERE — by rule tables, lookups, and arithmetic — never generated
freehand by an LLM. The same complaint + same evidence always produces the
same numbers. Reference data lives in reference/*.json so the rules can be
recalibrated without touching code.

The LLM (in agents/risk_analysis.py) only writes the narrative that explains
a result these tools already fixed.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REF_DIR = Path(__file__).parent.parent / "reference"

_SEVERITY_LEVELS = ["S1", "S2", "S3", "S4", "S5"]        # ascending
_PROBABILITY_LEVELS = ["P1", "P2", "P3", "P4", "P5"]      # ascending


def _load_json(filename: str) -> dict:
    path = _REF_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Tool 1: Severity scoring ───────────────────────────────────────────────

def score_severity(complaint_text: str) -> dict:
    """
    Deterministic keyword classifier against reference/severity_keywords.json.
    Checks S5 down to S1 (highest first); returns the HIGHEST level with at
    least one matched phrase. Same text always returns the same result.

    If no level matches, defaults to S2 with default_applied=True — a
    conservative, visible fallback rather than a silent guess.
    """
    keywords = _load_json("severity_keywords.json")
    text_lower = (complaint_text or "").lower()

    for level in reversed(_SEVERITY_LEVELS):  # S5 -> S1
        phrases = [p for p in keywords.get(level, []) if not p.startswith("_")]
        matched = [p for p in phrases if p.lower() in text_lower]
        if matched:
            return {
                "severity_level": level,
                "matched_phrases": matched,
                "rule_version": "severity_keywords.json",
                "default_applied": False,
            }

    default_level = "S2"
    logger.warning(
        "score_severity: no keyword match in complaint text — defaulting to %s "
        "(flagged for human review)", default_level,
    )
    return {
        "severity_level": default_level,
        "matched_phrases": [],
        "rule_version": "severity_keywords.json",
        "default_applied": True,
    }


# ── Tool 2: Probability computation ────────────────────────────────────────

def compute_probability(
    event_count: int,
    recall_count: int,
    growth_rate_30d: float | None,
    cluster_size: int | None,
) -> dict:
    """
    Deterministic numeric threshold table against
    reference/probability_thresholds.json. Same counts always return the
    same probability level.
    """
    cfg = _load_json("probability_thresholds.json")
    breakpoints = cfg["event_count_breakpoints"]
    recall_rule = cfg["recall_override"]
    growth_rule = cfg["growth_rate_overrides"]
    cluster_rule = cfg["cluster_size_overrides"]

    event_count = event_count or 0
    recall_count = recall_count or 0
    growth_rate_30d = growth_rate_30d if growth_rate_30d is not None else 0.0
    cluster_size = cluster_size or 0

    if event_count <= breakpoints["P1"]:
        level = "P1"
    elif event_count <= breakpoints["P2"]:
        level = "P2"
    elif event_count <= breakpoints["P3"]:
        level = "P3"
    elif event_count <= breakpoints["P4"]:
        level = "P4"
    else:
        level = "P5"

    basis = [f"event_count={event_count} -> base level {level}"]

    def _idx(lvl: str) -> int:
        return _PROBABILITY_LEVELS.index(lvl)

    if recall_count >= recall_rule["min_recall_count_for_P3"] and _idx(level) < _idx("P3"):
        level = "P3"
        basis.append(f"recall_count={recall_count} >= {recall_rule['min_recall_count_for_P3']} -> bumped to P3")

    if cluster_size >= cluster_rule["P5_min_cluster_size"] and _idx(level) < _idx("P5"):
        level = "P5"
        basis.append(f"cluster_size={cluster_size} >= {cluster_rule['P5_min_cluster_size']} -> bumped to P5")
    elif cluster_size >= cluster_rule["P4_min_cluster_size"] and _idx(level) < _idx("P4"):
        level = "P4"
        basis.append(f"cluster_size={cluster_size} >= {cluster_rule['P4_min_cluster_size']} -> bumped to P4")

    if growth_rate_30d >= growth_rule["P5_min_growth_rate"] and _idx(level) < _idx("P5"):
        level = "P5"
        basis.append(f"growth_rate_30d={growth_rate_30d} >= {growth_rule['P5_min_growth_rate']} -> bumped to P5")
    elif growth_rate_30d >= growth_rule["P4_min_growth_rate"] and _idx(level) < _idx("P4"):
        level = "P4"
        basis.append(f"growth_rate_30d={growth_rate_30d} >= {growth_rule['P4_min_growth_rate']} -> bumped to P4")

    return {
        "probability_level": level,
        "basis": "; ".join(basis),
        "rule_version": "probability_thresholds.json",
    }


# ── Tool 3: Risk matrix lookup ─────────────────────────────────────────────

def apply_risk_matrix(severity_level: str, probability_level: str) -> str:
    """Pure deterministic lookup against reference/risk_matrix.json. No LLM ever touches this."""
    matrix = _load_json("risk_matrix.json")
    key = f"{severity_level}_{probability_level}"
    if key not in matrix:
        raise ValueError(f"No risk matrix entry for {key!r}")
    return matrix[key]


# ── Tool 4: Citation validator ─────────────────────────────────────────────

def validate_evidence_citations(
    cited_ids: list,
    matching_events: list,
    matching_recalls: list,
    similar_event_ids: list,
) -> dict:
    """
    Cross-checks every ID the LLM cited against IDs that actually exist in
    state. Strips anything hallucinated. This is what makes the
    constitutional guardrail enforceable in code, not just in the prompt.
    """
    real_ids = set()
    for e in matching_events or []:
        if e.get("report_number"):
            real_ids.add(e["report_number"])
    for r in matching_recalls or []:
        if r.get("recall_id"):
            real_ids.add(r["recall_id"])
    for sid in similar_event_ids or []:
        if sid:
            real_ids.add(sid)

    valid_ids = [cid for cid in cited_ids if cid in real_ids]
    invalid_ids = [cid for cid in cited_ids if cid not in real_ids]

    if invalid_ids:
        logger.warning("validate_evidence_citations: stripped hallucinated IDs: %s", invalid_ids)

    return {
        "valid_ids": valid_ids,
        "invalid_ids": invalid_ids,
        "valid_count": len(valid_ids),
    }


# ── Tool 5: CAPA requirements lookup ───────────────────────────────────────

def lookup_capa_requirements(risk_level: str) -> dict:
    """Deterministic risk_level -> CAPA timing/notification requirements."""
    requirements = _load_json("capa_requirements.json")
    if risk_level not in requirements:
        raise ValueError(f"No CAPA requirements defined for risk_level={risk_level!r}")
    return requirements[risk_level]


# ── Tool 6: Escalation flags ───────────────────────────────────────────────

def compute_escalation_flags(risk_level: str, evidence_count: int) -> dict:
    """Deterministic, no LLM involvement."""
    escalation_required = risk_level in ("ALARP", "UNACCEPTABLE")
    prrc_notification = risk_level == "UNACCEPTABLE"
    gate3_passed = not (risk_level == "UNACCEPTABLE" and evidence_count == 0)
    return {
        "escalation_required": escalation_required,
        "prrc_notification_required": prrc_notification,
        # confirmed root cause + active distribution — always a human decision, never automated
        "fsca_required": False,
        "gate3_passed": gate3_passed,
    }
