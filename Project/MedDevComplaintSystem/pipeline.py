"""
Medical Device Complaint Pipeline — Working skeleton.

Risk Analysis Agent: fully implemented (ISO 14971, LLM, guardrails).
All other agents: mock stubs returning hardcoded realistic data.

Usage:
    python pipeline.py
    python pipeline.py --complaint "Your complaint text here"

Requires: OPENAI_API_KEY in environment or .env file.
"""

import os
import sys
import uuid
import argparse
from typing import TypedDict, Optional

import logging

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agents.risk_analysis import risk_analysis_agent  # noqa: F401 (re-exported for graph)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── Shared State ──────────────────────────────────────────────────────────────
class ComplaintState(TypedDict):
    complaint_text: str
    trace_id: str

    # Written by extraction_agent
    modality: Optional[str]
    manufacturer: Optional[str]
    device_model: Optional[str]
    software_version: Optional[str]
    component: Optional[str]
    failure_mode: Optional[str]
    severity_indicator: Optional[str]
    qms_complaint_category: Optional[str]
    software_related: Optional[bool]
    is_safety_related: Optional[bool]
    confidence: Optional[float]
    gate1_passed: Optional[bool]

    # Written by similarity_module
    cluster_id: Optional[int]
    cluster_label: Optional[str]
    cluster_size: Optional[int]
    trend_flag: Optional[str]
    growth_rate_30d: Optional[float]
    similar_event_ids: Optional[list]

    # Written by retrieval_agent
    matching_events: Optional[list]
    matching_recalls: Optional[list]
    regulatory_context: Optional[str]
    low_confidence_retrieval: Optional[bool]

    # Written by risk_analysis_agent
    hazardous_situation: Optional[str]
    harm: Optional[str]
    severity_level: Optional[str]
    severity_rationale: Optional[str]
    probability_level: Optional[str]
    probability_rationale: Optional[str]
    risk_level: Optional[str]
    evidence_basis: Optional[list]
    uncertainty: Optional[str]
    capa_immediate: Optional[str]
    capa_investigation: Optional[str]
    capa_corrective: Optional[str]
    capa_preventive: Optional[str]
    capa_verification: Optional[str]
    capa_effectiveness: Optional[str]
    capa_precedent: Optional[str]
    escalation_required: Optional[bool]
    prrc_notification_required: Optional[bool]
    fsca_required: Optional[bool]
    gate3_passed: Optional[bool]

    # Written by report_agent
    document_id: Optional[str]
    report_markdown: Optional[str]
    approval_status: Optional[str]

    # Pipeline control
    escalation_notice: Optional[str]
    messages: list


# ── Debug helpers ─────────────────────────────────────────────────────────────

_W = 64

def _hdr(name: str) -> None:
    print(f"\n{'─' * _W}")
    print(f"  {name}")
    print(f"{'─' * _W}")

def _reading(fields: dict) -> None:
    print("  ← reading from state:")
    for k, v in fields.items():
        print(f"      {k:<30} = {v}")

def _writing(fields: dict) -> None:
    print("  → writing to state:")
    for k, v in fields.items():
        val = str(v)
        if len(val) > 80:
            val = val[:77] + "..."
        print(f"      {k:<30} = {val}")


# ── Mock Agents ───────────────────────────────────────────────────────────────
# These return pre-baked realistic data so risk_analysis_agent has something
# to work with. Replace each with a real implementation when ready.

def extraction_agent(state: ComplaintState) -> dict:
    """MOCK — Replace with real LLM extraction when ready."""
    _hdr("EXTRACTION AGENT  [MOCK]")
    _reading({"complaint_text": (state.get("complaint_text") or "")[:80] + "..."})

    out = {
        "modality": "MRI",
        "manufacturer": "Philips Medical Systems",
        "device_model": "Achieva 1.5T",
        "software_version": "5.7.1",
        "component": "Image reconstruction software",
        "failure_mode": "Banding artifacts in SSFP sequence",
        "severity_indicator": "S3",
        "qms_complaint_category": "IMG-QUAL",
        "software_related": True,
        "is_safety_related": False,
        "confidence": 0.92,
        "gate1_passed": True,
    }
    _writing({k: v for k, v in out.items()})
    return out


def retrieval_agent(state: ComplaintState) -> dict:
    """MOCK — Replace with real ChromaDB + openFDA retrieval when ready."""
    _hdr("RETRIEVAL AGENT  [MOCK]")
    _reading({
        "failure_mode": state.get("failure_mode"),
        "modality":     state.get("modality"),
        "manufacturer": state.get("manufacturer"),
    })

    events = [
        {
            "report_number": "MW3021547",
            "relevance_score": 0.87,
            "narrative_snippet": (
                "MRI banding artifacts observed during cardiac SSFP imaging on "
                "Philips Achieva 1.5T. Images non-diagnostic, repeat scan required."
            ),
            "manufacturer": "Philips Medical Systems",
            "product_code": "LNH",
            "date_received": "2024-03-15",
        },
        {
            "report_number": "MW2998341",
            "relevance_score": 0.79,
            "narrative_snippet": (
                "Horizontal banding artifacts in SSFP sequences. Attributed to "
                "RF pulse timing software error. SW version 5.6.2. Diagnostic delay."
            ),
            "manufacturer": "Philips Medical Systems",
            "product_code": "LNH",
            "date_received": "2023-11-08",
        },
        {
            "report_number": "MW3014892",
            "relevance_score": 0.71,
            "narrative_snippet": (
                "Image artifact pattern consistent with gradient timing mismatch. "
                "Affected cardiac cine imaging. Patient rescanned on different unit."
            ),
            "manufacturer": "Philips Medical Systems",
            "product_code": "LNH",
            "date_received": "2024-01-22",
        },
    ]
    recalls = [
        {
            "recall_id": "Z-2024-00423",
            "reason_for_recall": (
                "Software defect causing image reconstruction artifacts in SSFP "
                "cardiac sequences on Philips Achieva MRI systems"
            ),
            "root_cause": "Software design — RF pulse timing calculation error in reconstruction kernel",
            "action": "Software update v5.7.2 released; field correction advisory issued to all sites",
            "manufacturer": "Philips Medical Systems",
            "product_code": "LNH",
            "recall_class": "II",
            "relevance_score": 0.93,
        }
    ]
    regulatory_context = (
        "Three prior MAUDE adverse events (MW3021547, MW2998341, MW3014892) and one Class II "
        "recall (Z-2024-00423) found for the same failure mode (SSFP banding artifacts) on "
        "Philips LNH devices. Root cause: RF pulse timing software defect. Software fix "
        "available as v5.7.2."
    )

    print("  → writing to state:")
    print(f"      {'matching_events':<30} = {len(events)} events " +
          ", ".join(e["report_number"] for e in events))
    print(f"      {'matching_recalls':<30} = {len(recalls)} recall(s): " +
          ", ".join(r["recall_id"] for r in recalls))
    print(f"      {'regulatory_context':<30} = {regulatory_context[:70]}...")
    print(f"      {'low_confidence_retrieval':<30} = False")

    return {
        "matching_events": events,
        "matching_recalls": recalls,
        "regulatory_context": regulatory_context,
        "low_confidence_retrieval": False,
    }


def similarity_module(state: ComplaintState) -> dict:
    """MOCK — Replace with real HDBSCAN + UMAP + temporal analysis when ready."""
    _hdr("SIMILARITY MODULE  [MOCK]")
    _reading({
        "complaint_text": (state.get("complaint_text") or "")[:60] + "...",
        "modality":       state.get("modality"),
    })

    out = {
        "cluster_id": 12,
        "cluster_label": "MRI_SSFP_artifact_cluster",
        "cluster_size": 47,
        "trend_flag": "emerging",
        "growth_rate_30d": 0.31,
        "similar_event_ids": ["MW3021547", "MW2998341", "MW3014892", "MW3019003"],
    }
    _writing(out)
    return out


def report_agent(state: ComplaintState) -> dict:
    """MOCK — Replace with real ISO 13485 document generation when ready."""
    _hdr("REPORT AGENT  [MOCK]")
    _reading({
        "risk_level":                 state.get("risk_level"),
        "severity_level":             state.get("severity_level"),
        "probability_level":          state.get("probability_level"),
        "escalation_required":        state.get("escalation_required"),
        "prrc_notification_required": state.get("prrc_notification_required"),
        "capa_immediate":             (state.get("capa_immediate") or "")[:60],
        "capa_precedent":             state.get("capa_precedent"),
        "evidence_count":             len(state.get("evidence_basis") or []),
        "uncertainty":                (state.get("uncertainty") or "")[:60],
    })

    doc_id = f"SR-2026-{str(uuid.uuid4())[:4].upper()}"
    out = {
        "document_id": doc_id,
        "report_markdown": (
            f"# Signal Report {doc_id}\n\n"
            f"**Risk Level:** {state.get('risk_level', 'N/A')}\n"
            f"**Escalation Required:** {state.get('escalation_required', False)}\n"
            f"**PRRC Notification:** {state.get('prrc_notification_required', False)}\n\n"
            f"*[MOCK] Full report body to be generated by the real report agent.*\n\n"
            f"> AI-assisted draft — requires human review and approval per AIMS policy."
        ),
        "approval_status": "DRAFT",
    }
    _writing({"document_id": doc_id, "approval_status": "DRAFT"})
    return out


# risk_analysis_agent → imported from agents/risk_analysis.py

# ── Gate routers ──────────────────────────────────────────────────────────────

def gate1_router(state: ComplaintState) -> str:
    passed = state.get("gate1_passed", False)
    _hdr("GATE 1  (post-extraction)")
    print(f"  confidence  = {state.get('confidence')}")
    print(f"  failure_mode= {state.get('failure_mode')}")
    if passed:
        print("  ✓ PASS → proceeding to retrieve + cluster")
    else:
        print("  ✗ HALT — confidence too low or missing required fields")
    return "proceed" if passed else "halt"


def gate3_router(state: ComplaintState) -> str:
    passed = state.get("gate3_passed", False)
    _hdr("GATE 3  (post-risk-analysis)")
    print(f"  risk_level      = {state.get('risk_level')}")
    print(f"  evidence_count  = {len(state.get('evidence_basis') or [])}")
    if passed:
        print("  ✓ PASS → proceeding to report")
    else:
        print("  ✗ HALT — UNACCEPTABLE risk with zero evidence citations")
    return "proceed" if passed else "halt"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_complaint_graph():
    graph = StateGraph(ComplaintState)

    graph.add_node("extract",       extraction_agent)
    graph.add_node("retrieve",      retrieval_agent)
    graph.add_node("cluster",       similarity_module)
    graph.add_node("risk_analysis", risk_analysis_agent)
    graph.add_node("report",        report_agent)

    graph.set_entry_point("extract")

    # Gate 1: halt if extraction failed
    graph.add_conditional_edges("extract", gate1_router, {
        "proceed": "retrieve",
        "halt":    END,
    })

    # retrieve → cluster → risk_analysis (sequential for simplicity; parallelize later)
    graph.add_edge("retrieve",      "cluster")
    graph.add_edge("cluster",       "risk_analysis")

    # Gate 3: halt if UNACCEPTABLE risk with no citations
    graph.add_conditional_edges("risk_analysis", gate3_router, {
        "proceed": "report",
        "halt":    END,
    })

    graph.add_edge("report", END)
    return graph.compile()


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_result(result: dict):
    sep = "=" * 64
    print(f"\n{sep}")
    print("PIPELINE RESULT")
    print(sep)

    if result.get("risk_level") is None:
        print("Pipeline halted before risk analysis (gate 1 or gate 3 failed).")
        print(f"gate1_passed: {result.get('gate1_passed')}")
        print(f"gate3_passed: {result.get('gate3_passed')}")
        print(sep)
        return

    print(f"Risk Level          : {result.get('risk_level')}")
    print(f"  Severity          : {result.get('severity_level')} — {result.get('severity_rationale')}")
    print(f"  Probability       : {result.get('probability_level')} — {result.get('probability_rationale')}")
    print(f"  Hazardous Sit.    : {result.get('hazardous_situation')}")
    print(f"  Harm              : {result.get('harm')}")
    print()
    print(f"Escalation Required : {result.get('escalation_required')}")
    print(f"PRRC Notification   : {result.get('prrc_notification_required')}")
    print(f"FSCA Required       : {result.get('fsca_required')}")
    print(f"Gate 3 Passed       : {result.get('gate3_passed')}")
    print()
    print("Evidence Basis:")
    for ev in result.get("evidence_basis", []):
        print(f"  [{ev.get('source')}] {ev.get('id')} — {ev.get('relevance')}")
    print()
    print(f"Uncertainty         : {result.get('uncertainty')}")
    print()
    print("CAPA:")
    print(f"  Immediate         : {result.get('capa_immediate')}")
    print(f"  Investigation     : {result.get('capa_investigation')}")
    print(f"  Corrective        : {result.get('capa_corrective')}")
    print(f"  Preventive        : {result.get('capa_preventive')}")
    print(f"  Verification      : {result.get('capa_verification')}")
    print(f"  Effectiveness     : {result.get('capa_effectiveness')}")
    print(f"  Precedent         : {result.get('capa_precedent')}")
    print()
    print(f"Document ID         : {result.get('document_id')}")
    print(f"Approval Status     : {result.get('approval_status')}")
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

DEFAULT_COMPLAINT = (
    "During routine cardiac MRI on a Philips Achieva 1.5T (SW v5.7.1), "
    "banding artifacts appeared throughout the SSFP sequence. Images were "
    "non-diagnostic. Repeat scan required, extending patient procedure time by "
    "45 minutes. Imaging technologist reported a similar incident last week."
)


def main():
    parser = argparse.ArgumentParser(description="Medical device complaint pipeline (Risk Analysis live, others mocked)")
    parser.add_argument("--complaint", type=str, default=DEFAULT_COMPLAINT, help="Complaint narrative text")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to the .env file in this directory.")
        sys.exit(1)

    app = build_complaint_graph()

    print(f"\nTrace ID: {(tid := str(uuid.uuid4()))}")
    print(f"Complaint: {args.complaint[:120]}{'...' if len(args.complaint) > 120 else ''}\n")

    result = app.invoke({
        "complaint_text": args.complaint,
        "trace_id": tid,
        "messages": [],
    })

    print_result(result)


if __name__ == "__main__":
    main()
