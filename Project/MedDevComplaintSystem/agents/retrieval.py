"""
Retrieval Agent — live openFDA API queries (single-pass RAG).

Scope: ChromaDB and the NetworkX knowledge graph from the original spec require
pre-built indices from downloaded FDA data that don't exist yet. This implementation
covers Source 2 only (live openFDA API) — the fastest path to real evidence.

Strategy:
    1. Query openFDA by manufacturer only (exact-field AND queries on generic_name
       reliably 404 — openFDA's fielded search wants exact catalog values we don't
       have). Pull a wider candidate set (limit=25).
    2. Score each candidate locally by keyword overlap between
       (modality + failure_mode) and the record's narrative/reason text —
       a deterministic stand-in for embedding similarity.
    3. Filter < 0.3, sort by relevance, return top 5 per category.

Resilience: HTTP 429 → exponential backoff (3 retries). Any persistent failure
(network, 404 "no matches", malformed response) degrades to an empty list —
this agent never crashes the pipeline.
"""

import re
import time
import logging
import requests

logger = logging.getLogger(__name__)

_EVENTS_URL = "https://api.fda.gov/device/event.json"
_RECALLS_URL = "https://api.fda.gov/device/recall.json"
_RELEVANCE_THRESHOLD = 0.3
_CANDIDATE_LIMIT = 25
_TOP_K = 5

_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "of", "to", "and", "or", "with",
    "is", "was", "were", "during", "for", "by", "this", "that",
}


def _keywords(text: str) -> set:
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _relevance_score(query_terms: set, candidate_text: str) -> float:
    """Keyword overlap ratio 0.0-1.0 — deterministic substitute for embedding similarity."""
    if not query_terms:
        return 0.0
    candidate_terms = _keywords(candidate_text)
    overlap = len(query_terms & candidate_terms)
    return round(min(1.0, overlap / len(query_terms)), 2)


def _get_with_retry(url: str, params: dict, max_retries: int = 3) -> dict | None:
    """GET with exponential backoff on HTTP 429. Returns None on persistent failure."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("openFDA 429 — retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                # openFDA returns 404 for "no matches" — not an error, just empty.
                return {"results": []}
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("openFDA request failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def _query_events(manufacturer_term: str) -> list[dict]:
    if not manufacturer_term:
        return []
    params = {
        "search": f'device.manufacturer_d_name:"{manufacturer_term}"',
        "limit": _CANDIDATE_LIMIT,
    }
    data = _get_with_retry(_EVENTS_URL, params)
    if not data:
        return []
    return data.get("results", [])


def _query_recalls(manufacturer_term: str) -> list[dict]:
    if not manufacturer_term:
        return []
    params = {
        "search": f'recalling_firm:"{manufacturer_term}"',
        "limit": _CANDIDATE_LIMIT,
    }
    data = _get_with_retry(_RECALLS_URL, params)
    if not data:
        return []
    return data.get("results", [])


def _score_events(raw_events: list[dict], query_terms: set) -> list[dict]:
    scored = []
    for r in raw_events:
        device = (r.get("device") or [{}])[0] or {}
        narrative = " ".join(
            mdr.get("text", "") for mdr in (r.get("mdr_text") or [])
        )
        problems = " ".join(r.get("product_problems") or [])
        candidate_text = f"{device.get('generic_name', '')} {narrative} {problems}"

        score = _relevance_score(query_terms, candidate_text)
        if score < _RELEVANCE_THRESHOLD:
            continue

        scored.append({
            "report_number": r.get("report_number") or r.get("mdr_report_key", "UNKNOWN"),
            "relevance_score": score,
            "narrative_snippet": narrative[:300].strip() or "(no narrative text)",
            "manufacturer": device.get("manufacturer_d_name", ""),
            "product_code": device.get("device_report_product_code", ""),
            "date_received": r.get("date_received", ""),
        })
    scored.sort(key=lambda e: -e["relevance_score"])
    return scored[:_TOP_K]


def _score_recalls(raw_recalls: list[dict], query_terms: set) -> list[dict]:
    scored = []
    for r in raw_recalls:
        reason = r.get("reason_for_recall", "")
        root_cause = r.get("root_cause_description", "")
        description = r.get("product_description", "")
        candidate_text = f"{reason} {root_cause} {description}"

        score = _relevance_score(query_terms, candidate_text)
        if score < _RELEVANCE_THRESHOLD:
            continue

        scored.append({
            "recall_id": r.get("product_res_number") or r.get("res_event_number", "UNKNOWN"),
            "reason_for_recall": reason,
            "root_cause": root_cause,
            "action": r.get("action", ""),
            "manufacturer": r.get("recalling_firm", ""),
            "product_code": r.get("product_code", ""),
            "recall_class": (r.get("openfda") or {}).get("device_class", "N/A"),
            "relevance_score": score,
        })
    scored.sort(key=lambda e: -e["relevance_score"])
    return scored[:_TOP_K]


def retrieval_agent(state: dict) -> dict:
    """
    Live single-pass RAG via openFDA API.

    Reads (written by extraction_agent): manufacturer, modality, failure_mode

    Writes: matching_events, matching_recalls, regulatory_context, low_confidence_retrieval
    """
    manufacturer = state.get("manufacturer") or ""
    modality = state.get("modality") or ""
    failure_mode = state.get("failure_mode") or ""

    # openFDA's manufacturer field wants the firm's primary name, not the full
    # legal entity string — using the first token ("Philips" from "Philips
    # Medical Systems") matches far more reliably than the full string.
    manufacturer_term = manufacturer.split()[0] if manufacturer else ""

    print(f"  ← reading from state:")
    print(f"      manufacturer = {manufacturer}  (search term: '{manufacturer_term}')")
    print(f"      modality     = {modality}")
    print(f"      failure_mode = {failure_mode}")

    print(f"  → querying openFDA device/event.json ...")
    raw_events = _query_events(manufacturer_term)
    print(f"      {len(raw_events)} candidate event(s) returned")

    print(f"  → querying openFDA device/recall.json ...")
    raw_recalls = _query_recalls(manufacturer_term)
    print(f"      {len(raw_recalls)} candidate recall(s) returned")

    query_terms = _keywords(modality) | _keywords(failure_mode)
    events = _score_events(raw_events, query_terms)
    recalls = _score_recalls(raw_recalls, query_terms)

    low_confidence = (len(events) == 0 and len(recalls) == 0)

    if events or recalls:
        context = (
            f"Found {len(events)} matching adverse event(s) and {len(recalls)} matching "
            f"recall(s) for {manufacturer or 'unknown manufacturer'} devices related to "
            f"'{failure_mode or modality}' (relevance >= {_RELEVANCE_THRESHOLD})."
        )
    else:
        context = (
            f"No FDA adverse events or recalls above relevance threshold "
            f"({_RELEVANCE_THRESHOLD}) found for {manufacturer or 'unknown manufacturer'} "
            f"devices related to '{failure_mode or modality}'."
        )

    print(f"  → writing to state:")
    print(f"      matching_events            = {len(events)} (post-filter, top-{_TOP_K})")
    for e in events:
        print(f"          [{e['relevance_score']}] {e['report_number']} — {e['narrative_snippet'][:50]}")
    print(f"      matching_recalls           = {len(recalls)} (post-filter, top-{_TOP_K})")
    for r in recalls:
        print(f"          [{r['relevance_score']}] {r['recall_id']} — {r['reason_for_recall'][:50]}")
    print(f"      low_confidence_retrieval   = {low_confidence}")

    return {
        "matching_events": events,
        "matching_recalls": recalls,
        "regulatory_context": context,
        "low_confidence_retrieval": low_confidence,
    }
