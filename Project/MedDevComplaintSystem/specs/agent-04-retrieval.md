# Agent 4: Retrieval Agent

| Field | Value |
|---|---|
| Epic | FDA Evidence Retrieval |
| Owner | M2 |
| Autonomy Level | L1 single-pass RAG → upgrade to L2 ReAct only if Precision@5 < 0.65 on gold set |
| Priority | P0 |
| Estimate | 4 days |
| Status | Not started |
| Regulatory Basis | IEC 62304 §9.2 (investigate problem); ISO 14971 §10 (post-production information) |

## User Story

As a **Risk Analyst (Agent 2)**, I need a ranked list of comparable FDA adverse events and recall records for the device and failure mode described in a complaint, so that my risk assessment and CAPA recommendations are grounded in real regulatory precedent rather than generated from prior knowledge alone.

## Context

Agent 4 is the evidence backbone of the pipeline. It translates the structured extraction (Agent 4 consumes `ExtractionOutput`) into targeted queries across three sources: a local ChromaDB vector store (offline recall and event data), the live openFDA API, and a NetworkX knowledge graph connecting devices to manufacturers, product codes, and their regulatory history. All results are relevance-scored and summarized before handoff — raw payloads never pass downstream. Agent 4 runs in parallel with the Similarity Module after extraction. Start with single-pass RAG; only add the ReAct loop if the baseline fails the Precision@5 target.

## Scope

- **In:** ChromaDB semantic search, openFDA API queries (adverse events + recalls), knowledge graph traversal, relevance scoring, filtering, top-5 summarization.
- **Out:** Similarity clustering and trend detection (Similarity Module), risk scoring (Agent 2), report formatting (Agent 6), any LLM-based re-ranking.

## Input / Output

**Input:** `ExtractionOutput` (specifically: `device_model`, `manufacturer`, `modality`, `failure_mode`, `qms_complaint_category`, `software_version`)

**Output:** `RetrievalOutput`
```json
{
  "matching_events": [
    {
      "report_number": "string — MAUDE MDR number",
      "relevance_score": "float 0.0–1.0",
      "narrative_snippet": "string — first 300 chars of mdr_text",
      "event_type": "string",
      "manufacturer": "string",
      "product_code": "string",
      "date_received": "string",
      "date_accessed": "ISO 8601"
    }
  ],
  "matching_recalls": [
    {
      "recall_id": "string — recall number",
      "reason_for_recall": "string",
      "root_cause": "string",
      "action": "string — corrective action taken",
      "manufacturer": "string",
      "product_code": "string",
      "recall_class": "I | II | III",
      "relevance_score": "float 0.0–1.0",
      "date_accessed": "ISO 8601"
    }
  ],
  "regulatory_context": "string — 1–2 sentence summary of retrieved evidence pattern",
  "total_events_available": "int — from openFDA meta.results.total",
  "retrieval_method": "single-pass-rag | react",
  "react_trace": ["string — Thought/Action/Observation steps if ReAct used"],
  "low_confidence_flag": "boolean — true if all relevance scores < 0.3"
}
```

## Three-Source Retrieval Strategy

### Source 1: ChromaDB (Local Vector Store)
- **Collections:** `event_narratives`, `recall_reasons`, `combined`
- **Embedding model:** `all-MiniLM-L6-v2`
- **Query:** Embed the complaint's `failure_mode + symptom` field; cosine similarity search
- **Metadata filters:** `product_code`, `modality`, `manufacturer` (when available)
- **Results:** Top-K candidates (K=10 before filtering)

### Source 2: openFDA API (Live Queries)
- **Adverse events endpoint:** `device/event.json` — filter by `device.device_report_product_code` and `date_received` range
- **Recalls endpoint:** `device/recall.json` — filter by `product_code` and `root_cause_description`
- **Rate limiting:** max 240 requests/minute; retry with exponential backoff (3 retries) on HTTP 429
- **Graceful degradation:** If API fails, continue with ChromaDB local results only; flag as `retrieval_method: local-only`

### Source 3: Knowledge Graph (NetworkX)
- **Graph schema:**
  ```
  Device ──has_code──→ ProductCode (LNH, JAK, LLZ, etc.)
  Device ──made_by───→ Manufacturer
  Manufacturer ──also_makes──→ Device (cross-product linking)
  ProductCode ──has_events──→ AdverseEvent
  ProductCode ──has_recalls──→ Recall
  Recall ──root_cause──→ RootCauseCategory
  ```
- **Traversal:** Starting from `manufacturer` and `product_code` nodes, traverse 2 hops to collect all linked recalls and events
- **Value:** Surfaces same-manufacturer recalls for different product codes (e.g. Philips MRI recall → Philips CT recall for same software platform)

### Hybrid Re-ranking
After collecting candidates from all 3 sources:
1. Score each result: `hybrid_score = 0.6 × vector_sim + 0.4 × graph_proximity`
2. Discard all items with `hybrid_score < 0.3`
3. Deduplicate by recall_id / report_number
4. Return top-5 events + top-5 recalls (summarized, not raw)

## ReAct Upgrade Path (only if Precision@5 < 0.65 on gold set)

If single-pass RAG fails the target, upgrade to iterative retrieval:

```
Thought: "I need more evidence about Philips MRI reconstruction failures"
Action: query_openfda(product_code="LNH", keyword="reconstruction artifact")
Observation: [3 events returned, relevance 0.71–0.85]
Thought: "These are relevant. Also check recalls for same manufacturer"
Action: query_graph(manufacturer="Philips", hop=2, filter="Software design")
Observation: [2 recalls, root_cause="Software design"]
→ STOP (sufficiency condition: ≥ 3 relevant results OR 5 iterations OR 30s timeout)
```

**Hard caps:** 5 iterations maximum; 30s timeout; fallback = best-so-far results.

## Acceptance Criteria

- [ ] Given a complaint with known product code, when ChromaDB is queried, then results include at least one document with matching product code in metadata.
- [ ] Given all retrieved items having relevance < 0.3, when Gate 2 is evaluated, then `low_confidence_flag == true` is set and all items are discarded before handoff to Agent 2.
- [ ] Given 0 results from openFDA API (valid product code, no matching events), when Gate 2 is evaluated, then `matching_events == []` and `regulatory_context` states "No matching FDA adverse events found".
- [ ] Given openFDA returning HTTP 429, when retry logic runs, then agent retries with exponential backoff and falls back to ChromaDB-only results after 3 failures — never crashes.
- [ ] Given Knowledge Graph traversal from a Philips MRI complaint, when graph is traversed 2 hops, then Philips recalls for other modalities (CT, Ultrasound) are included in candidates.
- [ ] Given top-10 raw results, when summarization runs, then only top-5 are returned per category (events, recalls) — not all 10.
- [ ] Given gold relevance judgments, when Precision@5 is computed on 20 test cases, then P@5 > 0.65.

## Technical Approach

1. Embed `failure_mode + " " + symptom + " " + modality` from `ExtractionOutput`; use `all-MiniLM-L6-v2`.
2. Query ChromaDB with embedding + metadata filter (`product_code` if available).
3. Simultaneously query openFDA API for adverse events and recalls (parallel HTTP requests).
4. Traverse NetworkX graph: start from matched `manufacturer` + `product_code` nodes, collect all linked recalls within 2 hops.
5. Merge all candidates; compute hybrid score; filter and deduplicate; return top-5 per category.
6. Summarize each result to 300-char snippet before packaging into `RetrievalOutput` — enforce token budget discipline.
7. If all results below threshold: set `low_confidence_flag = true`, still return (with empty matching lists); Agent 1 Gate 2 handles downstream behavior.
8. Populate `react_trace` only if ReAct upgrade is active.

## Dependencies

- **Blocked by:** `schemas.py`, ChromaDB index (built by M1 from downloaded data), NetworkX knowledge graph (built by M2), openFDA API access
- **Blocks:** Agent 2 (needs RetrievalOutput), Agent 1 Gate 2, ablation study A1 (no-RAG ablation)

## Test Plan

- **Unit:**
  - Schema-valid `RetrievalOutput` from mock ChromaDB with 5 fixtures
  - Relevance filter: inject 3 results with scores 0.1, 0.25, 0.4 → only 0.4 survives
  - HTTP 429 simulation → assert retry + fallback to local-only (no crash)
  - Graph traversal: inject Philips MRI node → assert Philips CT recalls appear in results
  - Top-5 cap: inject 10 results → assert only 5 returned per category
- **Eval:** Precision@5 on 20-example gold set (with relevance judgments from M6)
- **Ablation baseline:** Run without Agent 4 (mock empty retrieval) to measure Agent 2 quality drop (Ablation A1)

## Definition of Done

- [ ] ChromaDB queried and returns schema-valid results on 5 test complaints
- [ ] openFDA API queried and HTTP 429 retry handled gracefully
- [ ] Knowledge graph traversal surfaces cross-product manufacturer recalls
- [ ] All results below 0.3 threshold discarded before handoff
- [ ] Top-5 cap enforced (never more than 5 events + 5 recalls passed downstream)
- [ ] Precision@5 > 0.65 on gold set
- [ ] `low_confidence_flag` triggers Gate 2 warning correctly in orchestrator integration test
