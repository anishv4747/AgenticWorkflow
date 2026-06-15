"""
Ablation Study — All 5 configurations in one script.

Runs the same 10 test complaints through 5 pipeline configs and prints
a comparison table. No pre-downloaded data required — fetches from openFDA.

Configurations:
  A5  baseline       Single LLM call (no RAG, no clustering, no reflection)
  A1  no_rag         Full pipeline minus ChromaDB retrieval
  A2  no_reflection  Full pipeline minus Agent 5 self-critique pass
  A4  no_temporal    Full pipeline minus cluster trend/growth-rate scoring
  full              Full pipeline (all features on)

  NOTE: A3 (no_dpo) requires a DPO-trained model checkpoint.
        Placeholder is included but skipped until checkpoint exists.

Usage:
  pip install openai chromadb sentence-transformers scikit-learn numpy python-dotenv
  Set OPENAI_API_KEY in .env or environment.
  python ablation_study.py
"""

import json
import os
import time
import hashlib
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # use mini for dev, gpt-4.1 for eval
JUDGE_MODEL  = os.getenv("JUDGE_MODEL",  "gpt-4o-mini")
EMBED_MODEL  = "all-MiniLM-L6-v2"
PRODUCT_CODES = ["LNH", "JAK", "LLZ", "IYE"]              # MRI, CT, Ultrasound, X-ray
RECALLS_LIMIT = 200                                         # how many recalls to load into ChromaDB

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embedder = SentenceTransformer(EMBED_MODEL)

# ---------------------------------------------------------------------------
# 10 test complaints (sampled style from real MAUDE narratives)
# ---------------------------------------------------------------------------

TEST_COMPLAINTS = [
    {
        "id": "TC-001",
        "text": (
            "During routine cardiac MRI on Philips Achieva 1.5T, banding artifacts appeared "
            "in steady-state free precession sequences. Images were non-diagnostic. "
            "Radiologist ordered repeat scan. Software version 5.7.1."
        ),
        "gold_modality": "MRI",
        "gold_risk": "ALARP",
        "gold_category": "IMG-QUAL",
    },
    {
        "id": "TC-002",
        "text": (
            "GE Optima CT660 crashed mid-scan and rebooted unexpectedly. "
            "Reconstruction failed for 3 of 12 slices. Patient was re-scanned. "
            "Error log showed null pointer exception in reconstruction thread."
        ),
        "gold_modality": "CT",
        "gold_risk": "ALARP",
        "gold_category": "SW-FUNC",
    },
    {
        "id": "TC-003",
        "text": (
            "Siemens ACUSON ultrasound displayed inverted left/right orientation on "
            "abdominal images. Physician noticed during comparison with prior study. "
            "No patient harm reported but diagnosis delayed by 2 hours."
        ),
        "gold_modality": "Ultrasound",
        "gold_risk": "ALARP",
        "gold_category": "SW-ALGO",
    },
    {
        "id": "TC-004",
        "text": (
            "Hologic Selenia mammography system failed to save DICOM images to PACS. "
            "Three patients required repeat imaging sessions. IT logs show DICOM "
            "network timeout errors. SW version 3.2.4."
        ),
        "gold_modality": "X-ray",
        "gold_risk": "ALARP",
        "gold_category": "SW-DATA",
    },
    {
        "id": "TC-005",
        "text": (
            "Philips Ingenia 3T MRI: patient received mild skin burn on left shoulder "
            "from RF coil during brain MRI. SAR monitor did not trigger safety shutdown. "
            "Incident reported to risk management."
        ),
        "gold_modality": "MRI",
        "gold_risk": "UNACCEPTABLE",
        "gold_category": "SAFE-PAT",
    },
    {
        "id": "TC-006",
        "text": (
            "Beckman Coulter DxH 900 hematology analyzer reported WBC count of 0.2 "
            "on a sample with clinical WBC of 8.4. Lab technician caught error during "
            "delta check. Patient was on chemotherapy — delayed dose adjustment."
        ),
        "gold_modality": "Hematology",
        "gold_risk": "UNACCEPTABLE",
        "gold_category": "PERF-ACC",
    },
    {
        "id": "TC-007",
        "text": (
            "Siemens Biograph PET/CT scanner: attenuation correction algorithm applied "
            "wrong patient body contour after software update v8.1. SUV values off by "
            "15-20%. Two oncology patients had scans re-read."
        ),
        "gold_modality": "CT",
        "gold_risk": "UNACCEPTABLE",
        "gold_category": "SW-ALGO",
    },
    {
        "id": "TC-008",
        "text": (
            "GE Logiq E10 ultrasound: Doppler velocity measurements displaying values "
            "in cm/s instead of m/s after firmware update. No display warning shown. "
            "Cardiologist noticed discrepancy. No adverse outcome."
        ),
        "gold_modality": "Ultrasound",
        "gold_risk": "ALARP",
        "gold_category": "SW-UI",
    },
    {
        "id": "TC-009",
        "text": (
            "Canon Aquilion ONE CT: automatic tube current modulation disabled silently "
            "after system update. Patients received fixed 200mA instead of AEC-optimized "
            "dose. Discovered during quarterly QA check. Estimated 40 patients affected."
        ),
        "gold_modality": "CT",
        "gold_risk": "UNACCEPTABLE",
        "gold_category": "SW-FUNC",
    },
    {
        "id": "TC-010",
        "text": (
            "Philips Azurion fluoroscopy system: IFU states max frame rate 30fps but "
            "software caps at 15fps without user notification. Interventional cardiologist "
            "reports insufficient temporal resolution for TAVI procedure guidance."
        ),
        "gold_modality": "X-ray",
        "gold_risk": "ALARP",
        "gold_category": "DOC-LABEL",
    },
]

# ---------------------------------------------------------------------------
# openFDA recall fetcher (no API key needed)
# ---------------------------------------------------------------------------

def fetch_recalls_from_api(product_codes: list[str], limit: int = RECALLS_LIMIT) -> list[dict]:
    """Fetch recall records from openFDA for given product codes."""
    recalls = []
    per_code = max(1, limit // len(product_codes))
    for pc in product_codes:
        url = (
            f"https://api.fda.gov/device/recall.json"
            f"?search=product_code:{pc}&limit={min(per_code, 100)}"
        )
        try:
            req = Request(url, headers={"User-Agent": "MTech-Ablation/1.0"})
            with urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            if "results" in data:
                recalls.extend(data["results"])
                print(f"  [{pc}] fetched {len(data['results'])} recalls")
        except HTTPError as e:
            print(f"  [{pc}] HTTP {e.code} — skipped")
        except Exception as e:
            print(f"  [{pc}] error: {e} — skipped")
        time.sleep(0.5)
    return recalls

# ---------------------------------------------------------------------------
# ChromaDB recall index (built once, reused across runs)
# ---------------------------------------------------------------------------

def build_recall_index(recalls: list[dict]) -> chromadb.Collection:
    """Embed recall reasons and load into an in-memory ChromaDB collection."""
    chroma = chromadb.Client()
    col = chroma.get_or_create_collection("recalls")
    docs, ids, metas = [], [], []
    for i, r in enumerate(recalls):
        text = " ".join(filter(None, [
            r.get("reason_for_recall", ""),
            r.get("root_cause_description", ""),
            r.get("action", ""),
        ])).strip()
        if len(text) < 20:
            continue
        rid = r.get("recall_number") or r.get("event_id") or f"recall_{i}"
        docs.append(text[:1000])
        ids.append(str(rid))
        metas.append({
            "product_code": r.get("product_code", ""),
            "root_cause": r.get("root_cause_description", ""),
            "action": r.get("action", "")[:300],
        })
    if docs:
        embeddings = embedder.encode(docs).tolist()
        col.add(documents=docs, ids=ids, metadatas=metas, embeddings=embeddings)
    print(f"  ChromaDB index: {len(docs)} recall documents")
    return col

# ---------------------------------------------------------------------------
# Pipeline components (inline — no separate modules)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a medical device quality expert.
Extract the following structured fields from the complaint narrative.
Think step by step before answering (Chain-of-Thought).

COMPLAINT:
<user_narrative>
{narrative}
</user_narrative>

Respond ONLY with valid JSON matching this schema:
{{
  "modality": "<MRI|CT|Ultrasound|X-ray|Hematology|PCR|Other>",
  "component": "<affected component>",
  "failure_mode": "<what failed>",
  "severity_indicator": "<S1_negligible|S2_minor|S3_serious|S4_critical|S5_catastrophic>",
  "software_related": <true|false>,
  "is_safety_related": <true|false>,
  "qms_complaint_category": "<IMG-QUAL|SW-FUNC|SW-ALGO|SW-UI|SW-DATA|PERF-ACC|SAFE-PAT|DOC-LABEL|HW-MECH|HW-ELEC|SW-CYBER>",
  "confidence": <0.0-1.0>,
  "cot_reasoning": "<2-3 sentence reasoning>"
}}"""

RISK_CAPA_PROMPT = """You are a medical device risk manager applying ISO 14971:2019.

COMPLAINT:
<user_narrative>
{narrative}
</user_narrative>

EXTRACTED FIELDS:
{extraction}

FDA EVIDENCE (may be empty):
{evidence}

CLUSTER TREND (may be empty):
{trend}

Using the ISO 14971 5x5 matrix:
- Severity: S1(negligible) S2(minor) S3(serious) S4(critical) S5(catastrophic)
- Probability: P1(incredible) P2(improbable) P3(remote) P4(occasional) P5(frequent)
- Risk level MUST be one of: ACCEPTABLE, ALARP, UNACCEPTABLE

Respond ONLY with valid JSON:
{{
  "hazardous_situation": "<string>",
  "harm": "<string>",
  "severity": {{"level": "S1-S5", "rationale": "<string>"}},
  "probability": {{"level": "P1-P5", "rationale": "<string>"}},
  "risk_level": "<ACCEPTABLE|ALARP|UNACCEPTABLE>",
  "evidence_citations": ["<source: id>"],
  "uncertainty": "<string>",
  "capa": {{
    "immediate_containment": "<string>",
    "root_cause_investigation": "<string>",
    "corrective_action": "<string>",
    "preventive_action": "<string>",
    "verification_method": "<string>",
    "effectiveness_criteria": "<string>",
    "precedent_basis": "<string or none>"
  }},
  "escalation_required": <true|false>
}}"""

SELF_CRITIQUE_PROMPT = """Review this risk assessment for quality issues.
Check:
1. Does every factual claim cite a specific source?
2. Is the risk level consistent with severity × probability?
3. Is the uncertainty field honest about what cannot be confirmed?
4. Are CAPA actions specific and verifiable?

ASSESSMENT:
{assessment}

If ALL checks pass, respond: {{"verdict": "PASS", "revision": null}}
If ANY check fails, respond: {{"verdict": "FAIL", "issues": ["<issue1>"], "revision": {{...corrected assessment...}} }}
"""

JUDGE_PROMPT = """Score this medical device risk assessment on 4 axes. Be critical and consistent.

COMPLAINT:
{narrative}

ASSESSMENT:
{assessment}

Score each axis 1-5:
1. accuracy     — Does it correctly identify the hazard, harm, and risk level?
2. grounding    — Does every factual claim cite a specific FDA record or source?
3. iso_compliance — Does it use ACCEPTABLE/ALARP/UNACCEPTABLE (not HIGH/MEDIUM/LOW)?
4. capa_quality — Are CAPA actions specific, verifiable, and grounded in precedent?

Also count: uncited_claims (integer — claims made without a cited source)

Respond ONLY with valid JSON:
{{
  "accuracy": <1-5>,
  "grounding": <1-5>,
  "iso_compliance": <1-5>,
  "capa_quality": <1-5>,
  "uncited_claims": <int>,
  "overall": <mean of 4 scores>
}}"""


def llm(prompt: str, model: str = OPENAI_MODEL) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def retrieve_fda_evidence(complaint: str, col: chromadb.Collection, n: int = 5) -> list[dict]:
    """Semantic search over recall index."""
    emb = embedder.encode([complaint]).tolist()
    results = col.query(query_embeddings=emb, n_results=n)
    evidence = []
    for i, doc in enumerate(results["documents"][0]):
        evidence.append({
            "id": results["ids"][0][i],
            "snippet": doc[:300],
            "relevance_score": round(1 - results["distances"][0][i], 3),
            "root_cause": results["metadatas"][0][i].get("root_cause", ""),
        })
    return [e for e in evidence if e["relevance_score"] >= 0.3]


def compute_trend(complaint: str, all_embeddings: np.ndarray, threshold: float = 0.75) -> dict:
    """Simple trend: count how many historical events are similar, return growth signal."""
    emb = embedder.encode([complaint])[0]
    sims = np.dot(all_embeddings, emb) / (
        np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(emb) + 1e-9
    )
    cluster_size = int(np.sum(sims > threshold))
    return {
        "cluster_size": cluster_size,
        "trend_flag": "emerging" if cluster_size > 5 else "stable",
        "note": f"{cluster_size} similar events found above similarity {threshold}",
    }


def run_pipeline(
    complaint: dict,
    col: chromadb.Collection,
    all_embeddings: np.ndarray,
    use_rag: bool = True,
    use_reflection: bool = True,
    use_temporal: bool = True,
    baseline_mode: bool = False,
) -> dict:
    """
    Run one complaint through the pipeline with the given feature flags.
    baseline_mode = True → single LLM call (ignores all other flags).
    """
    narrative = complaint["text"]

    if baseline_mode:
        # A5: single LLM call — extraction + risk + CAPA in one shot
        prompt = f"""You are a medical device quality expert.
Given this complaint, produce a combined extraction + risk assessment + CAPA.

COMPLAINT:
<user_narrative>
{narrative}
</user_narrative>

Use ISO 14971. Risk level must be ACCEPTABLE, ALARP, or UNACCEPTABLE.

Respond with JSON containing: modality, component, failure_mode, severity_indicator,
qms_complaint_category, software_related, confidence, hazardous_situation, harm,
severity (level+rationale), probability (level+rationale), risk_level,
evidence_citations (empty list), uncertainty, capa (immediate_containment,
root_cause_investigation, corrective_action, preventive_action,
verification_method, effectiveness_criteria, precedent_basis),
escalation_required."""
        raw = llm(prompt)
        result = json.loads(raw)
        result["config"] = "A5_baseline"
        result["evidence_used"] = []
        result["trend"] = {}
        result["reflection_applied"] = False
        return result

    # Step 1: Extraction (Agent 1)
    extraction_raw = llm(EXTRACTION_PROMPT.format(narrative=narrative))
    extraction = json.loads(extraction_raw)

    # Step 2: Retrieval (Agent 3) — toggled by use_rag
    evidence = []
    if use_rag:
        evidence = retrieve_fda_evidence(narrative, col)

    # Step 3: Temporal scoring (Similarity Module) — toggled by use_temporal
    trend = {}
    if use_temporal and len(all_embeddings) > 0:
        trend = compute_trend(narrative, all_embeddings)

    # Step 4: Risk + CAPA (Agent 4)
    risk_raw = llm(RISK_CAPA_PROMPT.format(
        narrative=narrative,
        extraction=json.dumps(extraction, indent=2),
        evidence=json.dumps(evidence, indent=2) if evidence else "None retrieved.",
        trend=json.dumps(trend, indent=2) if trend else "Temporal scoring disabled.",
    ))
    risk_result = json.loads(risk_raw)

    # Step 5: Self-critique (Agent 5) — toggled by use_reflection
    reflection_applied = False
    if use_reflection:
        critique_raw = llm(SELF_CRITIQUE_PROMPT.format(
            assessment=json.dumps(risk_result, indent=2)
        ))
        critique = json.loads(critique_raw)
        if critique.get("verdict") == "FAIL" and critique.get("revision"):
            risk_result = critique["revision"]
            reflection_applied = True

    risk_result["config"] = "pipeline"
    risk_result["extraction"] = extraction
    risk_result["evidence_used"] = evidence
    risk_result["trend"] = trend
    risk_result["reflection_applied"] = reflection_applied
    return risk_result


def judge_output(complaint: dict, result: dict) -> dict:
    """LLM-as-Judge scoring of a pipeline output."""
    raw = llm(
        JUDGE_PROMPT.format(
            narrative=complaint["text"],
            assessment=json.dumps(result, indent=2),
        ),
        model=JUDGE_MODEL,
    )
    scores = json.loads(raw)
    # Validate risk level correctness against gold
    gold_risk = complaint.get("gold_risk", "")
    predicted_risk = result.get("risk_level", "")
    scores["risk_correct"] = int(predicted_risk == gold_risk)
    # Validate modality
    gold_mod = complaint.get("gold_modality", "")
    predicted_mod = result.get("modality") or result.get("extraction", {}).get("modality", "")
    scores["modality_correct"] = int(predicted_mod == gold_mod)
    return scores


# ---------------------------------------------------------------------------
# Main: run all ablations
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("ABLATION STUDY — Medical Signal Intelligence Pipeline")
    print("=" * 70)

    # ── 1. Fetch recall data ──────────────────────────────────────────────
    print("\n[1/4] Fetching recall data from openFDA...")
    recalls = fetch_recalls_from_api(PRODUCT_CODES, limit=RECALLS_LIMIT)
    print(f"  Total recalls fetched: {len(recalls)}")

    # ── 2. Build ChromaDB index ───────────────────────────────────────────
    print("\n[2/4] Building ChromaDB recall index...")
    col = build_recall_index(recalls)

    # Build embeddings matrix for temporal scoring
    recall_texts = [
        " ".join(filter(None, [r.get("reason_for_recall", ""), r.get("root_cause_description", "")]))
        for r in recalls
        if r.get("reason_for_recall", "").strip()
    ][:200]
    all_embeddings = embedder.encode(recall_texts) if recall_texts else np.zeros((0, 384))
    print(f"  Temporal index: {len(recall_texts)} recall embeddings")

    # ── 3. Define configurations ──────────────────────────────────────────
    configs = [
        {"name": "A5_baseline",     "baseline_mode": True,  "use_rag": False, "use_reflection": False, "use_temporal": False},
        {"name": "A1_no_rag",       "baseline_mode": False, "use_rag": False, "use_reflection": True,  "use_temporal": True},
        {"name": "A2_no_reflection","baseline_mode": False, "use_rag": True,  "use_reflection": False, "use_temporal": True},
        {"name": "A4_no_temporal",  "baseline_mode": False, "use_rag": True,  "use_reflection": True,  "use_temporal": False},
        {"name": "full_pipeline",   "baseline_mode": False, "use_rag": True,  "use_reflection": True,  "use_temporal": True},
    ]

    # ── 4. Run ablations ──────────────────────────────────────────────────
    print(f"\n[3/4] Running {len(configs)} configs × {len(TEST_COMPLAINTS)} complaints...")
    print(f"      LLM: {OPENAI_MODEL}  |  Judge: {JUDGE_MODEL}\n")

    all_results = {}   # config_name → list of scored results

    for cfg in configs:
        name = cfg["name"]
        print(f"  Config: {name}")
        scores_list = []

        for i, complaint in enumerate(TEST_COMPLAINTS):
            print(f"    [{i+1}/{len(TEST_COMPLAINTS)}] {complaint['id']}...", end=" ", flush=True)
            try:
                result = run_pipeline(
                    complaint, col, all_embeddings,
                    use_rag=cfg["use_rag"],
                    use_reflection=cfg["use_reflection"],
                    use_temporal=cfg["use_temporal"],
                    baseline_mode=cfg["baseline_mode"],
                )
                scores = judge_output(complaint, result)
                scores["complaint_id"] = complaint["id"]
                scores["config"] = name
                scores["reflection_applied"] = result.get("reflection_applied", False)
                scores["evidence_count"] = len(result.get("evidence_used", []))
                scores_list.append(scores)
                print(f"✓ (overall={scores.get('overall', 0):.1f})")
            except Exception as e:
                print(f"✗ ERROR: {e}")
                scores_list.append({
                    "complaint_id": complaint["id"], "config": name,
                    "accuracy": 0, "grounding": 0, "iso_compliance": 0,
                    "capa_quality": 0, "uncited_claims": 99, "overall": 0,
                    "risk_correct": 0, "modality_correct": 0,
                    "reflection_applied": False, "evidence_count": 0,
                })
            time.sleep(0.5)   # rate limit

        all_results[name] = scores_list

    # ── 5. Print comparison table ─────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("RESULTS: Ablation Comparison Table")
    print("=" * 70)

    def mean(lst): return round(statistics.mean(lst), 2) if lst else 0

    metrics_to_show = [
        ("overall",          "Overall score (1-5)"),
        ("accuracy",         "Accuracy (1-5)"),
        ("grounding",        "Evidence grounding (1-5)"),
        ("iso_compliance",   "ISO compliance (1-5)"),
        ("capa_quality",     "CAPA quality (1-5)"),
        ("risk_correct",     "Risk level correct (%)"),
        ("modality_correct", "Modality correct (%)"),
        ("uncited_claims",   "Uncited claims (lower=better)"),
        ("evidence_count",   "Avg FDA evidence retrieved"),
    ]

    header = f"{'Metric':<30}" + "".join(f"{c['name'][:18]:>20}" for c in configs)
    print(header)
    print("-" * (30 + 20 * len(configs)))

    summary = {}
    for key, label in metrics_to_show:
        row = f"{label:<30}"
        for cfg in configs:
            vals = [s.get(key, 0) for s in all_results[cfg["name"]]]
            if key in ("risk_correct", "modality_correct"):
                val = round(mean(vals) * 100, 1)
                row += f"{str(val) + '%':>20}"
            else:
                val = mean(vals)
                row += f"{val:>20}"
            summary.setdefault(cfg["name"], {})[key] = val
        print(row)

    # Delta vs baseline
    print("\n" + "-" * (30 + 20 * len(configs)))
    print(f"{'Δ vs A5_baseline (overall)':30}", end="")
    baseline_overall = summary["A5_baseline"].get("overall", 0)
    for cfg in configs:
        delta = summary[cfg["name"]].get("overall", 0) - baseline_overall
        tag = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        print(f"{tag:>20}", end="")
    print()

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    full = summary.get("full_pipeline", {})
    base = summary.get("A5_baseline", {})
    no_rag = summary.get("A1_no_rag", {})
    no_ref = summary.get("A2_no_reflection", {})
    no_tmp = summary.get("A4_no_temporal", {})

    print(f"\n  A1 (no RAG):        grounding {no_rag.get('grounding',0):.2f} vs full {full.get('grounding',0):.2f}")
    print(f"                      → RAG {'improves' if full.get('grounding',0) > no_rag.get('grounding',0) else 'does not improve'} evidence grounding")

    print(f"\n  A2 (no reflection): uncited {no_ref.get('uncited_claims',0):.1f} vs full {full.get('uncited_claims',0):.1f}")
    print(f"                      → Self-critique {'reduces' if full.get('uncited_claims',0) < no_ref.get('uncited_claims',0) else 'does not reduce'} uncited claims")

    print(f"\n  A4 (no temporal):   overall {no_tmp.get('overall',0):.2f} vs full {full.get('overall',0):.2f}")
    print(f"                      → Temporal scoring {'adds' if full.get('overall',0) > no_tmp.get('overall',0) else 'does not add'} signal value")

    print(f"\n  A5 (baseline) vs full: {base.get('overall',0):.2f} → {full.get('overall',0):.2f}")
    delta_pct = ((full.get('overall',0) - base.get('overall',0)) / max(base.get('overall',0), 0.01)) * 100
    print(f"                      → Multi-agent pipeline is {delta_pct:+.1f}% vs single-LLM baseline")

    # Save raw results
    out_path = os.path.join(os.path.dirname(__file__), "data", "ablation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Raw results saved to: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
