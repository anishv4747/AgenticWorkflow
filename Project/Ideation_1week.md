# Ideation: 1-Week Build Plan
## Multi-Agent Regulatory Signal Intelligence System

> Compressed from the 4-week plan. Build timeline: **1 week, 6 members (M1–M6)**.
> Source of truth for architecture and schemas remains [System_Design.md](System_Design.md).
> Regulatory grounding and technique catalog in [Ideation.md](Ideation.md).

---

## What This System Does (The Goal)

A Quality Manager (QM) at a medical imaging company receives a complaint:

> *"During cardiac MRI on Philips Achieva 1.5T, the SSFP reconstruction produced banding artifacts. Radiologist couldn't interpret the images. Patient needed a repeat scan."*

Today, that QM spends **4–8 hours** manually:
1. Classifying the failure mode
2. Searching internal CAPA records for similar past issues
3. Running keyword searches across FDA MAUDE (the adverse event database)
4. Checking the FDA recall database for same manufacturer/model
5. Drafting a risk assessment using ISO 14971 methodology
6. Writing CAPA recommendations citing precedents
7. Formatting a report for the Quality Review Board (QRB)

**This system does steps 1–7 in under 2 minutes, producing a draft the QM reviews and approves.**

This is **decision support, not decision automation**. The QM remains in the loop. Nothing is autonomously submitted to regulators.

---

## What the System Outputs

For the complaint above, the system generates a structured Signal Report containing:

| Section | Content |
|---------|---------|
| Extracted fields | Modality: MRI, Component: reconstruction pipeline, Failure mode: banding artifact, Severity: S3 (Serious) |
| Pattern match | Cluster #4 — "Image Quality/Reconstruction Artifacts", 43 similar events, growing at +12%/month |
| FDA evidence | 12 matching MAUDE events (report numbers cited), 2 Philips recalls with "Software design" root cause |
| Risk assessment | Severity S3 × Probability P4 = **UNACCEPTABLE** per ISO 14971 5×5 matrix |
| CAPA recommendation | Immediate: flag SW v5.7.1. Investigate SSFP algorithm. Corrective: patch. Preventive: add automated image QA |
| Citations | 14 traceable FDA records, 0 unsupported claims |
| Compliance fields | ISO 13485 §8.2.2 category: `IMG-QUAL`, IEC 62304 §9 mapping, escalation flags |

The QM reviews this report, edits if needed, and approves. It then enters the QMS as a controlled document.

---

## Why the Data Is Already Useful

The team has already downloaded:
- **3,701 adverse events** from openFDA MAUDE (MRI, CT, Ultrasound, X-ray, Hematology, PCR)
- **3,299 recalls** — 100% have reason_for_recall, root cause, and corrective action text
- **32.6% of recalls cite "Software design"** as root cause — these are ground-truth CAPA examples
- **98.4% of events have narrative text** (average 836 characters) — rich for NLP

This is not toy data. These are real FDA records about real device failures.

---

## The 5-Component Pipeline

```
Complaint Text
      ↓
[Agent 1: Extraction]          — What failed? Where? How severe? (ISO 13485 §8.2.2)
      ↓ Gate 1
[Similarity Module]  ‖  [Agent 3: Retrieval]     — parallel
    (non-LLM)              (RAG over FDA data)
      ↓                          ↓
[Agent 4: Risk + CAPA]     — ISO 14971 risk matrix + CAPA draft (merged, no over-orchestration)
      ↓ Gate 3
[Agent 5: Report Assembly] — Self-critique loop, controlled document output
      ↓
QM Review → Approve/Reject → QMS Record
```

Each component has a **clear owner**, runs **standalone with mock inputs**, and passes typed **JSON schemas** to the next stage. Schemas are frozen Day 1 — this is the most important integration risk control.

---

## What Is Cut for 1 Week

| Technique | Why Cut |
|-----------|---------|
| RLHF / PPO | Requires reward model training + unstable RL loop — days of setup |
| LoRA / QLoRA fine-tuning | GPU training time kills the schedule |
| Knowledge distillation (GPT-4 → Phi-3) | Depends on fine-tuning above |
| Full DSPy compilation | Optimization loops need labeled data you don't have time to build |
| Graph RAG (NetworkX knowledge graph) | 80% of retrieval value from flat ChromaDB in 20% of the time |
| Active learning loop | Needs iterative labeling infrastructure |
| Tree-of-Thought | Simplify to Chain-of-Thought |
| Contrastive embedding fine-tuning | Pretrained `all-MiniLM-L6-v2` is sufficient for demo quality |
| Full ReAct multi-turn | Single-pass RAG first; upgrade only if it scores below 0.50 on gold set |

These are noted as **future work** in the final report — cutting them is not a weakness, it is scope discipline.

---

## What Is Kept (MTech Depth, Achievable in 1 Week)

| Technique | Where | Why Keep |
|-----------|-------|---------|
| Structured output + CoT | Agent 1 | Core to extraction quality, fast to implement |
| sentence-transformers embeddings | Similarity Module | Non-LLM, deterministic, fast |
| HDBSCAN clustering | Similarity Module | No hyperparameter needed, handles noise |
| UMAP 2D projection | Similarity Module + Dashboard | High demo value, 2 hours to implement |
| Temporal anomaly detection (z-score) | Similarity Module | sklearn IsolationForest — 1 hour |
| Single-pass RAG (ChromaDB) | Agent 3 | Core retrieval — non-negotiable |
| ISO 14971 5×5 risk matrix | Agent 4 | Encoded logic, not ML — fast to build |
| Self-critique (1 round cap) | Agent 5 | Demonstrates evaluator-optimizer pattern |
| Validation gates (Gate 1, 2, 3) | Orchestrator | Pure logic, no LLM, critical for reliability |
| **DPO alignment** | Agent 5 | The MTech contribution — see below |
| LLM-as-Judge evaluation | Evaluation | 20–30 labeled examples, fast to run |
| Baseline single-LLM-call | Ablation | Built Day 1 — all ablations compare against it |
| LangSmith tracing | Observability | trace_id through all components |

---

## DPO: The MTech Contribution

**Direct Preference Optimization** replaces RLHF/PPO as the alignment study.

**What it is**: You collect pairs of (good report, bad report) for the same input. DPO fine-tunes a smaller model (Phi-3-mini or Llama-3-8B) to prefer the good version — without training a separate reward model.

**Why it fits in 1 week**:
- TRL library handles training
- 30 synthetic preference pairs are sufficient to show a measurable result
- Good reports: structured, ISO 14971 language, traceable citations
- Bad reports: vague, no citations, informal language
- M5 generates these pairs from real MAUDE narratives on Day 1

**The result**: A before/after comparison — baseline vs DPO-aligned report quality.

**How to frame it**: *"We applied DPO alignment to the report generation component using domain-simulated QM preferences, achieving X% win-rate over the unaligned baseline — demonstrating that alignment techniques transfer to regulated-domain document generation."*

---

## The Baseline Is Your Best Asset

Day 1 deliverable: a **single LLM call** that takes a complaint and returns extraction + risk + CAPA in one generation with 20 relevant recalls in context.

This is not a shortcut. It is **Ablation #1** — the control condition for every experiment.

| Ablation | What You Remove | What You Measure |
|---------|----------------|-----------------|
| 1 | Everything (baseline) | Control — F1, rubric, hallucination rate |
| 2 | Agent 3 (no RAG) | Precision@5 drop without FDA retrieval |
| 3 | CoT in Agent 1 | Extraction F1 drop |
| 4 | Self-critique in Agent 5 | Hallucination rate increase |
| 5 | DPO alignment | Report win-rate drop |

Five ablations from one baseline is a rigorous MTech evaluation study.

---

## 1-Week Day-by-Day Plan

### Day 1 — Foundation (everyone unblocked by end of day)

| Member | Task | Deliverable |
|--------|------|-------------|
| M1 | Load SQLite from downloaded data. Set up ChromaDB. Embed 3K+ recalls and 3.7K events. Normalize Philips/GE/Siemens manufacturer names | `signal_intelligence.db` populated. ChromaDB with 3 collections |
| M2 | Write `schemas.py` (Pydantic v2 models + mock factories for all 5 components). **This is P0 — everyone else depends on it** | `schemas.py` frozen. All downstream members can work with `mock_extraction()` etc. |
| M3 | Single-LLM-call baseline: complaint + 20 recalls in context → extraction + risk + CAPA in one prompt. Test on 3 real MAUDE narratives | `baseline.py` running, metrics recorded (this is Ablation #1) |
| M4 | Run HDBSCAN on embeddings. Generate UMAP 2D projection. Static scatter plot colored by failure mode | First cluster visualization |
| M5 | Design 30 DPO preference pairs: (good_report, bad_report) for same MAUDE narrative input. Good = citations + ISO language. Bad = vague + no citations | `data/dpo_pairs.jsonl` — training data ready |
| M6 | Label 20 gold MAUDE narratives (extraction fields + risk level). Design LLM-as-Judge prompts with rubric | `eval/gold_set.json`, judge prompt |

**End of Day 1**: baseline running end-to-end, schemas frozen, everyone has mock inputs.

---

### Day 2 — Core Agents

| Member | Task | Deliverable |
|--------|------|-------------|
| M1 | Software-relevance pre-filter (keyword: software, image, DICOM, algorithm, reconstruction). Expand event download via API pagination to 2019+ | ~5K–10K software-relevant events in DB |
| M2 | Agent 3: ChromaDB similarity search + openFDA recall lookup by product code | `retrieval.py` working standalone |
| M3 | Agent 1: CoT extraction with structured output. Map to `qms_complaint_category` (IMG-QUAL, SW-FUNC, etc.) | `extraction.py` working standalone |
| M4 | UMAP temporal slider in Streamlit (color by month). Add cluster growth rate labels | Streamlit page loading with temporal view |
| M5 | Agent 4: encode ISO 14971 5×5 matrix. Severity S1–S5, Probability P1–P5 (calibrated to dataset event counts), risk level = ACCEPTABLE / ALARP / UNACCEPTABLE | `risk_capa.py` working standalone |
| M6 | Gate 1 (confidence < 0.5 → reject) and Gate 3 (HIGH risk + 0 citations → escalate) logic | `orchestrator.py` gate logic |

---

### Day 3 — Integration + Agent 5

| Member | Task | Deliverable |
|--------|------|-------------|
| M1 | Wire orchestrator: connect all components with `trace_id` propagation. LangSmith logging | End-to-end pipeline on 1 test complaint |
| M2 | Gate 2 logic (0 results → warn). Agent 3 re-ranking by relevance score (discard < 0.3) | Retrieval with quality filter |
| M3 | Agent 5 report assembly with 1-round self-critique. Check: citations present? schema complete? uncertainty disclosed? | `report.py` with self-critique |
| M4 | Temporal anomaly detection: z-score on cluster growth rate per 30-day window. Flag clusters with z > 2 | `trend_alert` field in similarity output |
| M5 | DPO training run on 30 preference pairs using TRL. Log win-rate before/after | DPO checkpoint + win-rate metric |
| M6 | Run LLM-as-Judge on 20 gold examples. Compare baseline vs Agent 1 extraction F1 | First evaluation results |

---

### Day 4 — Evaluation + Ablation

| Member | Task | Deliverable |
|--------|------|-------------|
| M1 | End-to-end smoke test on 5 real MAUDE narratives. Log latency + cost per report | Operational metrics (cost/latency/completion rate) |
| M2 | Ablation: remove Agent 3 (no RAG) → measure Precision@5 drop. Compare to baseline | Ablation #2 results |
| M3 | Ablation: remove CoT → measure extraction F1 drop. Compare to baseline | Ablation #3 results |
| M4 | Ablation: remove Similarity Module → show degraded clustering. UMAP without trend detection | Ablation #5 results |
| M5 | DPO vs baseline comparison: report rubric score, citation density, win-rate | Ablation #4 / DPO study results |
| M6 | Cohen's kappa: LLM-Judge scores vs M6 human scores on 20 gold examples. Compile metrics table | Evaluation report — all 8 metrics |

---

### Day 5 — Demo Polish + Presentation

All members:
- Demo script: 5 compelling MAUDE narratives through full pipeline
- Architecture diagram
- Results table (baseline → ablations → full system)
- Poster / slides
- Write-up: system design, experiments, limitations, future work

**Day 5 gate**: Full pipeline runs on demo examples without errors. DPO results recorded. All 5 ablations have numbers.

---

## Team Ownership Summary

| Member | Primary Component | Key Deliverable |
|--------|------------------|----------------|
| M1 | Data engineering + orchestrator wiring | DB loaded, ChromaDB populated, pipeline connected |
| M2 | `schemas.py` + Agent 3 (Retrieval) | Frozen contracts + RAG over FDA data |
| M3 | Agent 1 (Extraction) + Agent 5 (Report) | CoT extraction + self-critique report |
| M4 | Similarity Module + Dashboard | HDBSCAN/UMAP + Streamlit temporal dashboard |
| M5 | Agent 4 (Risk + CAPA) + DPO | ISO 14971 risk matrix + alignment experiment |
| M6 | Evaluation + LLM-as-Judge | Gold set + metrics table + ablation study |

---

## Evaluation Targets

| Metric | Target | How Measured |
|--------|--------|-------------|
| Extraction F1 | > 0.80 | Gold set (M6), structured field comparison |
| Retrieval Precision@5 | > 0.65 | M6 relevance labels on Agent 3 output |
| Cluster silhouette score | > 0.40 | sklearn `silhouette_score` on HDBSCAN output |
| Risk/CAPA rubric | > 3.0 / 5.0 | LLM-as-Judge on 20 examples |
| Hallucination rate | < 15% | % claims without citation (Judge-scored) |
| DPO win-rate | > 60% | Preference eval on held-out pairs |
| LLM-Judge / human kappa | > 0.60 | Cohen's kappa on 20 shared examples |
| Cost per report | < $0.50 | Token count × GPT-4.1 pricing |

---

## Risks Specific to 1-Week Timeline

| Risk | Mitigation |
|------|-----------|
| schemas.py not ready Day 1 morning | M2 must finish by noon — blocks all other members. Escalate immediately if blocked |
| OpenAI API costs exceed budget | Use GPT-4o-mini during development. GPT-4.1 only for final evaluation runs |
| DPO training takes too long | Google Colab Pro (~$10). Phi-3-mini on 30 pairs trains in < 2 hours |
| Not enough labeled data for evaluation | 20 examples is acceptable minimum. LLM-as-Judge supplements human labels |
| Integration bugs on Day 3 | Every component runs standalone (mock inputs). Integration is just wiring, not rewriting |
| Demo fails on Day 5 | Identify 5 "safe" MAUDE narratives on Day 1 that produce clean outputs — rehearse with these |

---

## Regulatory Compliance (Non-Negotiable, Already Designed)

These are baked into the pipeline — do not simplify them out:

| Requirement | Where Encoded | What It Means in Code |
|------------|--------------|----------------------|
| ISO 13485 §8.2.2 complaint categories | Agent 1 output | `qms_complaint_category` field: `IMG-QUAL`, `SW-FUNC`, `SW-ALGO`, etc. |
| ISO 14971 risk methodology | Agent 4 | `risk_level` must be `ACCEPTABLE / ALARP / UNACCEPTABLE` — never `HIGH/MEDIUM/LOW` |
| Escalation logic | Agent 4 + Gate 3 | `escalation_required = true` when risk ∈ {ALARP, UNACCEPTABLE} |
| Document control | Agent 5 | Report has `document_id`, `revision`, `approval_status`, `generated_by` |
| AI transparency | Report footer | "AI-assisted draft — requires human review and approval" on every output |
| Citation requirement | Gate 3 | Reject UNACCEPTABLE risk with 0 evidence citations |
| Prompt injection mitigation | All agents | MAUDE narratives wrapped in `<user_narrative>` tags; system prompt forbids following instructions inside them |

---

## Final Positioning

> "An evidence-grounded multi-agent GenAI system for post-market quality signal detection in medical imaging and SaMD environments. Integrates internal complaint data with FDA/openFDA public safety intelligence, enhanced by DPO-based alignment, ISO 14971 risk methodology, and LLM-as-Judge evaluation. Built as decision support for Quality Managers — not decision automation."

This demonstrates:
- **Systems engineering**: multi-component pipeline, JSON schemas, validation gates, observability
- **Modern ML**: DPO alignment, contrastive embeddings, HDBSCAN/UMAP, temporal anomaly detection
- **Domain expertise**: ISO 14971, IEC 62304 §9, ISO 13485 §8.2.2 encoded in agent logic
- **Responsible AI**: human-in-the-loop, citation-first design, uncertainty disclosure, prompt injection mitigation
- **Rigorous evaluation**: 5 ablations, LLM-as-Judge, Cohen's kappa, outcome + operational metrics
