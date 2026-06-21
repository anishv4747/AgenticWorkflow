# PLAN_v2: Risk Analysis Agent — Autonomous Tool-Calling Design

This document supersedes `PLAN.md` §3.2.1 and §4 "Agent 2: Risk Analysis Agent" for the
**tool invocation model only**. The ISO 14971 methodology, risk matrix, CAPA rules, and
escalation logic in `PLAN.md` are unchanged — only *who decides when to call which tool*
has changed. `PLAN.md` remains the source of truth for everything else; this file does not
replace it.

## What changed from v1, and why

**v1**: `risk_analysis_agent` called its 6 deterministic tools as a **hardcoded sequential
Python function chain** (`score_severity` → `compute_probability` → `apply_risk_matrix` →
`load_past_reports` → `lookup_capa_requirements` → one freeform LLM narrative call →
`validate_evidence_citations` → `compute_escalation_flags`). The LLM only ever wrote prose;
code decided which tools ran, in what order, every single time.

**v2**: the LLM autonomously decides which tools to call, in what order, and how many times,
via a real Anthropic tool-calling loop — matching the `Reference/pattern2.py` architecture
(a `while` loop dispatching on `tool_use` blocks, with a `set_X`-style finalize tool) and
`Reference/pattern1.py`'s adaptive-memory style for episodic lookup.

Two scoped autonomy decisions were made deliberately, not by default:

1. **`score_severity`, `compute_probability`, `apply_risk_matrix` are now LLM-callable
   tools**, not just `load_past_reports`/CAPA lookup. This was a conscious choice to favor
   maximum autonomy over keeping these three as code-only calls. The trade-off this creates
   (the LLM might omit, misorder, or call them with wrong values) is closed by the
   enforcement mechanism below — the numbers still can never be LLM-invented, only the
   *decision to invoke* the tool that computes them is now autonomous.
2. **`load_past_reports` is now a true adaptive tool** — zero, one, or many calls, with
   parameters the LLM chooses (e.g. retrying with a broader modality or a shorter
   failure_mode keyword if the first query is empty), instead of an unconditional preload
   that ran before the LLM ever saw the case.

**Unchanged, deliberately**: `validate_evidence_citations` and `compute_escalation_flags`
stay **code-only, mandatory, never LLM-callable**. These are the "constitutional guardrail
enforced in code, not just in the prompt" (`PLAN.md` line 668; `CLAUDE.md`: "there is no
separate Citation Critic"). Exposing them as tools would let the LLM decide whether to run
its own guardrail, which defeats the point of a guardrail.

## Tool inventory

| Tier | Tool | LLM-callable? | Calls |
|---|---|---|---|
| Classification (mandatory, autonomous order) | `score_severity` | Yes | exactly 1 effective |
| | `compute_probability` | Yes | exactly 1 effective |
| | `apply_risk_matrix` | Yes | exactly 1 effective (re-callable if stale) |
| | `lookup_capa_requirements` | Yes | exactly 1 effective |
| Episodic memory | `load_past_reports` | Yes | 0 or more, LLM-chosen params |
| Finalize | `submit_risk_narrative` | Yes | exactly 1, terminates the loop |
| Guardrails (mandatory, code-only) | `validate_evidence_citations` | **No** | always, post-loop |
| | `compute_escalation_flags` | **No** | always, post-loop |

`submit_risk_narrative` is new — it replaces the old freeform-JSON-text + regex-parsing
(`_parse_json_response`) approach. Its `input_schema` defines every narrative field
(`hazardous_situation`, `harm`, rationale fields, `evidence_basis`, `uncertainty`, all
`capa_*` fields). The Anthropic API validates structured tool input against the schema,
which is strictly more reliable than parsing markdown-fenced freeform JSON.
**`severity_level`/`probability_level`/`risk_level` are deliberately absent from this
schema** — they are never LLM-authored, even at finalize time; the agent reads them from
the tool-call transcript it tracked during the loop, not from the finalize call's input.

## Enforcement mechanism — why autonomy doesn't mean the numbers can drift

Two checks run in code, intercepting tool dispatch (not as LLM-visible "checker tools" the
model could choose to skip — see `agents/risk_analysis.py::_dispatch_tool_call`):

1. **Value-fidelity check** (at `apply_risk_matrix` and `lookup_capa_requirements`): each
   call's arguments are compared against the most recently observed output of the upstream
   tool in this conversation. A mismatch is rejected with a `tool_result(is_error=True)`
   explaining the discrepancy, forcing the LLM to re-call with the correct values. This
   closes the loophole where an LLM could call `score_severity` once (satisfying a
   presence-only check) and then call `apply_risk_matrix` with fabricated levels.
2. **Presence + staleness check** (at `submit_risk_narrative`): rejects the finalize call
   unless `score_severity`, `compute_probability`, `apply_risk_matrix`, and
   `lookup_capa_requirements` were all successfully called in this conversation, AND a
   re-derivation of `apply_risk_matrix(last_severity_level, last_probability_level)` still
   matches the recorded `risk_level` — this catches the case where the LLM re-derives
   different severity/probability values *after* an earlier successful `apply_risk_matrix`
   call, leaving a stale `risk_level` on record.

Additionally, `score_severity`'s `complaint_text` argument and `compute_probability`'s
`event_count`/`recall_count`/`growth_rate_30d`/`cluster_size` arguments are **always
overridden from `state` at dispatch time**, regardless of what the LLM supplies in the tool
call. The schema still asks the model to supply them (keeps the tool self-documenting and
lets the model reason about its own call), but the dispatch layer never trusts those values
— removing an entire class of hallucinated-input failure modes.

## Loop safety

`agents/risk_analysis.py` caps the tool-calling loop at `MAX_ITERS = 6` iterations /
`TIME_BUDGET_S = 30` seconds — consistent with `CLAUDE.md`'s existing reliability-pattern
convention (Agent 3 ReAct: 5 iters/30s; Agent 5 self-critique: 2 rounds/20s).

On cap-exceeded without a successful `submit_risk_narrative`: a **deterministic fallback**
runs (`_deterministic_fallback`), not further LLM self-assessment. It calls
`score_severity` → `compute_probability` → `apply_risk_matrix` → `lookup_capa_requirements`
directly for whichever weren't already obtained, guaranteeing `severity_level` /
`probability_level` / `risk_level` are **never missing**, even in the worst case. Only
narrative prose fields are left `null`, with `uncertainty` explicitly flagging that the
narrative is incomplete and requires human authoring. `compute_escalation_flags` then
naturally fails `gate3_passed` closed for an `UNACCEPTABLE`-with-zero-citations fallback
case — the correct conservative behavior.

## Batching independent tool calls

Claude can return multiple `tool_use` blocks in a single turn when calls don't depend on
each other's output. The dispatch loop already iterates every block per turn and returns
every result together before the next API call (required by the tool-use protocol) — same
shape as `Reference/pattern2.py`'s `weather_agent`/`surge_agent` and
`Reference/pattern3.py`'s `nutrition_agent`. The system prompt hints that
`score_severity` + `compute_probability` + `load_past_reports` can be requested together in
the first turn (none depend on each other), while `apply_risk_matrix` →
`lookup_capa_requirements` remain strictly sequential. This typically collapses a run to
3-4 turns instead of 6 sequential round-trips. No LangGraph fan-out is needed — this stays
one node/one agent; the parallelism is within a single model turn, not across graph nodes.

## Revised reproducibility target

`PLAN.md` line 672 previously stated: *"severity_level/probability_level/risk_level must be
byte-identical across repeated runs of the same input (reproducibility test)."* That target
assumed a fixed call sequence and no longer holds verbatim once tool invocation is
LLM-autonomous. It is replaced by three more precise targets:

- **(a) Value reproducibility**: given identical `state` inputs, *if* the agent reaches
  `submit_risk_narrative`, the resulting `severity_level`/`probability_level`/`risk_level`/
  CAPA-timing fields are still always sourced from the same deterministic tool functions on
  the same inputs — guaranteed by the value-fidelity + staleness checks, not by call order.
- **(b) Trajectory non-determinism, now explicit and accepted**: the number of tool calls,
  their order, batching, and retry pattern may vary between runs of the same input. This is
  the accepted cost of autonomy and is no longer treated as a defect.
- **(c) New regression target**: 0% rate of `risk_level` (or `severity_level`/
  `probability_level`) being accepted into a finalized report from an unenforced or stale
  tool call — testable directly against `_dispatch_tool_call`'s `called_tools`/staleness
  logic (e.g. a unit test that feeds a `submit_risk_narrative` call before the mandatory
  tools and asserts it's rejected with `is_error=True`).

## Updated Agent 2 classification

`PLAN.md`'s Agent 2 "Autonomy" row (line 633) is still accurate in **outcome** — numbers are
never LLM-generated — but should be read with this addendum: **tool *invocation* is now
LLM-autonomous** (an L1 Augmented LLM with tool-use and a code-enforced finalize gate)
rather than hardcoded tool sequencing. This is a deliberate, user-approved exception to
`CLAUDE.md`'s "use the lowest autonomy level that solves the problem" principle, scoped to
this agent only — it was chosen explicitly over the lower-autonomy alternative (keeping
classification tools code-only) during the design of this change.

`PLAN.md` lines 641–650 ("Responsibilities: tool call → tool call → narrow LLM call →
validate") should be read as superseded by:

1. The agent enters a tool-calling loop. The LLM may call `score_severity`,
   `compute_probability`, `apply_risk_matrix`, `lookup_capa_requirements` (mandatory,
   autonomous order/batching, value-fidelity enforced) and `load_past_reports` (optional,
   zero-or-more, adaptive) in any sequence respecting data dependencies.
2. A code-side check at `submit_risk_narrative` rejects finalization unless all four
   mandatory tools were called and their chained values are internally consistent.
3. On successful finalize: `validate_evidence_citations` (mandatory, code-only) strips
   hallucinated citations.
4. `compute_escalation_flags` (mandatory, code-only) computes final escalation flags.
5. A loop cap (6 iterations / 30s) triggers the deterministic fallback described above if
   the LLM never successfully finalizes.

No other `PLAN.md` sections change. Gate 3's trigger conditions (`PLAN.md` line 624) are
unaffected — `gate3_passed`/`evidence_basis` are still computed identically by
`compute_escalation_flags` after the loop.

## Files touched by this change

- `agents/risk_analysis.py` — full rewrite of `risk_analysis_agent`: new `_TOOLS` schema
  list, new `_AGENT_SYSTEM` prompt, new dispatch loop (`_dispatch_tool_call`), deterministic
  fallback (`_deterministic_fallback`). `load_past_reports`/`init_db`/`save_report`/SQLite
  setup unchanged, just newly dispatchable as a tool. Return dict shape unchanged (21 keys,
  matches `ComplaintState` in `pipeline.py`).
- `agents/risk_tools.py` — **no logic changes**. All 6 functions were already
  JSON-schema-compatible (plain dict/str/bool/int/float/list returns). `apply_risk_matrix`'s
  bare-string return is wrapped as `{"risk_level": ...}` at the dispatch layer in
  `risk_analysis.py`, not inside `risk_tools.py`.
- `pipeline.py` — no changes. `ComplaintState`, node registration, and `gate3_router` all
  already match the unchanged 21-key return contract.
- `README.md` — "two-pass LLM" wording updated to "autonomous tool-calling loop" for
  consistency with this design.
