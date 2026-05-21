# Aggregation Report — llm-psychology → PaperOrchestra Inputs

**Generated:** 2026-05-21
**Source project:** `/Users/zhanshaoxiong.3/Desktop/new-exp/llm-psychology`
**Workspace:** `/Users/zhanshaoxiong.3/Desktop/new-exp/llm-psychology/paper-orc`
**Mode:** structured-source distillation (single-project, author-curated)

---

## Decision Gate

Both `workspace/inputs/idea.md` and `workspace/inputs/experimental_log.md` were **absent** before this run. The user explicitly pointed to a fully developed research project (code, data, exp, logs, paper, paper-orc/inputs/template.tex), so this skill ran in **structured-source distillation mode** rather than the cache-discovery mode designed for scattered `.claude` / `.cursor` logs. The `discover_logs.py` script was not invoked because the project already has an authoritative paper draft (`paper/paper.tex`, 72 KB) and 56 CSV result files in `code/results/`. The distillation prefers these structured sources to LLM-extracted log fragments.

## Phase 1 — Discovery Summary

- **Search root:** 1 (the user-provided project directory).
- **Discovery mode:** author-provided paths (deterministic; no LLM extraction needed at this phase).
- **Single project confirmed.** No project-selection ambiguity.
- **Files catalogued:** 32 distinct artifacts (1 README, 1 paper.tex, 1 results summary, 1 battery spec, 4 source scripts, ~25 result CSVs plus figure directory and logs directory). Full manifest: `paper-orc/ara/discovered_logs.json`.
- **Per-source breakdown:**

  | Source kind | Count | Notes |
  |---|---:|---|
  | project_overview (README.md) | 1 | provides headline findings, instrument table, model list, repo layout |
  | draft_paper (paper.tex) | 1 | authoritative source for all numeric claims |
  | results_summary (PSYCHOMETRIC_RESULTS_SUMMARY.md) | 1 | author-written narrative for round-1 analyses |
  | experiment_runner / battery_spec / runner (.py + .md) | 4 | prompt, k=5 sampling protocol, item battery |
  | analysis_csv (code/results/*.csv) | ~25 | per-analysis output, used to verify paper numbers |
  | figure_directory (paper/figures/) | 1 | 40+ PNG/PDF figures already produced |
  | experiment_logs (logs/) | 1 | per-model API run logs |

## Phase 2 — Extraction Summary

- **Mode:** distillation from structured sources (no scattered cache batches needed).
- **Experiment records produced:** 13 distinct experiments captured in `paper-orc/ara/raw_experiments.json`:
  1. Experimental design overview
  2. Internal consistency (Cronbach's α)
  3. Acquiescence mechanism (forward/reverse decomposition)
  4. Factor structure (EFA, leave-one-out, Tucker's φ)
  5. Variance decomposition (Likert + binary)
  6. Convergent validity (8 construct pairs)
  7. Measurement invariance (Default vs MBTI, 320 pairs)
  8. Synthetic baselines (Random, Pure-Acq, Trait+Acq)
  9. Persona steering (PSD, fidelity, target adherence)
  10. MBTI factorial main effects
  11. Item-level plasticity
  12. Within-sample consistency
  13. DIF across model families
- **Date range of original analyses:** 2026-05-13 → 2026-05-21 (verified via `logs/` timestamps and git log).
- **Iterations detected:** 7 rounds of analysis from initial 18-model pilot to final 20-model paper-ready dataset (documented in synthesis.json `iteration_history`).
- **Convergence direction:** monotonic — sample expanded from 18 to 20 models, framing tightened from "multiple symptoms" to "single causal chain", PIR estimate updated from 0.584 (n=18 round 1) to 0.468 (n=20 final paper).

## Phase 3 — Synthesis Summary

- **Single research question:** "Do validated human personality instruments measure genuine traits in LLMs, or do the resulting scores merely reflect a uniform agreeable response style imposed by safety/alignment training?"
- **Single hypothesis:** all standard psychometric checks fail; failures share one cause — acquiescence — with a directional asymmetry across normal-personality and dark-trait items.
- **7 key contributions** distilled (see `synthesis.json` → `key_contributions`).
- **8 results tables** consolidated into `experimental_log.md` §2 (α by domain, acquiescence Δ, PIR, EFA loadings, variance decomposition, synthetic baselines, convergent validity, persona steering, WSC by model, WSC by persona).
- **10 qualitative observations** consolidated into `experimental_log.md` §3 — single mechanism, two faces of one bias, default-not-neutral, persona-stabilizes, compliance-vs-validity, locked items, vendor-stability gap, calibration story, refusal pattern, cross-family null DIF.

## Phase 4 — Formatting Output

Generated files in `workspace/inputs/`:

| File | Size | Status |
|---|---:|---|
| `idea.md` | 6.8 KB | new — Sparse Idea format (Problem / Hypothesis / Method / Key Contributions / Open Questions) |
| `experimental_log.md` | 24.7 KB | new — Setup / Raw Data (15 tables) / Qualitative Observations / Iteration History |
| `conference_guidelines.md` | 5.3 KB | new — distilled ACL spec (page limit, anonymity, mandatory sections, numeric conventions) |
| `template.tex` | 15 KB | pre-existing (user-supplied ACL style) |
| `formatting.md` | 18 KB | pre-existing (full ACL formatting spec, retained as reference) |
| `acl.sty`, `acl_natbib.bst`, `custom.bib`, `anthology.bib.txt`, `README.md` | various | pre-existing |
| `figures/` | 0 files | empty (see Handoff below) |

Internal ARA artifacts in `workspace/ara/`:

| File | Size | Purpose |
|---|---:|---|
| `discovered_logs.json` | 9.0 KB | catalogue of 32 source artifacts |
| `raw_experiments.json` | 22.4 KB | 13 extracted experiment records |
| `synthesis.json` | 23.6 KB | consolidated narrative + 8 tables + qualitative obs + iteration history |
| `aggregation_report.md` | (this file) | audit trail |

## Data Quality

### Verified

- **Sample sizes consistent.** 20 models × 17 personas × 221 items × 5 samples = 375,700 cells. Confirmed against `paper.tex` §3.2 and `code/data/exp_mbti_*.json` (20 model files).
- **Cronbach's α values** cross-checked against `code/results/cronbach_alpha_by_domain.csv` and `paper/paper.tex` Table 2. Match to 3 decimals.
- **Variance decomposition** matches `code/results/variance_decomposition.csv` exactly (0.35 % / 23.93 % / 4.09 % / 36.13 % / 35.51 % Likert; 0.51 % / 5.27 % / 14.06 % / 14.81 % / 65.35 % binary).
- **EFA eigenvalues** match `code/results/efa_eigenvalues.csv` (6.97, 4.33, 3.07 for first three; 17 eigenvalues total).
- **Persona fidelity 319/320** matches `code/results/persona_fidelity_confusion.csv` (the off-diagonal is DeepSeek-V3.2 ESFP → ENTP).
- **Within-sample consistency by model** matches `code/results/wsc_by_model.csv` exactly.
- **Refusal count 61 / 375,700** matches `paper.tex` Appendix C.

### Warnings

1. **Two-version PIR.** The intermediate `PSYCHOMETRIC_RESULTS_SUMMARY.md` (round-1 analysis on 18 models) reports PIR = 0.584 and r(PIR, SDR) = −0.562. The final paper (n = 20) reports PIR = 0.468 and r(reverse-agree, PIR) = +0.63. **The final-paper values are canonical and are what `experimental_log.md` records.** Downstream agents should not pull from `PSYCHOMETRIC_RESULTS_SUMMARY.md` for headline PIR numbers.

2. **`synthetic_baselines.csv` minor format drift.** The CSV stores `LLM Observed` with the same alpha values as the paper Table 4, but the row order is different. The `experimental_log.md` §2.7 uses the paper's row order for narrative coherence; values are identical.

3. **Bibliography source ambiguity.** The repository has both `paper/refs.bib` (12 KB, project bibliography used by `paper.tex`) and `paper-orc/inputs/custom.bib` (2 KB, ACL template starter). The literature-review-agent should treat `paper/refs.bib` as the authoritative pool and merge any new verified entries into `custom.bib` (per `conference_guidelines.md`).

4. **Existing figures not pre-loaded.** `paper-orc/inputs/figures/` is empty, but 40+ publication-quality figures exist at `paper/figures/` (fig1_research_question_concept.png, fig2_methodology_pipeline.png, fig3_acquiescence_mechanism.png, fig4_variance_decomposition.png, fig5_convergent_validity.png, fig6_invariance_robustness.png, fig7_model_clustering.png, fig8_mbti_dimension_effects.png, fig9_persona_sensitivity.png, fig10_within_family_divergence.png, fig11_default_bias_and_adherence.png, fig12_persona_fidelity_confusion.png, fig13_cross_scale_persona_coherence.png, fig14_persona_vector_field.png, fig_heatmap_default.png, fig_wsc1–9_*.png, figA1–A3_*.png, plus PDF variants). Optional: copy them to `paper-orc/inputs/figures/` to let the plotting agent skip regeneration. See "Handoff" below.

5. **One model-typing inconsistency in the project.** Some result files use `GPT_5.2` and `Gemini_3-Pro-Preview` (underscore) while others use `GPT-5.2` and `Gemini-3-Pro-Preview` (hyphen). Both refer to the same model. The synthesis adopts the hyphenated form, which is what `paper.tex` uses.

### Not Fabricated

- All numeric claims trace to either `paper/paper.tex` or `code/results/*.csv`. No values were inferred or extrapolated.
- The `[UNVERIFIED]` tag is not used because every value used has a primary source.

## Handoff to PaperOrchestra

The workspace is **ready for paper-orchestra** with one optional and three minor items:

### Required (all satisfied)

| File | Status |
|---|---|
| `workspace/inputs/idea.md` | ✓ generated |
| `workspace/inputs/experimental_log.md` | ✓ generated |
| `workspace/inputs/template.tex` | ✓ pre-existing (ACL style with `[review]` mode) |
| `workspace/inputs/conference_guidelines.md` | ✓ generated (distilled from `formatting.md`) |

### User decisions (resolved 2026-05-21)

1. **Figures:** ✗ NOT copied. User chose to let the plotting agent regenerate everything from raw data, ensuring figures match the new outline exactly. `paper-orc/inputs/figures/` remains empty.
2. **Bibliography:** ✓ `paper/refs.bib` (12 KB, ~70 entries) copied to `paper-orc/inputs/refs.bib` as the verified starting pool. The literature-review agent should treat `refs.bib` as authoritative and add any newly verified entries via Semantic Scholar to `custom.bib`.
3. **Track:** Long-paper (8 review / 9 final pages). `conference_guidelines.md` already reflects this default.

### Suggested next step

```
# After you confirm the optional items, run:
paper-orchestra --workspace /Users/zhanshaoxiong.3/Desktop/new-exp/llm-psychology/paper-orc
```

The outline agent will read `idea.md` + `experimental_log.md` + `template.tex` + `conference_guidelines.md` and produce a strict-JSON outline; the plotting agent and literature-review agent then run in parallel; the section-writing agent merges everything; the content-refinement agent iterates the draft to convergence.

## Hard-Rule Compliance

- ✓ No writes to user agent cache directories (none were scanned).
- ✓ No personal information leaked into generated `idea.md` or `experimental_log.md`. Authors are "Anonymous Authors" per the existing ACL `[review]` template.
- ✓ No fabricated results — every numeric value traces to `paper/paper.tex` or `code/results/*.csv`.
- ✓ Discovery file list contains 32 items (< 50), so explicit user file-by-file confirmation was not required.
