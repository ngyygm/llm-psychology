# Auto Review Log: Model-Dependent Response Style Variation in Open LLMs

**Paper**: EMNLP 2026 Findings
**Started**: 2026-03-25
**Reviewer**: MiniMax-M2.7

---

## Round 1 (2026-03-25)

### Assessment (Summary)
- Score: 3/10
- Verdict: Not Ready
- Key criticisms:
  1. HEXACO-H reliability catastrophically low (α=0.27) — central finding unreliable
  2. Floor effects in 13/33 models inflate variance
  3. Massive acquiescence bias (PC1=52.3%) invalidates personality interpretation
  4. No human baseline comparison
  5. Low ICC (0.22-0.46) across scales
  6. Single prompt format — zero external validity

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

# Review of "Model-Dependent Response Style Variation in Open LLMs"

## Overall Score: 3/10

The paper presents a large-scale empirical investigation but is fundamentally undermined by critical psychometric limitations that make the core findings unreliable.

## Weaknesses with Minimum Fixes

### 1. Catastrophically Low Reliability of HEXACO-Honesty (α=0.27)
- Problem: This is the paper's central finding yet fails basic psychometric standards (α < 0.30 is essentially noise).
- Fix: Either (a) use 15-20 validated HEXACO-H items from the full 60-item scale, or (b) drop HEXACO-H from the primary analysis and use scales with α > 0.70.

### 2. Floor Effects in 13/33 Models
- Problem: Floor effects inflate variance estimates, likely producing spurious "discriminative" patterns.
- Fix: Report results excluding models with >50% floor responses; acknowledge floor effects as confound.

### 3. Massive Acquiescence Bias (PC1 = 52.3%)
- Problem: Over half the variance is response style, not content.
- Fix: Apply standard psychometric corrections or reframe as "response style differences."

### 4. No Human Baseline
- Problem: Cannot determine if findings reflect human-like personality expression or LLM-specific artifacts.
- Fix: Include supplementary human dataset.

### 5. Low ICC (0.22-0.46)
- Problem: Poor internal consistency undermines treating items as measuring a single construct.
- Fix: Exclude scales with α < 0.60 from primary analyses.

### 6. Single Prompt Format
- Problem: Zero external validity.
- Fix: Test at least 2-3 prompt variations.

## Readiness: NOT READY

</details>

### Actions Taken
1. **Reframing**: The paper already uses "response style" terminology. Strengthen this framing — these are NOT personality measurements but systematic response pattern differences that reveal model alignment training.
2. **HEXACO-H caveat**: Add explicit caveat that HEXACO-H has low reliability and floor effects. Present it as exploratory signal, not validated measurement.
3. **Floor effect analysis**: Already done (alignment artifact classification). Need to also report results excluding floor-effected models.
4. **Acquiescence correction**: Already analyzed (PC1). Need to report PC1-corrected results in main tables.
5. **ICC discussion**: Add discussion of why low ICC is expected for deterministic-ish synthetic respondents.

### Status
- Continuing to Round 2

---

## Round 2 (2026-03-25)

### Assessment (Summary)
- Score: 5/10
- Verdict: Not Ready — One Critical Fix Required
- Key criticism: Remove HEXACO-H from primary analysis entirely
- Also: clarify ICC interpretation, acknowledge single-prompt limitation

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

# Simulated Review — Round 2

## Re-Score: 5/10 (marginal reject; major revision needed)

### Critical Weakness: HEXACO-H α=0.27 Cannot Be Salvaged

With α=0.27, the 5 items do not measure the same underlying construct. Any pattern is likely noise.

**Minimum fix:** Remove HEXACO-H from primary analyses entirely. Present only as footnote.

### What Would Earn 7-8/10 (Accept)
1. Remove HEXACO-H from paper entirely
2. Add one paragraph discussing ICC interpretation
3. One sentence framing single-prompt limitation

**Bottom line:** The paper's core empirical story (BFI, model differences, alignment trajectory) is solid. The HEXACO-H inclusion is the only thing preventing recommendation.

</details>

### Actions Taken
1. **HEXACO-H removed from primary ANOVA**: Now analyzing 8 dimensions (BFI-5 + Collectivism + Intuition + UA). All 8 Bonferroni-significant.
2. **HEXACO-H as descriptive observation**: Floor effect in 5/12 Study 1 models noted. Non-floor models still show significant model differences (Kruskal-Wallis H=49.63, p=5.57e-09).
3. **Floor-effect robustness**: No floor effects on any of the 8 primary dimensions.
4. **ICC interpretation**: Framed as "within-model sensitivity to generation parameters."
5. **Single-prompt caveat**: Acknowledged as proof-of-concept limitation.

### Status
- Continuing to Round 3

---

## Round 3 (2026-03-25)

### Assessment (Summary)
- Score: 6/10
- Verdict: Conditionally Acceptable
- Remaining concerns (all writing-level):
  1. "Proof-of-concept" framing weakens contribution — consider multi-prompt or re-frame
  2. ICC interpretation should be neutral (avoid favorable spin)
  3. Discuss acquiescence (PC1=52%) as both limitation and methodological contribution
  4. Address generalizability of Chinese model focus

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

# Re-Evaluation: EMNLP Findings Submission Readiness

## Overall Score: 6/10 (improved from 5/10)

## ✅ Strengths
- HEXACO-H removal: appropriately handled
- Statistical rigor: Bonferroni + FDR, α>0.74
- Effect sizes: substantial η²p (0.19-0.30)
- Study 2 adds temporal depth

## ⚠️ Remaining Concerns (all minor)
1. "Proof-of-concept" framing weakens contribution
2. ICC interpretation warrants scrutiny (avoid spin)
3. Acquiescence bias (PC1=52%) — limitation or finding?
4. Generalizability beyond Chinese models

## Final Verdict
Ready for submission IF: neutralize ICC language, add multi-prompt caveat, discuss generalizability.

</details>

### Actions Taken
- All remaining issues are paper-writing level (no new experiments needed)
- Will be addressed during paper writing phase

### Status
- **LOOP COMPLETE** — Score ≥ 6/10 threshold reached
- Final score: 6/10 (up from 3/10)
- Progression: 3 → 5 → 6 across 3 rounds

---

## Round 4 (2026-03-26) — Deep Research Review

### Assessment (Summary)
- Score: 6/10
- Verdict: Borderline
- Key criticisms:
  1. n=15 insufficient for structural analysis (correlations, PCA unreliable)
  2. HEXACO-H (α=0.27) still heavily used in main text
  3. Study 4 confounds scale with alignment (8B vs 397B)
  4. Dense vs MoE confounded with provider
  5. Convergent validity method should be preliminary
  6. Response style not operationalized

### Actions Taken
1. **Structural analysis reframed**: All n=15 analyses marked "exploratory", confidence caveats added
2. **HEXACO-H moved to appendix**: Main text reduced to 1 sentence, detailed analysis in Appendix D
3. **Study 4 caveated**: Explicitly acknowledges scale-alignment confound, reframed as "preliminary evidence"
4. **Architecture claims weakened**: "fully confounded with provider" caveat added
5. **Convergent validity reframed**: Moved to appendix, marked "preliminary"
6. **Inter-dimension structure compressed**: §4.3 reduced to 3-sentence summary, full analysis in appendix

### Status
- Continuing to Round 5

---

## Round 5 (2026-03-26)

### Assessment (Summary)
- Score: **7/10**
- Verdict: **Accept**
- All previous criticisms addressed

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

# 审稿报告 (Round 5)

## 总分：7/10
## 推荐决定：Accept (EMNLP Findings)

上一轮(Round 4)批评全部充分解决。

核心贡献稳健：15 families × 9 dimensions × 61 items × 12 seeds = 10,980 responses的systematic response style variation量化。

所有exploratory structural claims正确标注为preliminary并移至附录。

Response style indicators（midpoint, extreme, acquiescence）有效操作化了关键概念。

</details>

### Actions Taken
1. **Response style indicators**: Computed per-model midpoint responding (20-87%), extreme responding (2-53%), acquiescence bias (-0.57 to +0.24). New §4.7 with Figure.
2. **All structural analysis moved to appendix**: §4.3 compressed to 3-sentence summary
3. **HEXACO-H fully in appendix**: Main text mentions only α=0.27, all interpretive analysis in Appendix D

### Status
- **LOOP COMPLETE** — Score 7/10 (Accept threshold)
- Final progression: 3 → 5 → 6 → 6 → 7 across 5 rounds
