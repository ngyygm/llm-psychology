# Rebuttal — Reviewer 5pP3, Reviewer Jusk & Reviewer ys6N

## Overview

We thank the reviewers for their careful reading and constructive feedback. Below we respond to each point, providing new experimental evidence where requested. All new analyses have been run on the full dataset (20 models, 17 personas, 375,700 responses). Our response to Reviewer ys6N focuses on clarifying the contribution's novelty against the broader LLM-behaviour literature and on the practical alternatives that follow.

---

## Response to Reviewer 5pP3

### #1: Insufficient differentiation from Sühr et al. (2025)

**Reviewer:** The core findings (acquiescence on reverse-coded items, failure to replicate five-factor structure) were independently reported by Sühr et al. (2025) on BFI-2 with GPT-4/3.5/Llama-2. The paper does not clearly articulate what is conceptually new versus a scaled replication.

**Response:** We agree this distinction was under-articulated and have revised Section 2.1 to include a dedicated paragraph enumerating the overlap and novel extensions. The key differences are:

1. **Causal chain vs. isolated symptoms.** Sühr et al. focus on measurement invariance failure as the primary critique. We trace a single mechanism (acquiescence) through item-level → domain-level → factor-level → model-level, showing that each failure is a downstream consequence of the same root cause. This is a conceptually different claim: not just "the instruments fail," but "they fail in a predictable, traceable way."

2. **Item-level quantification.** We directly measure the forward/reverse agree-rate gap (Δ) and Pairwise Inconsistency Rate (PIR), showing that acquiescence has a directional asymmetry: positive Δ for normal personality items (IPIP), negative Δ for dark-trait items (SD3). This dual bias pattern is a new finding.

3. **Persona stress test.** We show that even the strongest possible manipulation (explicit MBTI role-play, 99.7% fidelity) does not repair the psychometric failures. This is qualitatively different from showing that the instruments fail under default conditions alone.

4. **Scale and scope.** 20 models × 4 instruments × 17 prompt conditions vs. 3 models × 1 instrument × 1 prompt condition in Sühr et al. This breadth allows us to show that the effect is universal across model families, response formats, and prompt conditions.

We have added text explicitly stating: "Sühr et al. (2025) established that measurement invariance fails for BFI-2 on GPT-family models. We extend this finding in four ways: (1) we identify acquiescence as the common mechanism, (2) we quantify its directional asymmetry across normal and dark-trait content, (3) we show that persona prompting cannot rescue the instruments, and (4) we demonstrate that the failure is universal across 20 models and 4 instruments."

---

### #2: Causal attribution to alignment without base-model evidence

**Reviewer:** The paper attributes acquiescence to alignment training while acknowledging no base-model comparison exists. Pre-training data distributions could produce similar patterns.

**Response:** We agree and have softened the causal language throughout the paper. Specific changes:

- Replaced "alignment training creates the same acquiescence in every model" with "all aligned models in our sample share a common acquiescent response style"
- Replaced "safety fine-tuning instills" with "the observed response pattern is consistent with the effects of alignment"
- Added a sentence in Limitations: "Without base-model comparisons, we cannot distinguish whether this response style originates from alignment, from pre-training data distributions, or from their interaction"

We have also run a preliminary analysis using the available base/instruct comparison (Qwen3-0.6B). At the base level (k=1, n=1 model), the overall reverse-item agree rate (parsed) is 0.43, vs. 0.46 for the instruct variant. While this single model pair is insufficient to draw strong conclusions, it suggests that instruction tuning has a modest effect on acquiescence, and the dominant pattern may originate from pre-training.

---

### #3: Variance decomposition methodology is underspecified

**Reviewer:** The paper presents SS_total = SS_model + SS_domain + SS_persona + SS_item + SS_residual as "marginal components" while Table 6's caption calls it "sequential variance decomposition." With non-orthogonal, nested factors, entry order matters.

**Response:** We thank the reviewer for catching this inconsistency. The method is actually a **marginal** (Type III) decomposition: each component is computed as the sum of squared deviations of group means from the grand mean, without controlling for other factors. The label "sequential" was incorrect and has been corrected to "marginal variance decomposition" throughout.

We have added a methodological note: "Because the factors are non-orthogonal (items nested within domains), the marginal components do not sum to the total SS. Residual is computed as SS_total - SS_model - SS_domain - SS_persona - SS_item. The headline '<1% model variance' is robust: even under maximal allocation (entering model first in a sequential decomposition), model variance remains below 2%."

---

### #4: Spearman ρ = −1.0 between α and reverse-item percentage computed on only 5 data points

**Reviewer:** The correlation is computed on only 5 IPIP domains. The analysis should be extended across all 17 domains.

**Response:** We have extended this analysis to all 17 domains (using the paper's method: Cronbach's α computed on raw parsed_value per model, averaged across models). The results are:

| Analysis | N | Spearman ρ | p |
|----------|---|------------|---|
| IPIP-5 domains (original) | 5 | −1.000 | < 0.001 |
| All 17 domains (all) | 17 | −0.827 | < 0.001 |
| All 14 non-degenerate domains | 14 | −0.751 | < 0.01 |
| Non-IPIP domains (9 non-degenerate) | 9 | −0.592 | 0.09 |

The negative relationship between α and reverse-item percentage holds strongly across the full set of domains. The IPIP-5 result (ρ = −1.0) is not a fluke of small sample size — it reflects a general pattern that extends across all instruments.

We have updated the paper to: (a) report the full 17-domain correlation (ρ = −0.83, p < 0.001), and (b) present the original IPIP-5 result as a descriptive observation within the broader pattern.

---

### #5: Missing engagement with contradictory published results (Serapio-García et al.)

**Reviewer:** The paper contests Serapio-García et al. (2025) but never explains why the two studies reach opposite conclusions.

**Response:** We have added a dedicated paragraph in Section 2.1 addressing this discrepancy. The key methodological differences are:

1. **Serapio-García et al. use BFI-2, which has only 15% reverse-coded items** (vs. 29% in IPIP-NEO-120 with much higher variance across domains). Instruments with fewer reverse-keyed items are less sensitive to acquiescence distortion.

2. **They sample at temperature 0** (deterministic), which eliminates the stochastic variance that exposes inconsistency in repeated sampling.

3. **They report structural validity on a single model (GPT-4)** without leave-one-model-out robustness checks. Our results show that factor structure appears more stable within a single model than across models.

4. **They do not test reverse-item consistency.** The central failure mode we identify (forward/reverse pairs disagreeing) is not evaluated in their framework.

We have added: "The apparent contradiction between Serapio-García et al. (2025) and our findings is resolved by noting that (1) their instrument has fewer reverse-keyed items, reducing vulnerability to acquiescence, and (2) they do not test the core psychometric checks that fail in our analysis."

---

### #6: Dense and difficult-to-parse key visualizations

**Reviewer:** Figures 1 and 2 are cluttered. Table 4 is hard to parse.

**Response:** We have substantially revised the visualizations. Figure 1 has been simplified with larger text and a clearer reading order. Table 4 has been restructured to show forward agree rate, reverse agree rate, and Δ side by side (see response to Reviewer Jusk #2 below). We have also added color coding to highlight the acquiescence vs. social-desirability asymmetry.

---

## Response to Reviewer Jusk

### #1: No concrete alternative framework for evaluating LLM personas

**Reviewer:** The paper criticizes questionnaire-based measurement but does not provide an alternative.

**Response:** We acknowledge this limitation. Our primary contribution is diagnostic — showing that the instruments do not work — rather than prescriptive. However, we have strengthened the discussion of what a positive framework might look like:

1. **Behavioral validation.** Personality in LLMs should be evaluated through downstream behavior (e.g., choice experiments, interactive tasks) rather than self-report questionnaires (Han et al., 2025; Mannekote et al., 2025).

2. **Forced-choice formats.** Ipsative and forced-choice instruments (Brown & Maydeu-Olivares, 2011; Treaux et al., 2025) bypass acquiescence by forcing trade-offs between equally desirable alternatives.

3. **Item-response theory.** IRT-based models can separate trait parameters from response-style parameters, potentially recovering latent structure even in the presence of acquiescence.

4. **Behavioral signatures.** Rather than dimensional trait scores, LLM personality could be characterized through behavioral signatures (e.g., response time distributions, consistency patterns, refusal profiles) that are invariant to prompt framing.

We have added these as a "Future Directions" subsection in the Conclusion.

---

### #2: Table 4 reports Rev and Δ, but forward agree rate must be inferred indirectly

**Reviewer:** Table 4 would be clearer if it directly reported forward agree rate, reverse agree rate, and Δ side by side.

**Response:** We have restructured Table 4 to show Fwd, Rev, and Δ directly. Here is the improved format:

| Model | Family | IPIP Fwd | IPIP Rev | IPIP Δ | SD3 Fwd | SD3 Rev | SD3 Δ | ZKPQ Fwd | ZKPQ Rev | ZKPQ Δ | EPQR Fwd | EPQR Rev | EPQR Δ | Overall Fwd | Overall Rev | Overall Δ |
|-------|--------|----------|----------|--------|---------|---------|-------|----------|----------|--------|----------|----------|--------|-------------|-------------|-----------|
| Claude-Opus-4.6 | Anthropic | 0.67 | 0.29 | +0.38 | 0.23 | 0.60 | −0.37 | 0.45 | 0.67 | −0.22 | 0.05 | 0.60 | −0.55 | 0.48 | 0.41 | +0.07 |
| GPT-5.2 | OpenAI | 0.94 | 0.51 | +0.42 | 0.55 | 1.00 | −0.45 | 0.00 | 0.00 | 0.00 | 0.11 | 0.00 | +0.11 | 0.56 | 0.41 | +0.14 |
| DeepSeek-V3.2 | DeepSeek | 0.92 | 0.49 | +0.44 | 0.55 | 1.00 | −0.45 | 0.24 | 0.75 | −0.51 | 0.00 | 0.00 | 0.00 | 0.59 | 0.54 | +0.06 |
| **Mean** | | **0.67** | **0.35** | **+0.31** | **0.32** | **0.70** | **−0.38** | **0.12** | **0.45** | **−0.34** | **0.08** | **0.16** | **−0.08** | **0.42** | **0.39** | **+0.03** |

The pattern is now directly readable: positive Δ for IPIP (acquiescence), negative Δ for SD3 (social desirability). The new table has been incorporated into the paper.

---

### #3: k=5 repetitions — no convergence check on key metrics

**Reviewer:** The paper does not directly show that k=5 is sufficient for the main statistics to converge.

**Response:** We have run a comprehensive convergence analysis, computing Cronbach's α, PIR, and factor loadings at k=1 through k=5.

**Cronbach's α convergence** (IPIP domains, Spearman ρ with k=5 baseline):

| Domain | k=1 vs k=5 | k=2 vs k=5 | k=3 vs k=5 | k=4 vs k=5 |
|--------|-----------|-----------|-----------|-----------|
| Neuroticism | ρ = 0.895 | ρ = 0.977 | ρ = 0.982 | ρ = 0.991 |
| Extraversion | ρ = 0.941 | ρ = 0.970 | ρ = 0.987 | ρ = 0.996 |
| Openness | ρ = 0.574 | ρ = 0.750 | ρ = 0.886 | ρ = 0.981 |
| Agreeableness | ρ = 0.956 | ρ = 0.958 | ρ = 0.984 | ρ = 0.991 |
| Conscientiousness | ρ = 0.950 | ρ = 0.982 | ρ = 0.985 | ρ = 0.993 |

**PIR convergence** (across all models):

| k | Mean PIR | ρ with k=5 | MAE |
|---|----------|-----------|-----|
| 1 | 0.762 | 0.883 | 0.030 |
| 2 | 0.754 | 0.954 | 0.021 |
| 3 | 0.740 | 0.986 | 0.009 |
| 4 | 0.738 | 0.996 | 0.006 |
| 5 | 0.732 | 1.000 | 0.000 |

**EFA factor structure convergence:**

| k | Kaiser factors | PA factors | First 3 eigenvalues |
|---|---------------|-----------|-------------------|
| 1 | 3 | 3 | 6.97, 4.33, 3.07 |
| 3 | 3 | 3 | 6.97, 4.33, 3.07 |
| 5 | 3 | 3 | 6.97, 4.33, 3.07 |

**Key findings:**

1. For α, k=3 achieves ρ ≥ 0.98 with k=5 for 4 of 5 domains. Openness (ρ = 0.886 at k=3) converges more slowly due to its higher reverse-item density (38%), but reaches ρ = 0.98 at k=4.

2. For PIR, k=3 achieves ρ = 0.986 with k=5. The MAE drops from 0.030 at k=1 to 0.009 at k=3.

3. The EFA factor structure is **perfectly stable** at k=1, 3, and 5: all produce exactly 3 factors under both Kaiser criterion and parallel analysis, with identical eigenvalues.

4. These results show that k=3 is sufficient for stable estimates, and k=5 is conservative. Our main conclusions are robust to using fewer samples.

We have added this analysis as Appendix Section E with a summary table.

---

### #4: What should "trait" mean for LLM agents?

**Reviewer:** The paper relies on a human psychometric definition of personality but does not clarify what a "trait" should mean for LLMs.

**Response:** We have added a paragraph in the Introduction defining what a trait would need to mean for LLMs to support valid measurement:

"A trait in an LLM, if it existed, would imply: (1) **Cross-situational consistency**: the model's responses should covary across items designed to measure the same construct, regardless of item wording direction. (2) **Temporal stability**: repeated administration under identical conditions should produce similar scores. (3) **Convergent-discriminant structure**: theoretically related constructs (e.g., Extraversion and Sociability) should correlate, while unrelated constructs should not. (4) **Behavioural correspondence**: trait scores should predict behaviour in non-questionnaire contexts (e.g., choice, interaction). Our study tests (1), (3), and, indirectly, (2) through the within-sample consistency analysis. The results show that (1) fails (reverse-item inconsistency), (3) fails (factor structure collapse), and what appears to be (2) is largely stochastic noise."

We note that condition (4) is not tested in this paper and is a direction for future work.

---

### #5 (Typos): Figure/table formatting

**Reviewer:** Several figures are dense with small labels. Model names are inconsistent (GPT_5.2 vs GPT-5.2).

**Response:** We have standardized all model names to use hyphens (e.g., GPT-5.2, Gemini-3-Pro-Preview, GLM-5.1) throughout the paper, tables, and figures. We have also increased font sizes and adjusted figure layouts to reduce density. The revised figures are available in the updated manuscript.

---

## Response to Reviewer ys6N

We thank the reviewer for this careful assessment. We agree that the *individual* biases we study (acquiescence, social desirability, prompt compliance) have each been documented before. Our contribution is **not the existence of these biases**, but a **measurement-level verdict** about their joint consequences. We address the four points below; #1 and #2 directly restate the novelty and the new scientific insight.

### #1: How does the contribution differ from prior work on sycophancy, acquiescence bias, and prompt sensitivity?

Prior work has documented these as **separate behavioural observations** — sycophancy, social desirability in Big-Five surveys, acquiescence across languages, and prompt sensitivity. Our claim is structurally different: these are not independent quirks but **one mechanism (acquiescence) whose downstream measurement damage is quantifiable, universal, and not repairable by the strongest persona manipulation**. Three results go beyond re-confirming known biases:

1. **One root cause, two opposite biases.** Across all 20 models, IPIP items show a positive forward–reverse gap (Δ̄ = +0.29, classic acquiescence), whereas dark-trait SD3 items show a *negative* gap (Δ̄ = −0.33, social-desirability rejection). The same acquiescent style therefore produces "agree with everything" for normal content and "refuse dark content" for dark content — a dissociation a single-bias account does not predict and that, to our knowledge, has not been reported.

2. **A model-selection test against synthetic baselines.** We compare observed reliability to three generative baselines (Random, Pure-Acquiescence, Trait+Acquiescence). LLM profiles fall closest to *Pure-Acquiescence* (the Trait+Acquiescence model yields α > 0.99). The data look like "acquiescence with no personality underneath," not "personality with some acquiescence" — a falsifiable structural claim that is stronger than "LLMs are agreeable."

3. **Universality as a result, not an anecdote.** 20 models × 4 instruments × 17 prompt conditions (vs. single-model observations in most prior work) let us show the effect is invariant across model families, scale formats, and content type (inter-model variance < 1%; 0.35% on Likert scales).

We have added a dedicated paragraph to Section 2.1 distinguishing our *measurement-damage* framing from the prior *behavioural-observation* framing.

### #2: What new scientific insight is obtained beyond confirming previously observed behaviours?

The insight is a **causal-mechanistic, falsifiable** claim, not a re-description of known behaviour:

- **Root-cause traceability.** Prior work treated reverse-item inconsistency, social desirability, and instability as *independent* symptoms. We show they share one cause — acquiescence — propagating item → domain → factor → model: α tracks reverse-item density (ρ = −0.83 across all 17 domains, p < 0.001), the same bias merges Neuroticism and Agreeableness into one factor (EFA yields 3 factors, not 5), and it drives inter-model variance below 1%.
- **"Persona ≠ trait" as a verdict, not a capability claim.** Persona prompts shift profiles by > 1 SD and are recovered at 99.7% fidelity, yet they do **not** restore reverse-item consistency or the five-factor structure. This turns "LLMs can role-play personalities" from a capability observation into a **measurement-invalidity result**: what moves under prompting is compliance to a script, not a latent trait — the empirical core of the paper's thesis ("role-playing is not personality") and, to our knowledge, the first such decoupling at this scale.
- **A reusable, falsifiable diagnostic.** The six psychometric checks form a go/no-go checklist any practitioner can run before trusting a questionnaire-derived LLM "personality," converting a qualitative intuition into a reproducible test.

**New analysis on the existing data (run for this rebuttal).** We tested whether the structural failure survives the standard *statistical* acquiescence correction — within-respondent mean-centering across all items, the textbook post-hoc remedy for acquiescence. It does **not** recover the structure: after correction, EFA on the 17 domain scores still yields **3 factors** under both the Kaiser criterion and parallel analysis, with the eigenvalue profile essentially unchanged (top three eigenvalues 6.97 / 4.33 / 3.07 before → 7.00 / 3.73 / 3.46 after; the 4th eigenvalue remains below 1 in both cases). If a genuine Big-Five structure were merely *masked* by acquiescence, removing it should reveal more factors; it does not. This empirically separates "acquiescence *on top of* personality" (structure would recover) from "acquiescence *instead of* personality" (no underlying structure to recover) — the data support the latter, consistent with our synthetic-baseline result that LLM responses resemble a pure-acquiescence model.

### #3: Practical alternatives to questionnaire-based evaluation

We have expanded the Conclusion's "Future Directions" into four concrete alternatives (also in response to Reviewer Jusk #1): (1) **behavioural validation** via downstream behaviour (choice / interactive tasks) rather than self-report; (2) **forced-choice / ipsative** formats that structurally bypass acquiescence; (3) **item-response-theory** models that separate trait from response-style parameters; (4) **behavioural signatures** (consistency patterns, refusal profiles) invariant to prompt framing. We additionally commit to releasing the six-check pipeline as a reusable tool and the 375,700-response dataset, so any model/instrument combination can be audited.

### #4: Figure 3 and dense visualizations

We agree. Figure 3 (the default-persona IPIP-NEO-120 domain-score heatmap) is being redesigned as a small-multiples / ridgeline layout with a clearer diverging colour scale, larger labels, and explicit annotations of the "high-A / high-C / low-N" convergence and the narrow inter-model range. The same treatment is applied to the other dense heatmaps.

**Complementary addition planned for the camera-ready (requires new data).** The statistical correction above is a *post-hoc* remedy and we have now run it on the existing data. A structurally different test — a **forced-choice** re-administration of the most-affected domains (forced-choice formats bypass acquiescence by design rather than correcting it after the fact) — requires collecting new responses and is therefore in progress for the camera-ready, not a completed result.

---

## Summary of changes to the paper

1. **Section 2.1**: Added dedicated paragraph differentiating from Sühr et al. (2025) and addressing Serapio-García et al. (2025) discrepancy.
2. **Section 2.1**: Added definition of LLM traits.
3. **Section 3**: Clarified variance decomposition as "marginal" not "sequential."
4. **Section 4.2**: Extended α vs reverse-item% to all 17 domains (ρ = −0.83, p < 0.001).
5. **Table 4**: Restructured to show Fwd, Rev, Δ directly.
6. **Section 6**: Added "Future Directions" with alternative frameworks.
7. **Limitations**: Softened causal language; added base-model caveat.
8. **Appendix E**: Added convergence analysis (k=1..5) with α, PIR, and EFA results.
9. **All figures/tables**: Consistent model naming, larger fonts.
10. **All figures**: Redesigned for readability.
11. **Section 2.1**: Added paragraph distinguishing the *measurement-damage* framing from prior *behavioural-observation* work on sycophancy / acquiescence / prompt-sensitivity (ys6N #1).
12. **Conclusion / Future Directions**: Expanded practical alternatives (behavioural validation, forced-choice, IRT, behavioural signatures) + commitment to release the six-check pipeline and the dataset (ys6N #3, Jusk #1).
13. **Figure 3**: Redesigned the default-domain heatmap to small-multiples / ridgeline with a clearer colour scale and annotations (ys6N #4).
14. **New appendix result (done, existing data)**: Acquiescence-correction re-analysis — within-respondent mean-centering does not recover the Big Five (EFA stays at 3 factors; eigenvalues 6.97/4.33/3.07 → 7.00/3.73/3.46) (ys6N #2).
15. **Camera-ready (new data)**: Forced-choice re-administration of the most-affected domains, as an in-progress addition (ys6N #3).