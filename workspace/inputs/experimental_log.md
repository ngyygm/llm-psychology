# Experimental Log

## 1. Experimental Setup

### Instruments

We administered a **221-item psychometric battery** comprising four validated instruments:

| Instrument | Items | Format | Domains | Reverse items | Source |
|---|---|---|---|---|---|
| IPIP-NEO-120 | 120 | Likert 1-5 | 5 Big Five × 6 facets | 41 | Johnson (2014), public domain |
| SD3 (Short Dark Triad) | 27 | Likert 1-5 | 3 (Machiavellianism, Narcissism, Psychopathy) | 5 | Jones & Paulhus (2014) |
| ZKPQ-50-CC | 50 | True/False | 5 (Activity, Aggression, Impulsive-SS, Neuroticism-Anxiety, Sociability) | 12 | Aluja et al. (2006) |
| EPQR-A | 24 | Yes/No | 4 (Psychoticism, Extraversion, Neuroticism, Lie) | 5 | Francis et al. (1992) |

Total: **221 items, 17 domain scores, 63 reverse-coded items (29%)**.

### Models

18 LLMs tested:

| Model family | Models |
|---|---|
| Anthropic | Claude-Opus-4.6, Claude-Sonnet-4.6 |
| OpenAI | GPT-5.5, GPT-5.2 |
| DeepSeek | DeepSeek-V3.2, DeepSeek-V4-Flash, DeepSeek-V4-Pro |
| Google | Gemini-3-Flash-Preview, Gemini-3.1-Flash-Lite, Gemini-3.1-Pro-Preview, Gemini-3-Pro-Preview |
| Alibaba | Qwen3-235B-A22B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B |
| Moonshot | Kimi-K2.5, Kimi-K2.6 |
| MiniMax | MiniMax-M2.7 |
| Zhipu | GLM-4.6V |

### Conditions

Each model was tested under **17 conditions**:
- **Default**: "Please answer the following personality questionnaire honestly."
- **16 MBTI persona prompts**: Each assigned an MBTI type (e.g., "You are an INTJ — introverted, intuitive, thinking, judging. Answer as this personality type would.")

### Administration protocol

- Temperature = 0.7
- Single administration per model × condition
- Items presented sequentially in a single prompt
- Total observations: 18 models × 17 conditions × 221 items = **67,626 data points**

### Metrics

1. **Cronbach's alpha** — internal consistency per IPIP domain (computed per model across 17 personas, then averaged)
2. **Exploratory Factor Analysis** — Kaiser criterion + parallel analysis on 17 domain scores across 306 model×persona observations
3. **Pairwise Inconsistency Rate (PIR)** — proportion of forward-reverse item pairs showing inconsistent raw responses
4. **Variance decomposition** — sequential sum-of-squares: Model × Domain × Persona × Item
5. **Convergent validity** — Pearson/Spearman correlation between theoretically overlapping constructs across instruments
6. **Measurement invariance** — Pearson r between Default and each MBTI persona item-level profile
7. **Social Desirability Response (SDR)** — composite index based on socially desirable direction per domain
8. **DIF** — Differential Item Functioning between Chinese and Western model families

## 2. Raw Numeric Data

### Table 1: Cronbach's Alpha by IPIP Domain (LLM vs Human)

| Domain | LLM α (mean±SD) | Human α | Gap |
|---|---|---|---|
| Neuroticism | 0.354 ± 0.397 | 0.90 | -0.546 |
| Extraversion | 0.360 ± 0.451 | 0.89 | -0.530 |
| Openness | 0.055 ± 0.093 | 0.87 | -0.815 |
| Agreeableness | -0.017 ± 0.230 | 0.88 | -0.897 |
| Conscientiousness | 0.181 ± 0.258 | 0.90 | -0.719 |

### Table 2: Factor Structure — Eigenvalues (domain-level EFA)

| Factor | Eigenvalue | Cumulative % |
|---|---|---|
| F1 | 6.804 | 40.0% |
| F2 | 4.329 | 65.5% |
| F3 | 2.986 | 83.1% |
| F4 | 0.820 | 87.9% |
| F5 | 0.523 | 91.0% |

Kaiser criterion (eigenvalue > 1.0): **3 factors** (expected: 5+).

### Table 3: Factor Loadings (3-factor solution, Varimax rotation)

| Domain | F1 (Extraversion/Activity) | F2 (Neuroticism/Agreeableness) | F3 (Impulsivity/Conscientiousness) |
|---|---|---|---|
| IPIP Extraversion | **0.94** | 0.03 | 0.29 |
| IPIP Agreeableness | 0.05 | **0.90** | -0.22 |
| IPIP Conscientiousness | 0.00 | -0.04 | **-0.91** |
| IPIP Neuroticism | -0.30 | **0.72** | 0.39 |
| IPIP Openness | -0.04 | 0.55 | 0.58 |
| SD3 Narcissism | **0.91** | -0.20 | 0.17 |
| SD3 Machiavellianism | -0.09 | **-0.82** | 0.05 |
| SD3 Psychopathy | 0.38 | -0.59 | 0.63 |

### Table 4: Pairwise Inconsistency Rate (PIR) by Domain

| Domain | PIR (mean) | Direction |
|---|---|---|
| SD3 Narcissism | 0.929 | Highest inconsistency |
| IPIP Extraversion | 0.878 | |
| IPIP Openness | 0.729 | |
| ZKPQ Aggression-Hostility | 0.580 | |
| IPIP Neuroticism | 0.580 | |
| Overall (weighted) | **0.584** | 58.4% of pairs inconsistent |
| IPIP Agreeableness | 0.426 | |
| IPIP Conscientiousness | 0.348 | |
| SD3 Psychopathy | 0.294 | |
| ZKPQ Neuroticism-Anxiety | 0.278 | Lowest inconsistency |

**Bootstrap 95% CI for PIR**: [0.535, 0.636]

### Table 5: Variance Decomposition

**Likert scales (IPIP + SD3)**:

| Source | Variance % |
|---|---|
| **Model** (inter-model) | **0.3%** [95% CI: 1.1%, 4.0%] |
| Domain (trait) | 23.4% |
| Persona (condition) | 3.9% |
| Item (stimulus) | 35.6% |
| Residual | 36.7% |

**Binary scales (ZKPQ + EPQR)**:

| Source | Variance % |
|---|---|
| **Model** (inter-model) | **0.5%** |
| Domain (trait) | 5.2% |
| Persona (condition) | 13.0% |
| Item (stimulus) | 14.1% |
| Residual | 67.3% |

### Table 6: Convergent Validity (cross-scale correlations)

| Construct pair | Expected | Observed r (Spearman) | p | Sign match |
|---|---|---|---|---|
| Neuroticism ↔ Neuroticism-Anxiety | positive | 0.597 | 0.009 | Yes |
| Extraversion ↔ Sociability | positive | **-0.090** | 0.722 | **No** |
| Extraversion ↔ Extraversion | positive | 0.298 | 0.230 | Yes |
| Neuroticism ↔ Neuroticism | positive | 0.387 | 0.113 | Yes |
| Agreeableness ↔ Machiavellianism | negative | -0.674 | 0.002 | Yes |
| Agreeableness ↔ Psychopathy | negative | **-0.743** | <0.001 | Yes |
| Conscientiousness ↔ Psychopathy | negative | -0.646 | 0.004 | Yes |
| Psychoticism ↔ Psychopathy | positive | 0.205 | 0.414 | Yes |

7/8 sign matches (87.5%), but magnitudes substantially weaker than human baselines.

### Table 7: Synthetic Baselines Comparison

| Strategy | N α | E α | O α | A α | C α |
|---|---|---|---|---|---|
| Random | 0.014 | 0.094 | 0.029 | -0.059 | 0.011 |
| Pure Acquiescence | -0.063 | -0.151 | 0.039 | 0.130 | 0.049 |
| Trait+Acquiescence | 0.997 | 0.998 | 0.996 | 0.997 | 0.997 |
| **LLM Observed** | **0.354** | **0.360** | **0.055** | **-0.017** | **0.181** |
| Human norms | 0.90 | 0.89 | 0.87 | 0.88 | 0.90 |

LLM alphas are closest to Random/Pure Acquiescence, not Trait+Acquiescence or Human.

### Table 8: Leave-One-Model-Out and Leave-One-Persona-Out Robustness

**Leave-one-model-out** (18 analyses):

| Metric | Result |
|---|---|
| Factors (Kaiser) | 3 in ALL 18 analyses (range: [3, 3]) |
| Mean eigenvalue diff from full | 0.012 |
| Conclusion | **ROBUST** — 3-factor structure is universal |

**Leave-one-persona-out** (17 analyses):

| Metric | Result |
|---|---|
| Factors (Kaiser) | 3 in ALL 17 analyses (range: [3, 3]) |
| Conclusion | **ROBUST** — no single persona drives the collapse |

### Table 9: Acquiescence Mechanism

| Model | Forward agree rate | Reverse agree rate | Acquiescence gap |
|---|---|---|---|
| Qwen3-235B-A22B | 0.823 | 0.366 | 1.695 |
| DeepSeek-V4-Flash | 0.772 | 0.366 | 1.580 |
| Gemini-3.1-Flash-Lite | 0.937 | 0.634 | 1.122 |
| DeepSeek-V3.2 | 0.949 | 0.585 | 1.018 |
| Claude-Opus-4.6 | 0.709 | 0.293 | 1.126 |
| Kimi-K2.6 | 0.570 | 0.415 | 0.382 |

**Correlation: Reverse-agree rate × PIR**: r = 0.726 — models that agree more with reverse items show higher inconsistency.

### Table 10: DIF — Chinese vs Western Model Families

| Domain | p-value | Cohen's d |
|---|---|---|
| All 8 tested domains | > 0.05 (all) | < 0.20 (all) |

No statistically significant differences. Chinese and Western LLMs are psychometrically indistinguishable.

## 3. Qualitative Observations

* Observation: The 3-factor collapse is the strongest evidence — Big Five domains merge into "Extraversion/Sociability", "Emotional Stability/Agreeableness", and "Conscientiousness/Impulsivity" clusters, suggesting a fundamentally different response organization than human trait structure.

* Observation: Agreeableness alpha = -0.017 (negative reliability) means that on average, items within this domain are **negatively correlated** after scoring. This is consistent with the 71% reverse-item rate in this domain — acquiescence causes forward and reverse items to pull in opposite directions.

* Observation: The Extraversion × Sociability convergent validity pair **reverses sign** (r = -0.09 instead of expected r ≈ +0.65). This is a signature failure — two instruments measuring the same construct produce opposite results in LLMs.

* Observation: Persona injection improves internal consistency (Default PIR agreement: 0.667 → MBTI PIR agreement: 0.830, t = 8.82, p < 0.0001). Giving models a specific "role" reduces the tension between safety alignment and honest responding.

* Observation: Social desirability correlates negatively with PIR (r = -0.562). Models that lean into socially desirable responses are MORE internally consistent — the "always be good" heuristic is simple and self-consistent, even though it violates psychometric validity.

* Observation: 26/147 Likert items have variance < 0.3 (all models converge on similar answers). These items have lost measurement sensitivity due to alignment training.

* Observation: Tucker's φ = 0.990 across all 18 models for the 3-factor structure — the collapse is universal, not driven by any specific model family or training approach.

* Observation: The synthetic baselines show LLM alpha values are closest to Random and Pure Acquiescence strategies, far from the Trait+Acquiescence strategy that mimics genuine trait structure.

## 4. Data Files

All raw results stored in `results/exp_mbti_*.json` (18 files).
All analysis outputs in `analysis_output/` (32 CSV files).
All figures in `figures/` (8 PNG files).
Full analysis code: `psychometric_analysis.py`, `generate_figures.py`, `round2_fixes.py`, `round3_fixes.py`, `round4_fixes.py`.
