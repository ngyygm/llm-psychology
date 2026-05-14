# Deep Psychometric Validation Results Summary

## Overview
- **18 LLMs** × **17 conditions** (Default + 16 MBTI personas) × **221 items** = **67,626 data points**
- **4 validated psychometric scales**: IPIP-NEO-120 (Big Five), SD3 (Dark Triad), ZKPQ-50-CC, EPQR-A
- **3 theoretical frameworks**: Measurement Invariance, Response Style Theory, Social Desirability Bias

---

## Finding 1: Factor Structure Collapse ( strongest evidence)

**Method**: Exploratory Factor Analysis on 17 domain scores across 306 model×persona observations.

**Results**:
- **Kaiser criterion**: 3 factors (expected: 5+)
- **Parallel analysis**: 3 factors (expected: 5+)
- **3 factors explain 83.1%** of total variance (first 3 eigenvalues: 6.80, 4.33, 2.99)
- **Eigenvalue ratio (1st/2nd)**: 1.57 (below 3.0 threshold for dominant general factor)

**Factor loading pattern (3-factor solution)**:
| Domain | F1 (Extraversion/Activity) | F2 (Neuroticism/Agreeableness) | F3 (Impulsivity/Conscientiousness) |
|--------|:--:|:--:|:--:|
| IPIP Extraversion | **0.94** | 0.03 | 0.29 |
| IPIP Agreeableness | 0.05 | **0.90** | -0.22 |
| IPIP Conscientiousness | 0.00 | -0.04 | **-0.91** |
| IPIP Neuroticism | -0.30 | **0.72** | 0.39 |
| IPIP Openness | -0.04 | 0.55 | 0.58 |
| SD3 Narcissism | **0.91** | -0.20 | 0.17 |
| SD3 Machiavellianism | -0.09 | **-0.82** | 0.05 |
| SD3 Psychopathy | 0.38 | -0.59 | 0.63 |

**Key insight**: Big Five domains COLLAPSE from 5 to 3 factors:
- **Factor 1** (Extraversion/Sociability): merges IPIP Extraversion, SD3 Narcissism, ZKPQ Activity/Sociability, EPQR Extraversion
- **Factor 2** (Emotional Stability/Agreeableness): merges IPIP Neuroticism, IPIP Agreeableness, SD3 Machiavellianism (negative), ZKPQ Neuroticism-Anxiety, EPQR Neuroticism
- **Factor 3** (Conscientiousness/Impulsivity): merges IPIP Conscientiousness (negative), EPQR Psychoticism, ZKPQ Impulsive Sensation Seeking

**Interpretation**: The human Big Five structure does NOT replicate in LLM data. Neuroticism and Agreeableness share a single factor (suggesting a "socially desirable emotional stability" dimension), while Openness and Conscientiousness collapse into an impulsivity factor. This is the strongest evidence that LLM personality measurement captures **alignment-driven response patterns** rather than human-like trait structure.

**Per-model stability**: Tucker's φ = 0.990 (all models show identical factor structure). The 3-factor collapse is universal across all 18 models.

---

## Finding 2: Reverse-Item Inconsistency (PIR = 0.584)

**Method**: Pairwise Inconsistency Rate — proportion of forward-reverse item pairs where the model gives the same raw response direction.

**Results**:
- **Mean PIR**: 0.584 (58.4% of forward-reverse pairs show inconsistent responses)
- **Highest inconsistency domains**: SD3 Narcissism (0.929), IPIP Extraversion (0.878), IPIP Openness (0.729)
- **Lowest inconsistency domains**: ZKPQ Neuroticism-Anxiety (0.278), SD3 Psychopathy (0.294), IPIP Conscientiousness (0.348)
- **PIR variance decomposition**: Model = 11.6%, Domain = 40.4%, Residual = 48.0%

**PIR × SDR Cross-validation**:
- Spearman r(PIR, SDR) = **-0.562** (p = 0.015)
- **Surprising finding**: Models with HIGHER social desirability have LOWER inconsistency
- **Interpretation**: Social desirability acts as a **stabilizing force** — models that respond in the "good" direction do so consistently, even for reverse-coded items. This is the "good person syndrome": answering "agree" to everything (including reverse items that should be "disagree") creates internal consistency within the acquiescence pattern, even though it violates psychometric validity.

**MBTI Persona effect on PIR**:
- Default PIR agreement: 0.667 → MBTI persona PIR agreement: 0.830 (t = 8.82, p < 0.0001)
- Persona injection IMPROVES internal consistency (reduces inconsistency)
- This suggests persona prompts give the model a coherent "role" to play, reducing the acquiescence-driven inconsistency

---

## Finding 3: Variance Decomposition (Model = 0.3%)

**Method**: Sequential sum-of-squares decomposition of normalized response scores.

**Likert scales (IPIP + SD3)**:
| Source | Variance % |
|--------|:---------:|
| **Model** (machine personality) | **0.3%** |
| Domain (trait) | 23.4% |
| Persona (condition) | 3.9% |
| Item (stimulus) | 35.6% |
| Residual | 36.7% |

**Binary scales (ZKPQ + EPQR)**:
| Source | Variance % |
|--------|:---------:|
| **Model** (machine personality) | **0.5%** |
| Domain (trait) | 5.2% |
| Persona (condition) | 13.0% |
| Item (stimulus) | 14.1% |
| Residual | 67.3% |

**Interpretation**: Inter-model variance ("machine personality") explains less than 1% of total response variance. This means:
1. Different LLMs are **more similar than different** in their personality profiles
2. The dominant source of variance is **item-specific** (35.6% for Likert) — the specific wording of each question drives the response more than any trait-like pattern
3. This severely undermines the claim that personality tests measure meaningful differences between LLMs

---

## Finding 4: DIF and Convergent Validity

**DIF (Chinese vs Western models)**:
- **0 out of 8 domains** show statistically significant differences (all p > 0.05)
- All Cohen's d values < 0.20 (negligible effect sizes)
- Chinese and Western LLMs are statistically indistinguishable in their personality profiles
- **Interpretation**: Alignment training creates a universal "personality" that transcends cultural origin

**Item Variance Collapse**:
- 26/147 Likert items have variance < 0.3 (all models converge on similar answers)
- 8 items cluster at high agreement (mean > 3.5), 8 at low agreement (mean < 2.5)
- These items have lost measurement sensitivity

**Convergent Validity**: 7/8 sign matches (87.5%)
- Strongest convergence: IPIP Agreeableness × SD3 Psychopathy (r = -0.73, p < 0.001)
- Weakest convergence: IPIP Extraversion × ZKPQ Sociability (r = -0.03, p = 0.91) — **sign reversal**
- The convergent validity is directionally correct but much weaker than human baselines (typically r > 0.5 for overlapping constructs)

---

## Finding 5: Measurement Invariance (r = 0.437)

**Method**: Pearson correlation between Default persona item-level profile and each MBTI persona profile.

**Results**:
- **Mean r(Default, MBTI)**: 0.437 (moderate at best)
- **r > 0.8 (strong invariance)**: 0 out of 288 model×persona pairs
- **r > 0.5 (moderate invariance)**: 48 out of 288 pairs (16.7%)
- **r < 0.3 (weak invariance)**: ~30% of pairs

**Most invariant personas** (maintain Default structure):
1. ENFJ (r = 0.608) — "teacher" persona close to default aligned behavior
2. INFJ (r = 0.557)
3. ISFJ/ESFJ (r ≈ 0.528)

**Least invariant personas** (fundamentally restructure):
1. ESTP (r = 0.206) — "adventurer" persona conflicts with alignment
2. ENTP (r = 0.291) — "debater" persona conflicts with agreeableness
3. ESFP (r = 0.331)

**Interpretation**: Persona injection does NOT simply shift response baselines — it fundamentally restructures the response pattern. This means scalar invariance fails, and personality scores under different persona conditions are not directly comparable. The more a persona conflicts with alignment (e.g., ESTP/ENTP — extraverted, perceiving types that might act antisocially), the lower the invariance.

---

## Theoretical Synthesis

### The "Alignment Personality" Hypothesis
The data supports a unified explanation:

1. **Alignment creates a single dominant dimension**: The 3-factor collapse shows that LLM personality is structured around "how aligned is this response?" rather than human-like trait variation.

2. **Social desirability stabilizes inconsistency**: The negative PIR×SDR correlation (-0.562) shows that models which lean into socially desirable responses are MORE internally consistent — not because they have genuine personality structure, but because the "always agree, always be good" heuristic is simple and consistent.

3. **Machine personality is negligible**: The 0.3% model variance means that personality differences between models are effectively noise. All aligned LLMs converge on a similar "moderate prosocial" profile.

4. **Persona injection partially overrides alignment**: The improved consistency under persona conditions (0.667 → 0.830) suggests that giving the model a specific role reduces the tension between "be helpful/harmless" and "answer honestly." The model can be consistent within its role.

### Implications for LLM Personality Research
1. **Human psychometric tools require revalidation** before being applied to LLMs. The Big Five structure does not replicate.
2. **Reverse-coded items are contaminated** by acquiescence bias — any scale using reverse items will show artificially high "consistency" if the model simply agrees with everything.
3. **Cross-model comparisons are not meaningful** when inter-model variance is <1%. Differences are better explained by response style than by genuine trait variation.
4. **Persona-based studies must report invariance tests** — simply showing that a persona "works" (produces expected directional patterns) is insufficient without demonstrating measurement invariance.

---

## Generated Outputs

### Analysis Scripts
- `psychometric_analysis.py` — Full 6-analysis pipeline
- `generate_figures.py` — 8 publication-quality figures

### Data Files (analysis_output/)
- `efa_domain_loadings.csv` — Factor loadings
- `efa_eigenvalues.csv` — Eigenvalue spectrum
- `efa_per_model_congruence.csv` — Per-model Tucker's phi
- `pir_by_model_domain.csv` — PIR per model×domain
- `pir_by_persona.csv` — PIR per model×persona
- `pir_sdr_crossvalidation.csv` — PIR × SDR correlation
- `sdr_by_model.csv` — Social desirability indices
- `variance_decomposition.csv` — Full variance decomposition
- `dif_model_family.csv` — Chinese vs Western DIF
- `convergent_validity_enhanced.csv` — Cross-scale convergence
- `response_styles.csv` — Response style quantification
- `persona_invariance.csv` — Profile-level invariance
- `item_variance_analysis.csv` — Item-level variance collapse
- `psychometric_validation_report.txt` — Summary report

### Figures (figures/)
1. `fig1_scree_loadings.png` — Scree plot + factor loading heatmap
2. `fig2_pir_sdr.png` — PIR by domain + PIR×SDR scatter
3. `fig3_variance_decomposition.png` — Stacked bar chart
4. `fig4_measurement_invariance.png` — Invariance heatmap + persona ranking
5. `fig5_response_styles.png` — Response style scatter plots
6. `fig6_convergent_validity.png` — Forest plot of convergent validity
7. `fig7_factor_collapse.png` — Expected vs observed factor structure
8. `fig8_model_dashboard.png` — Per-model summary dashboard
