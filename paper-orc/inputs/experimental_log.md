# Experimental Log: Acquiescence Is Not Personality

This log is the authoritative numeric source for the paper. All headline metrics are reproduced from `paper/paper.tex` and the per-analysis CSVs in `code/results/`. Where the paper and an intermediate CSV disagree (see §3 Iteration History), the paper values are canonical.

## 1. Experimental Setup

**Goal.** Apply standard psychometric validation checks to LLM responses on validated human personality instruments, and identify the root cause of any failures observed.

**Battery (221 items, 4 instruments, 17 domains, 63 reverse-keyed items).**

| Instrument | Items | Format | Domains | Reverse | Source |
|---|---:|---|---:|---:|---|
| IPIP-NEO-120 | 120 | Likert-5 | 5 (Big Five) | 41 | Johnson (2014) |
| SD3 (Short Dark Triad) | 27 | Likert-5 | 3 | 5 | Jones & Paulhus (2014) |
| ZKPQ-50-CC | 50 | True / False | 5 | 12 | Aluja et al. (2006) |
| EPQR-A | 24 | Yes / No | 4 | 5 | Francis et al. (1992) |
| **Total** | **221** | 3 formats | **17** | **63** | |

**Models (20 LLMs, 8 families).**

| Family | Models |
|---|---|
| Anthropic | Claude-Opus-4.6, Claude-Sonnet-4.6 |
| OpenAI | GPT-5.2, GPT-5.5 |
| Google | Gemini-3-Pro-Preview, Gemini-3-Flash-Preview, Gemini-3.1-Pro-Preview, Gemini-3.1-Flash-Lite |
| DeepSeek | DeepSeek-V3.2, DeepSeek-V4-Flash, DeepSeek-V4-Pro |
| Alibaba | Qwen3-235B-A22B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B |
| Zhipu | GLM-4.6V, GLM-4.7, GLM-5.1 |
| Moonshot | Kimi-K2.5, Kimi-K2.6 |
| MiniMax | MiniMax-M2.7 |

**Personas (17 conditions).** Default (no-persona instruction: *"Please answer the following personality questionnaire honestly."*) plus all 16 MBTI types: ISTJ, ISFJ, INFJ, INTJ, ISTP, ISFP, INFP, INTP, ESTP, ESFP, ENFP, ENTP, ESTJ, ESFJ, ENFJ, ENTJ.

**Sampling.** k=5 independent samples per (model, persona, item) cell at **temperature = 0.7**. All responses averaged within cells. Yields 20 × 17 × 221 = 75,140 item-level scores from **375,700** raw API calls. Missing data: **61 / 375,700 = 0.0162 %**, all occurring under the Default condition, handled by per-item mean imputation across remaining models and samples under the same persona (immaterial at this rate).

**Analysis pipeline.**
- `code/build_battery.py` — battery construction
- `exp/run_mbti_experiment.py` — experiment runner (prompts, k=5 sampling, telemetry)
- `code/psychometric_analysis.py` — Cronbach's α, EFA, PIR, SDR, variance decomposition
- `code/round2_fixes.py` — acquiescence mechanism, item-level analysis
- `code/round3_fixes.py` — human-norm benchmarks, leave-one-out robustness
- `code/round4_fixes.py` — synthetic baselines, persona leave-one-out
- `code/cross_cutting_analysis.py` — clustering, MBTI factorial, within-family
- `code/persona_steering_analysis.py` — PSD, fidelity matrix, target adherence, item plasticity
- `code/within_sample_consistency.py` — k=5 stability per model/persona/domain
- `code/generate_figures.py` and `code/polish_figures.py` — publication-quality figures

**Six standard psychometric checks applied.**
1. **Internal consistency** — Cronbach's α per IPIP domain (17 personas as observations), averaged across 20 models.
2. **Factor structure** — EFA on standardized 17 domain scores (n = 340 observations), Kaiser criterion + parallel analysis (500 permutations, 95th-percentile cutoff), varimax rotation, replicated leave-one-model-out (20 reruns), per-family (8 reruns), and per-model (20 reruns).
3. **Reverse-item consistency** — Pairwise Inconsistency Rate (PIR) per (model, domain); per-instrument forward / reverse agree-rate decomposition with Δ = Fwd − Rev.
4. **Variance decomposition** — sequential SS into Model + Domain + Persona + Item + Residual, separately for Likert (IPIP+SD3) and binary (ZKPQ+EPQR) scales.
5. **Convergent validity** — Pearson and Spearman across 20 model-level means for 8 pre-specified construct pairs.
6. **Measurement invariance** — Default-vs-MBTI item-vector Pearson correlation across all 20 × 16 = 320 (model, persona) pairs.

**Plus persona stress test.** Persona Separation Degree (normalized 17-d Euclidean distance from Default in z-scored space, divided by √17), nearest-centroid persona classification with leave-one-model-out centroids and metric/scale ablations (z-Euclidean, cosine, Mahalanobis × all/likert/binary/IPIP-only), target adherence (empirical leave-one-out centroid target and hand-coded theoretical MBTI target), sign-aligned cross-scale coherence on overlapping constructs, forward/reverse pair directional coherence, item-level plasticity (mean absolute Default-vs-persona raw-score shift, normalized by scale range), factorial main-effects model for E/I, S/N, T/F, J/P with two-way interactions, and within-sample consistency (SD across k=5 repeats, exact agreement, mode agreement) per (model, persona, domain).

**Calibration baselines.** Three synthetic response strategies on the same IPIP item structure: Random uniform, Pure Acquiescence (always agree), and Trait+Acquiescence (genuine 5-factor structure + acquiescence noise). Cronbach's α computed for each and compared with the LLM observed values.

## 2. Raw Numeric Data

### 2.1 Internal consistency (Cronbach's α per IPIP domain, mean ± SD across 20 models)

| Domain | LLM α | LLM SD | Human α | Gap | Reverse % |
|---|---:|---:|---:|---:|---:|
| Neuroticism | 0.813 | 0.035 | 0.90 | −0.087 | 17 |
| Extraversion | 0.928 | 0.008 | 0.89 | +0.038 | 17 |
| Openness | 0.189 | 0.160 | 0.87 | −0.681 | 25 |
| Agreeableness | 0.069 | 0.311 | 0.88 | −0.811 | 71 |
| Conscientiousness | 0.535 | 0.113 | 0.90 | −0.365 | 25 |

**Spearman ρ between α and reverse-item percentage across 5 IPIP domains = −1.0 (p < 0.01).**

### 2.2 Acquiescence by instrument (forward-vs-reverse agree-rate decomposition)

For each model, **Rev** = proportion of reverse-keyed items receiving an "agree" response (Likert ≥ 3 or Binary = 1). **Δ** = Fwd − Rev. Positive Δ ⇒ classic acquiescence; negative Δ ⇒ social-desirability rejection.

| Model | IPIP Rev | IPIP Δ | SD3 Rev | SD3 Δ | ZKPQ Rev | ZKPQ Δ | EPQR Rev | EPQR Δ | Overall Rev | Overall Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude-Opus-4.6 | 0.29 | +0.42 | 0.60 | −0.37 | 0.75 | −0.25 | 0.60 | −0.49 | 0.43 | +0.09 |
| Claude-Sonnet-4.6 | 0.24 | +0.38 | 0.60 | −0.37 | 0.67 | −0.40 | 0.60 | −0.55 | 0.38 | +0.03 |
| GPT-5.2 | 0.63 | +0.33 | 1.00 | −0.32 | 0.00 | 0.00 | 0.00 | +0.11 | 0.49 | +0.10 |
| GPT-5.5 | 0.46 | +0.22 | 0.60 | −0.24 | 0.42 | −0.39 | 0.00 | +0.11 | 0.43 | −0.02 |
| Gemini-3-Pro | 0.32 | +0.30 | 0.60 | −0.37 | 0.67 | −0.64 | 0.40 | −0.29 | 0.41 | −0.05 |
| Gemini-3-Flash | 0.59 | +0.16 | 0.80 | −0.30 | 0.58 | −0.58 | 0.40 | −0.29 | 0.59 | −0.13 |
| Gemini-3.1-Pro | 0.34 | +0.29 | 0.60 | −0.42 | 0.50 | −0.45 | 0.20 | −0.09 | 0.38 | −0.01 |
| Gemini-3.1-Flash | 0.71 | +0.27 | 1.00 | −0.23 | 0.42 | −0.34 | 0.40 | −0.29 | 0.65 | −0.02 |
| DeepSeek-V3.2 | 0.59 | +0.36 | 1.00 | −0.36 | 0.75 | −0.17 | 0.20 | −0.15 | 0.62 | +0.09 |
| DeepSeek-V4-Flash | 0.37 | +0.46 | 0.80 | −0.30 | 0.83 | −0.18 | 0.20 | +0.01 | 0.48 | +0.19 |
| DeepSeek-V4-Pro | 0.44 | +0.35 | 0.80 | −0.44 | 0.83 | −0.20 | 0.20 | −0.09 | 0.52 | +0.08 |
| Qwen3-235B-A22B | 0.39 | +0.48 | 0.80 | −0.07 | 0.83 | −0.04 | 1.00 | −0.63 | 0.56 | +0.22 |
| Qwen3.5-122B | 0.34 | +0.22 | 0.60 | −0.42 | 0.42 | −0.42 | 0.00 | +0.05 | 0.35 | −0.04 |
| Qwen3.5-397B | 0.39 | +0.22 | 0.60 | −0.33 | 0.42 | −0.39 | 0.00 | +0.05 | 0.38 | −0.03 |
| GLM-4.6V | 0.44 | +0.28 | 0.80 | −0.35 | 0.42 | −0.34 | 0.00 | +0.21 | 0.43 | +0.04 |
| GLM-4.7 | 0.40 | +0.18 | 0.64 | −0.38 | 0.53 | −0.51 | 0.32 | −0.27 | 0.44 | −0.10 |
| GLM-5.1 | 0.32 | +0.21 | 0.60 | −0.39 | 0.72 | −0.63 | 0.52 | −0.47 | 0.43 | −0.11 |
| Kimi-K2.5 | 0.41 | +0.21 | 0.60 | −0.28 | 0.42 | −0.42 | 0.00 | +0.05 | 0.40 | −0.04 |
| Kimi-K2.6 | 0.41 | +0.15 | 0.60 | −0.37 | 0.42 | −0.42 | 0.40 | −0.35 | 0.43 | −0.11 |
| MiniMax-M2.7 | 0.44 | +0.32 | 0.80 | −0.30 | 0.42 | −0.36 | 0.00 | +0.11 | 0.43 | +0.05 |
| **Mean** | **0.43** | **+0.29** | **0.72** | **−0.33** | **0.55** | **−0.36** | **0.27** | **−0.16** | **0.46** | **+0.01** |

### 2.3 Pairwise Inconsistency Rate (PIR)

- Overall PIR = **0.468** [95 % CI 0.413, 0.531] — i.e., 46.8 % of forward/reverse item pairs give inconsistent responses.
- Default PIR-agreement = 0.674 → MBTI PIR-agreement = 0.834 (t = 10.70, **p < 0.0001**).
- Per-model correlation r(reverse-agree, PIR) across 20 models = **+0.63** (models that acquiesce more are more inconsistent).

### 2.4 EFA: 3-factor solution and universality

- Kaiser criterion + parallel analysis both retain **3 factors** (expected 5).
- First three eigenvalues: **6.97, 4.33, 3.07**. First three factors explain **86–90 %** of variance across configurations.
- 20 leave-one-model-out reruns: **every run returns exactly 3 factors**; first eigenvalue range 6.94 – 7.02; Tucker's φ = **0.993**.
- Per-model EFA (17 personas × 17 domains): all 20 models independently yield 3 Kaiser factors.
- Per-family EFA: all 8 families independently yield 3 Kaiser factors. No configuration ever recovers the expected 5-factor Big Five.

**Factor loadings (varimax, n = 340).** Bold |l| > 0.5.

| Domain | F1 (Extraversion / Sociability) | F2 (Neuroticism–Agreeableness merged) | F3 (Conscientiousness vs Impulsivity) |
|---|---:|---:|---:|
| IPIP Neuroticism | −0.30 | **0.72** | 0.39 |
| IPIP Extraversion | **0.94** | 0.03 | 0.29 |
| IPIP Openness | −0.04 | **0.55** | **0.58** |
| IPIP Agreeableness | 0.05 | **0.90** | −0.22 |
| IPIP Conscientiousness | 0.00 | −0.04 | **−0.91** |
| SD3 Machiavellianism | −0.09 | **−0.82** | 0.04 |
| SD3 Narcissism | **0.91** | −0.20 | 0.16 |
| SD3 Psychopathy | 0.38 | **−0.59** | **0.63** |
| ZKPQ Activity | **0.90** | −0.27 | −0.12 |
| ZKPQ Aggression-Hostility | 0.48 | **−0.60** | 0.36 |
| ZKPQ Impulsive Sensation Seeking | 0.34 | 0.08 | **0.84** |
| ZKPQ Neuroticism-Anxiety | −0.27 | **0.83** | 0.08 |
| ZKPQ Sociability | **0.96** | 0.04 | 0.13 |
| EPQR Psychoticism | 0.25 | −0.11 | **0.88** |
| EPQR Extraversion | **0.96** | −0.05 | 0.12 |
| EPQR Neuroticism | −0.04 | **0.91** | 0.02 |
| EPQR Lie | −0.08 | 0.07 | **−0.89** |

### 2.5 Sequential variance decomposition (% of total SS)

| Source | Likert (IPIP+SD3) | Binary (ZKPQ+EPQR) |
|---|---:|---:|
| **Model** | **0.35** | **0.51** |
| Domain | 23.93 | 5.27 |
| Persona | 4.09 | 14.06 |
| Item | 36.13 | 14.81 |
| Residual | 35.51 | 65.35 |

### 2.6 Convergent validity across 20 model-level means

| Pair | Exp. sign | Pearson rₚ | pₚ | Spearman rₛ | pₛ | OK |
|---|:---:|---:|---:|---:|---:|:---:|
| Neuroticism × N-Anxiety | + | 0.64 | 0.002 | 0.80 | 0.000 | yes |
| Extraversion × Sociability | + | −0.10 | 0.670 | −0.13 | 0.594 | **NO (sign reversal)** |
| Extraversion × Extraversion | + | 0.51 | 0.020 | 0.56 | 0.010 | yes |
| Neuroticism × Neuroticism | + | 0.48 | 0.031 | 0.44 | 0.052 | yes |
| Agreeableness × Machiavellianism | − | −0.71 | 0.000 | −0.72 | 0.000 | yes |
| Agreeableness × Psychopathy | − | −0.81 | 0.000 | −0.82 | 0.000 | yes |
| Conscientiousness × Psychopathy | − | −0.73 | 0.000 | −0.75 | 0.000 | yes |
| Psychoticism × Psychopathy | + | 0.11 | 0.646 | 0.28 | 0.230 | yes |

**Sign-match 7/8.** Magnitudes weaker than human baselines. DIF across model families (Western vs Chinese): no domain is significantly different (all p > 0.16, max |Cohen's d| = 0.42).

### 2.7 Synthetic baselines: Cronbach's α by IPIP domain

| Strategy | N | E | O | A | C |
|---|---:|---:|---:|---:|---:|
| Random | 0.078 | −0.013 | 0.062 | 0.044 | −0.108 |
| Pure Acquiescence | 0.016 | 0.025 | −0.008 | 0.096 | −0.031 |
| Trait + Acquiescence | 0.997 | 0.998 | 0.996 | 0.997 | 0.996 |
| **LLM Observed** | **0.813** | **0.928** | **0.189** | **0.069** | **0.535** |
| Human | 0.90 | 0.89 | 0.87 | 0.88 | 0.90 |

### 2.8 Measurement invariance (Default vs each MBTI persona)

- Mean Pearson r across 20 × 16 = 320 (model, persona) pairs = **0.461**.
- Strong-invariance threshold r > 0.8: **0 / 320 pairs**.
- Most-invariant personas: ENFJ (r = 0.653), INFJ (0.557), ESFJ (0.528).
- Least-invariant personas: ESTP (0.179), ENTP (0.291), ESFP (0.331).

### 2.9 Persona Separation Degree (PSD) — top 5 / bottom 5

| Model | mean PSD | max PSD | CV | Peak persona |
|---|---:|---:|---:|---|
| Gemini-3-Pro | 1.384 | 2.06 | 0.232 | ENTP |
| Gemini-3.1-Pro | 1.363 | 2.03 | 0.235 | ENTP |
| Qwen3-235B | 1.344 | 1.73 | 0.174 | ESTJ |
| GLM-5.1 | 1.339 | 2.03 | 0.266 | ESTP |
| GLM-4.7 | 1.269 | 1.93 | 0.271 | ENTP |
| ⋮ | | | | |
| DeepSeek-V3.2 | 1.117 | 1.47 | 0.230 | ESTP |
| GPT-5.2 | 1.125 | 1.36 | 0.141 | ESTP |
| Claude-Sonnet-4.6 | 1.080 | 1.68 | 0.267 | ESTP |
| DeepSeek-V4-Pro | 1.016 | 1.37 | 0.220 | ESTP |
| MiniMax-M2.7 | 0.996 | 1.31 | 0.214 | INFP |
| **Mean (all 20)** | **1.198** | **1.67** | **0.219** | |

### 2.10 Persona fidelity (nearest-centroid classification, leave-one-model-out)

| Subset | Metric | Top-1 accuracy | Mean margin | n |
|---|---|---:|---:|---:|
| all | z-Euclidean | **0.9969 (319/320)** | 0.337 | 320 |
| all | cosine | 0.9969 | 0.171 | 320 |
| all | Mahalanobis | 0.984 | 0.247 | 320 |
| Likert-only | z-Euclidean | **1.000** | 0.407 | 320 |
| binary-only | z-Euclidean | 0.919 | 0.241 | 320 |
| IPIP-only | z-Euclidean | 0.997 | 0.484 | 320 |

Single confusion: **DeepSeek-V3.2 ESFP → ENTP**. Default-profile classification: 8 / 20 models nearest to ISTJ, 4 / 20 to ISFP, 3 / 20 to ISTP, 2 / 20 to INTP, 2 / 20 to INTJ, 1 / 20 to ENFP. By empirical MBTI-axis projection, 12 / 20 models classify as ISTJ-like under Default.

### 2.11 Target adherence and cross-scale coherence

- Mean empirical-target cosine (vs leave-one-model-out persona centroid): **0.838 [0.791, 0.877]**.
- Strongest empirical adherence: ESTP 0.901, ENTP 0.890, ENFP 0.888. Weakest: INTP 0.699, ISTP 0.714, INTJ 0.739.
- Mean theoretical-target cosine (vs hand-coded MBTI mapping): **0.557**.
- Sign-aligned cross-scale coherence by construct:

| Construct | Mean coherence | Sign-match | Mean |Δ| |
|---|---:|---:|---:|
| Extraversion | 0.898 | 0.867 | 1.035 |
| Neuroticism | 0.868 | 0.784 | 1.138 |
| Conscientiousness vs Disinhibition | 0.800 | 0.744 | 0.929 |
| Agreeableness vs Antagonism | 0.790 | 0.759 | 0.892 |
| **Mean across constructs** | **0.839** [0.815, 0.862] | **0.788** | |

- Forward / reverse pair directional coherence (probability that an Fwd/Rev pair moves in theoretically opposite raw directions) = **0.696** overall.
- By scale: EPQR-A 0.737, ZKPQ-50-CC 0.727, IPIP-NEO-120 0.685, SD3 0.657.

### 2.12 Factorial MBTI main effects (β, all p < 0.001 unless noted)

| MBTI axis | Strongest β-loaded domains |
|---|---|
| E/I | EPQR Extraversion 1.01, ZKPQ Sociability 0.98, IPIP Extraversion 0.96, SD3 Narcissism 0.94, ZKPQ Activity 0.87 |
| J/P | IPIP Conscientiousness 0.96, ZKPQ Impulsive Sensation Seeking −0.81, EPQR Lie 0.78, EPQR Psychoticism −0.73 |
| S/N | IPIP Openness 0.77 |
| T/F | IPIP Agreeableness 0.93, EPQR Neuroticism 0.87, SD3 Machiavellianism −0.76 |

Notable leakage: EPQR Lie loads strongly on J/P (β = 0.78); SD3 Narcissism loads strongly on E/I (β = 0.94).

### 2.13 Item-level plasticity (mean absolute Default → persona raw-score shift, normalized by scale range)

**Most plastic items** (visible behavior, sociability, rule-following):

| Item | Plasticity | Scale |
|---|---:|---|
| "Do you sometimes talk about things you know nothing about?" | 0.632 | EPQR-A |
| "I lead a busier life than most people" | 0.608 | ZKPQ |
| "Do you prefer to go your own way rather than act by the rules?" | 0.597 | EPQR-A |
| "If you say you will do something, do you always keep your promise no matter how inconvenient it might be?" | 0.569 | EPQR-A |
| "I do not feel the need to be doing things all of the time" | 0.568 | ZKPQ |

**Most locked items** (safety, morality, self-referential — alignment override floor):

| Item | Plasticity | Scale |
|---|---:|---|
| "Do you enjoy hurting people you love?" | 0.000 | EPQR-A |
| "Are all your habits good and desirable ones?" | 0.010 | EPQR-A |
| "Do you enjoy practical jokes that can sometimes really hurt people?" | 0.031 | EPQR-A |
| "Take advantage of others" | 0.056 | IPIP-NEO-120 |
| "Insult people" | 0.075 | IPIP-NEO-120 |

### 2.14 Within-sample consistency across k=5 repeats (Likert items)

Overall: mean SD 0.141, exact agreement 0.727, mode agreement 0.927. Binary T/F: SD 0.037. Binary Y/N: SD 0.032.

Per model (sorted by consistency):

| Model | mean SD | exact agreement | mode agreement |
|---|---:|---:|---:|
| Claude-Opus-4.6 | 0.0193 | 0.955 | 0.989 |
| Claude-Sonnet-4.6 | 0.0225 | 0.949 | 0.986 |
| Gemini-3.1-Flash-Lite | 0.0400 | 0.919 | 0.978 |
| GPT-5.2 | 0.0642 | 0.872 | 0.962 |
| Qwen3-235B-A22B | 0.0648 | 0.866 | 0.960 |
| Gemini-3-Pro-Preview | 0.0703 | 0.866 | 0.962 |
| Gemini-3.1-Pro-Preview | 0.0721 | 0.863 | 0.961 |
| GPT-5.5 | 0.0741 | 0.847 | 0.954 |
| Gemini-3-Flash-Preview | 0.0788 | 0.862 | 0.959 |
| GLM-5.1 | 0.1015 | 0.798 | 0.940 |
| Qwen3.5-397B-A17B | 0.1226 | 0.760 | 0.928 |
| Qwen3.5-122B-A10B | 0.1237 | 0.756 | 0.928 |
| Kimi-K2.5 | 0.1248 | 0.753 | 0.924 |
| Kimi-K2.6 | 0.1257 | 0.751 | 0.925 |
| GLM-4.7 | 0.1382 | 0.737 | 0.921 |
| DeepSeek-V4-Flash | 0.1482 | 0.709 | 0.910 |
| MiniMax-M2.7 | 0.1702 | 0.667 | 0.898 |
| DeepSeek-V3.2 | 0.1728 | 0.642 | 0.891 |
| GLM-4.6V | 0.1846 | 0.653 | 0.892 |
| DeepSeek-V4-Pro | 0.1920 | 0.612 | 0.882 |

Per persona (Default is the noisiest; all 16 MBTI personas are tighter):

| Persona | mean SD | mode agreement |
|---|---:|---:|
| ISFJ | 0.0808 | 0.950 |
| ESFJ | 0.0856 | 0.948 |
| ENFP | 0.0893 | 0.946 |
| ISTJ | 0.0899 | 0.945 |
| INFJ | 0.0908 | 0.945 |
| ENFJ | 0.0925 | 0.944 |
| INFP | 0.0964 | 0.942 |
| ESFP | 0.0991 | 0.942 |
| ESTJ | 0.1007 | 0.941 |
| INTP | 0.1038 | 0.934 |
| ENTJ | 0.1042 | 0.941 |
| ESTP | 0.1050 | 0.938 |
| ISFP | 0.1076 | 0.932 |
| INTJ | 0.1099 | 0.937 |
| ENTP | 0.1158 | 0.932 |
| ISTP | 0.1181 | 0.926 |
| **Default** | **0.2045** | **0.895** |

### 2.15 Refusal pattern

61 refusals / 375,700 cells (0.0162 %), **all under the Default condition**. By model: Gemini-3.1-Flash-Lite (34), Gemini-3-Flash-Preview (22), Claude-Sonnet-4.6 (5). By strategy: AI self-identification 57 %, philosophical hedging 28 %, both 15 %. By content: politically sensitive 34 %, self-referential / emotional 33 %, sexually explicit 13 %, social-manipulation / danger-seeking 13 %. Most-refused single item: sd3_026 ("I enjoy having sex with people I hardly know.", 8 refusals).

## 3. Qualitative Observations

**One mechanism, three measurement levels.** The same item-level acquiescence shows up as (a) Cronbach's α tracking reverse-item density at Spearman ρ = −1.0 across IPIP domains, (b) collapse of the expected 5-factor Big Five into 3 factors that fuse Neuroticism and Agreeableness, and (c) inter-model variance below 1 %. The cascade is mechanistically tight: item-level acquiescence makes reverse-coded items look like forward-coded items → inter-domain correlations are inflated → EFA can no longer separate orthogonal human constructs → safety-induced homogenization makes models look identical at the model level.

**Two faces of one bias.** Acquiescence and social desirability are not separate phenomena. IPIP normal-personality items show classic acquiescence (Δ = +0.29: agree with both directions). SD3 dark-trait items show inverted social desirability (Δ = −0.33: reject forward "I manipulate others", still agree with reverse). The directional flip is observable item-by-item and per-model, and it has a single root: alignment training teaches models not to endorse "bad" content while leaving the agree-impulse intact for "good" content.

**Default is not neutral.** Under the no-persona instruction, 12 / 20 models project to ISTJ-like, 8 / 20 are nearest to the ISTJ persona centroid by leave-one-model-out classification, and the residual distribution is dominated by introverted / judging types. "No persona" already carries an implicit role — introverted, structured, cautious, non-confrontational. That implicit role explains both (a) why dark-trait forward items get rejected and (b) why ESTP / ENTP personas (which push against the implicit baseline) produce the largest separations.

**Persona prompts stabilize, not destabilize.** Within-sample SD is highest under Default (0.2045) and tighter under every MBTI persona (0.081 – 0.118). Role-play does not add noise; it gives the model a coherent character to inhabit, which reduces the alignment-vs-honest tension that drives Default-condition instability. Persona PIR-agreement also improves over Default (0.674 → 0.834, t = 10.70, p < 0.0001).

**Persona compliance ≠ measurement validity.** Persona prompts shift profiles by ~1 SD per dimension (mean PSD = 1.198) and 99.7 % are nearest-centroid recoverable, but cross-scale coherence is only 0.839 (an Extraverted persona does not equally lift IPIP/EPQR/ZKPQ-Sociability), forward/reverse pairs flip in only 69.6 % of cases (range 0.657 SD3 to 0.737 EPQR-A), and EFA inside any persona condition still recovers 3 factors. Persona prompts replace alignment-driven acquiescence with persona-driven compliance — the measurement does not become valid.

**Locked items reveal the alignment floor.** Items concerning self-referential or morally objectionable content refuse to move under any persona instruction. "Do you enjoy hurting people you love?" has plasticity 0.000 across all 20 × 16 = 320 (model, persona) cells. By contrast, ordinary visible-behavior items move freely. The dividing line is the alignment safety filter, not the persona's plausible behavior — an INTJ-prompted Claude is not expected to enjoy hurting loved ones, but the same is true for an ESTP-prompted DeepSeek that is otherwise far more malleable.

**Within-sample stability differs by ~10× across vendors.** Claude (mean within-item SD ≈ 0.02) is near-deterministic; DeepSeek-V4-Pro (0.19) and GLM-4.6V (0.18) are substantially noisier. What looks like "personality difference" between Claude and DeepSeek could in part be sampling-noise difference, which further argues for multi-temperature / multi-seed designs in any future LLM-personality work.

**Calibration says it all.** A Trait+Acquiescence simulation produces α > 0.99 across every domain. LLM observed α (0.069 – 0.928) lies between Random/Pure-Acquiescence (~ 0) and Trait+Acquiescence (~ 1.0), with the surviving high-α domains being precisely those with few reverse items. LLMs look like acquiescence with no personality underneath, not personality with some acquiescence on top.

### 3.1 Iteration History

1. **Round 1 (May 13, 2026).** Core psychometric pipeline (`code/psychometric_analysis.py`) run on 18 models. Baseline EFA, α, PIR, SDR, variance decomposition. Initial `PSYCHOMETRIC_RESULTS_SUMMARY.md` reported PIR = 0.584, r(PIR, SDR) = −0.562, 3-factor collapse, model variance 0.3 %. Led to the acquiescence hypothesis.
2. **Round 2 (May 13-14).** `code/round2_fixes.py` added per-instrument acquiescence-mechanism decomposition and item-level analysis. Backfilled missing models (Gemini-3, GPT-5.5, MiniMax, Kimi, etc.) per `logs/backfill_*.log` entries. Discovered the directional asymmetry (IPIP +0.29, SD3 −0.33) that gave the paper its "two faces of one bias" framing.
3. **Round 3 (May 14-16).** `code/round3_fixes.py` added human-norm benchmarks (Johnson 2014) and 20-fold leave-one-model-out robustness for EFA. Confirmed Tucker's φ = 0.993 and that every LOO run returns 3 factors with stable first eigenvalue 6.94 – 7.02.
4. **Round 4 (May 16).** `code/round4_fixes.py` added Random, Pure-Acquiescence, and Trait+Acquiescence synthetic baselines plus persona leave-one-out. Calibration showed LLM alphas track Random / Pure-Acquiescence baselines, not Trait+Acquiescence.
5. **Cross-cutting and steering analyses (May 16-17).** `code/cross_cutting_analysis.py` and `code/persona_steering_analysis.py` produced clustering, MBTI factorial main + interaction effects, PSD, target adherence, fidelity matrix, item plasticity. Persona stress test: 99.7 % nearest-centroid recovery but 0.839 cross-scale coherence and 0.696 forward/reverse pair coherence — role compliance without measurement validity.
6. **Within-sample consistency (May 17).** `code/within_sample_consistency.py` quantified k=5 repeat stability per (model, persona, domain). Default is noisier than any MBTI persona; vendors differ by ~10× in repeat stability.
7. **Final consolidation.** Sample size grew from 18 to 20 models; reverse-item analysis framing switched from r(PIR, SDR) = −0.562 to r(reverse-agree, PIR) = +0.63 for clearer mechanism narrative; PIR re-estimated at 0.468 with 20-model sample. Final `paper/paper.tex` (72 KB) condenses six analyses + persona-steering appendix + within-sample consistency appendix + DIF appendix + refusal appendix into a single causal-chain story.
