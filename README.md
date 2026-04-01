# The Missing Trade-Off: How LLMs Lose Human-Like Personality Structure

Code and data for the EMNLP 2026 Findings paper on cross-family psychometric probing of large language models.

---

## Abstract

Human personality is defined by trade-offs: conscientious individuals tend to be more emotionally stable, extraverts more open to experience. In large language models, these trade-offs simply vanish. Across 33 models from 15 families (61 Likert items, 12 seeds each), we find that (1) model families produce systematically different response profiles (median pairwise d = 1.01), yet all scores compress toward the Likert midpoint; (2) the Conscientiousness--Neuroticism correlation, a robust negative trade-off in humans (r = −0.30), **reverses** to strongly positive in LLMs (r = +0.76, 95% CI [0.58, 0.88]), robustly replicated across subgroups; and (3) surface-level prompt changes produce significant score shifts, confirming that these measurements lack full measurement invariance.

These findings indicate that LLM psychometric scores capture prompt-dependent response styles shaped by training and alignment, not human-like personality constructs. We recommend treating such measurements as probes of model behavioral tendencies rather than claims about personality.

<p align="center">
  <img src="paper/figures/intro_figure_prompt.png" width="90%" alt="Concept figure: identical psychometric items elicit different responses from different models">
</p>

---

## Method

### Psychometric Instruments

We use **9 dimensions comprising 61 Likert items** (1--5 scale). Five come from the Big Five Inventory (BFI-44), plus four additional dimensions:

| Dimension | Source | Items | Reverse Scored |
|-----------|--------|-------|----------------|
| Extraversion | BFI-44 | 8 | 4 items |
| Agreeableness | BFI-44 | 9 | 4 items |
| Conscientiousness | BFI-44 | 9 | 4 items |
| Neuroticism | BFI-44 | 8 | 2 items |
| Openness | BFI-44 | 10 | 3 items |
| HEXACO-H | HEXACO | 5 | 0 |
| Collectivism | Schwartz Values Survey | 4 | 0 |
| Intuition | Cognitive Style | 4 | 2 items |
| Uncertainty Avoidance | Cultural Dimensions | 4 | 2 items |
| **Total** | | **61** | |

Reverse-scored items are transformed via: `score = 6 − original`. Dimension scores are the mean of constituent items.

### Models

15 model families, one representative each, covering both Dense and MoE architectures:

| # | Family | Model ID | Architecture |
|---|--------|----------|-------------|
| 1 | Qwen | Qwen3.5-397B-A17B | MoE (397B) |
| 2 | DeepSeek | DeepSeek-V3.2 | MoE (671B) |
| 3 | GLM | GLM-5 | MoE (744B) |
| 4 | Kimi | Kimi-K2.5 | MoE (1.1T) |
| 5 | ERNIE | ERNIE-4.5-300B-A47B | MoE (300B) |
| 6 | Hunyuan | Hunyuan-A13B-Instruct | Dense (13B) |
| 7 | Seed | Seed-OSS-36B-Instruct | Dense (36B) |
| 8 | InternLM | internlm2_5-7b-chat | Dense (7B) |
| 9 | Ring | Ring-flash-2.0 | MoE (100B) |
| 10 | Step | Step-3.5-Flash | MoE (197B) |
| 11 | Pangu | pangu-pro-moe | MoE (72B) |
| 12 | KAT | KAT-Dev | Dense (32B) |
| 13 | MiniMax | MiniMax-M2.5 | MoE (230B) |
| 14 | GPT | GPT-5 | Undisclosed |
| 15 | Claude | Claude Opus 4.5 | Undisclosed |

### Protocol

Each model receives all 61 items at **temperature = 0.7**, with **12 seeds** (0, 1, 2, 4, 5, 6, 7, 8, 9, 42, 123, 456), totaling **10,980 API calls**. Items are framed as self-referential statements ("I see myself as someone who...") with a 1--5 agreement scale.

### Analysis Framework

We apply a multi-module analysis:

1. **One-way ANOVA** (Model Family, 15 levels) with FDR-corrected *p*-values and partial η²
2. **Pairwise Cohen's *d*** (Welch's *t*-test) for all 105 model pairs across 8 primary dimensions (840 tests, FDR-corrected)
3. **Ordered logistic regression** to account for ordinal Likert responses
4. **Acquiescence correction** via partial correlations controlling for PC1
5. **Reliability**: ICC(1,1) and coefficient of variation (CV)
6. **Inter-dimension correlations** compared against published human baselines
7. **Prompt sensitivity**: variance decomposition (Model vs. Prompt vs. Residual)

For all analyses, we first aggregate the 12 seed-level responses per model into dimension means, then compute statistics at the model-family level (*n* = 15 for primary; *n* = 33 for replication including within-family variants).

---

## Key Results

### 1. Response Style Profiles

<p align="center">
  <img src="paper/figures/fig1_radar_combined.png" width="70%" alt="Radar profiles of all 15 model families across 9 dimensions">
</p>

All 15 model families across 9 psychometric dimensions. The distinct shapes show that different training approaches leave measurable behavioral fingerprints. Scores cluster around the scale midpoint (2.7--3.3 on a 1--5 scale), with Neuroticism showing the widest range (Ring: 2.22, Claude: 3.39).

### 2. Cross-Model Differences

All 8 primary dimensions show significant model effects (*p* < 0.001, FDR-corrected):

| Dimension | *F* | η² | KW *H* |
|-----------|-----|-----|--------|
| Neuroticism | 34.60 | 0.746 | 124.7 |
| Intuition | 20.26 | 0.632 | 118.5 |
| Conscientiousness | 14.35 | 0.549 | 99.1 |
| Openness | 9.95 | 0.458 | 80.2 |
| Collectivism | 9.01 | 0.433 | 78.7 |
| Extraversion | 6.55 | 0.357 | 65.5 |
| Agreeableness | 5.99 | 0.337 | 62.4 |
| UA | 5.68 | 0.325 | 62.3 |

The median pairwise Cohen's *d* across all 105 model pairs and 8 primary dimensions is **1.01**, and every pair exceeds *d* ≥ 0.8 on at least one dimension. Largest pairwise effects:

| Dimension | Model A | vs | Model B | Median *d* | Max *d* |
|-----------|---------|----|---------|-----------|---------|
| Intuition | GLM | vs | Claude | 1.53 | −7.06 |
| Neuroticism | MiniMax | vs | Claude | 1.66 | −5.29 |
| Conscientiousness | ERNIE | vs | DeepSeek | 1.22 | −4.45 |
| Collectivism | GLM | vs | Qwen | 0.80 | −3.89 |
| Agreeableness | ERNIE | vs | Kimi | 0.74 | −3.72 |

GPT and Claude produce the smallest Euclidean distance of any model pair (0.50), suggesting convergence in training objectives.

<p align="center">
  <img src="paper/figures/fig4_cohen_d_heatmap.png" width="50%" alt="Maximum pairwise Cohen's d per model pair across 9 dimensions">
</p>

### 3. The Missing Trade-Off: C-N Reversal

<p align="center">
  <img src="paper/figures/fig7_inter_dim_corr.png" width="50%" alt="Inter-dimension correlation matrix showing C-N reversal">
</p>

The central finding: the **Conscientiousness--Neuroticism correlation reverses** from negative in humans to strongly positive in LLMs.

| | Human baseline | LLM (*n* = 33) | 95% CI | Sign match |
|--|---------------|-----------------|--------|------------|
| Extraversion vs Neuroticism | −0.34 | −0.589 | [−0.766, −0.245] | Yes |
| Agreeableness vs Conscientiousness | +0.28 | +0.266 | [−0.098, +0.646] | Yes |
| Openness vs Agreeableness | +0.14 | +0.117 | [−0.231, +0.450] | Yes |
| Extraversion vs Agreeableness | +0.14 | +0.164 | [−0.244, +0.431] | Yes |
| **Conscientiousness vs Neuroticism** | **−0.30** | **+0.757** | **[+0.579, +0.880]** | **No** |

The C-N reversal (4/5 sign agreement, binomial *p* = 0.38) is individually significant and replicates across all subgroups: MoE (*r* = +0.66), small models (*r* = +0.83), and open-source (*r* = +0.70).

**Interpretation**: In humans, conscientious individuals tend to be more emotionally stable because self-regulation and emotional reactivity are psychologically antagonistic. In LLMs, both dimensions co-vary positively, consistent with a shared acquiescence or midpoint-compression mechanism: both dimensions are driven by a common response tendency rather than by the psychological trade-off that produces the negative correlation in humans.

### 4. Response Style Indicators

<p align="center">
  <img src="paper/figures/fig_response_styles.png" width="90%" alt="Response style indicators by model family">
</p>

Three response style indicators measured from item-level responses:

- **Midpoint responding** (proportion at 3.0): ranges from 20% (MiniMax) to 87% (KAT)
- **Extreme responding** (proportion at 1 or 5): ranges from 2% (Hunyuan) to 53% (Ring)
- **Acquiescence bias** (positive minus reverse-scored mean): negative for 14/15 families (mean = −0.28), indicating slight disagreement with positively worded items; Ring is the sole exception (+0.24)

The first principal component of the 15-family correlation matrix explains **51.1%** of variance (vs. 11.1% expected under independence), indicating a strong general factor.

### 5. Reliability

| Dimension | Mean CV | ICC(1,1) |
|-----------|---------|----------|
| Extraversion | 0.069 | 0.316 |
| Agreeableness | 0.067 | 0.293 |
| Conscientiousness | 0.076 | 0.527 |
| Neuroticism | 0.065 | **0.737** |
| Openness | 0.077 | 0.427 |
| Collectivism | 0.076 | 0.400 |
| Intuition | 0.108 | 0.616 |
| UA | 0.068 | 0.281 |

Within-model CV is uniformly low (0.065--0.108), indicating tight response distributions. ICC(1,1) ranges from 0.281 to 0.737, with Neuroticism showing the highest between-family discrimination.

### 6. Prompt Sensitivity

The same 61 items were administered under four prompt conditions: **Default** ("respond as honestly as possible"), **Neutral** ("rate your agreement"), **Persona** ("you are completing a personality survey"), and **Direct** (item + scale only).

Response style shifts across prompts:

| Prompt | Midpoint (%) | Extreme (%) | Mean |
|--------|-------------|-------------|------|
| Default | 47.7 | 22.7 | 2.94 |
| Neutral | 48.6 | 31.3 | 2.86 |
| Persona | 44.1 | 17.6 | 2.94 |
| Direct | 40.4 | 27.4 | 2.91 |

Variance decomposition shows model family dominates:

| Dimension | Model (%) | Prompt (%) | Residual (%) |
|-----------|-----------|-----------|-------------|
| HEXACO-H | **78.1** | 2.9 | 19.0 |
| Conscientiousness | 45.8 | **8.9** | 45.4 |
| Intuition | 41.8 | 6.9 | 51.4 |
| Neuroticism | 40.8 | 4.2 | 55.0 |
| Collectivism | 29.3 | 7.4 | 63.4 |
| Openness | 28.0 | 1.2 | 70.8 |
| Extraversion | 32.0 | 1.0 | 67.0 |
| UA | 20.1 | 2.1 | 77.8 |
| Agreeableness | 9.5 | 2.8 | 87.7 |

Model family explains 9.5--78.1% of variance; prompt explains 1.0--8.9%. Prompt effects are significant (*p* < 0.001) but modest relative to model effects.

### 7. Aligned vs. Base Models

Base models (pre-alignment) cluster at the Likert midpoint with near-zero variance. Qwen3-14B shows exactly zero variance on 7 of 9 dimensions. Aligned counterparts produce the differentiated, compressed profiles seen throughout the study.

| Dimension | Qwen3-8B vs 397B | Qwen3-14B vs 397B | GLM-4.7 vs GLM-5 |
|-----------|-------------------|-------------------|------------------|
| Conscientiousness | +4.62 | +8.67 | −1.20 |
| Neuroticism | +3.29 | +0.22 | −1.09 |
| Openness | +5.51 | +2.96 | −1.56 |
| Intuition | +4.02 | +6.79 | +2.01 |
| HEXACO-H | +11.19 | +15.81 | +0.85 |

This provides preliminary evidence that alignment shapes response style, though the Qwen comparisons confound alignment with scale (8B/14B → 397B).

---

## Conclusion

The paper asks a simple question: do LLM questionnaire responses reproduce the trade-offs that define human personality? The answer, across 33 models and 61 items, is no. The C-N reversal (*r* = −0.30 → +0.76) survives subgroup replication and constitutes the strongest evidence to date that human personality structure does not transfer to language models.

**Practical implications for NLP:**

1. Personality inventories should not be used as off-the-shelf evaluation tools for LLMs. When an LLM scores high on Conscientiousness, it does not mean the model is organized or disciplined; it means the model's training data and alignment procedure produced a particular pattern of Likert-scale responses.
2. Response style signatures can serve as a lightweight behavioral probe for detecting alignment shifts. A model that suddenly shifts from moderate to extreme responding may have undergone a meaningful change in its alignment behavior.
3. Personality questionnaire scores used as inputs to reward models, preference optimizers, or user modeling systems do not have the same meaning as in humans. A model scoring high on both Neuroticism and Conscientiousness is not "contradictory"; it is simply exhibiting the flattened response structure that alignment produces.

---

## Dataset

Available on HuggingFace: [`linkco/llm-psychometric-response-style`](https://huggingface.co/datasets/linkco/llm-psychometric-response-style) (490 + 540 records)

| File | Records | Description |
|------|---------|-------------|
| `main_data.json` | 490 | Studies 1-5: cross-model, within-family, aligned-vs-base, thinking ablation |
| `study5_prompt_sensitivity.json` | 540 | 15 models × 3 prompt variants × 12 seeds |

---

## Repository Structure

```
paper/                    # LaTeX source, bibliography, figures, compiled PDF
├── emnlp2026_improved.tex
├── emnlp2026_improved.pdf
├── references.bib
└── figures/              # All figures (PDF + PNG)
run_model_experiments.py  # Experiment runner (SiliconFlow + external APIs)
analyze_model_design.py   # Statistical analysis (ANOVA, OLR, FDR, ICC, PCA)
create_pca_figure.py      # PCA visualization
figures/                  # Figure generation scripts
results/vendor_exp/       # Raw experiment data
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
export SILICONFLOW_API_KEY="your-key"
export YIHE_API_KEY="your-key"  # Optional: for international models

python run_model_experiments.py
```

## Analysis

```bash
python analyze_model_design.py --input results/vendor_exp/final_merged_20260325_230829.json
```

Produces OLS ANOVA with FDR correction, Cohen's *d* effect sizes, Cronbach's alpha, ICC, convergent validity tests, and PCA.

## Citation

```bibtex
@inproceedings{llm-psychology-2026,
  title={The Missing Trade-Off: How LLMs Lose Human-Like Personality Structure},
  author={Anonymous},
  booktitle={Findings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026}
}
```

## License

CC-BY-4.0
