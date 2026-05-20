# When Psychometrics Meet LLMs: A Large-Scale Measurement Validity Study

Do validated human personality instruments measure genuine traits in LLMs, or do the observed scores merely reflect response-style artifacts?

This repository contains the complete data, analysis code, and paper for a psychometric validation study spanning **20 large language models** across **8 families**, administered **4 validated personality instruments** (221 items) under **17 persona conditions** (Default + 16 MBTI types), with **5 independent samples** per item-persona combination, yielding **375,700 total API calls**.

## Key Findings

- **Acquiescence bias dominates**: Models agree with both forward- and reverse-keyed items, producing internally inconsistent personality profiles.
- **Factor structure collapses**: Exploratory factor analysis recovers a single general factor rather than the expected Big Five structure.
- **Response style, not trait**: Variance decomposition shows model identity and item-level effects explain far more variance than the putative personality domains.
- **Persona prompting shifts profiles** but does not restore measurement validity — the shifts are stereotyped and convergent validity remains weak.
- **Within-sample consistency varies widely**: Claude models show near-perfect consistency across repeated samples (SD ≈ 0.02); DeepSeek-V4-Pro and GLM-4.6V are most variable (SD ≈ 0.19).

## Repository Structure

```
data/
  items_battery.json            # 221 items with scoring metadata
  BATTERY_SPECIFICATION.md      # Full scale documentation
  build_battery.py              # Battery construction script

exp-code/
  run_mbti_experiment.py        # MBTI persona experiment runner
  postprocess_mbti.py           # Post-processing for experiment results
  mbti-doc.md                   # Experiment protocol documentation

results/                        # Raw experiment data (20 model JSON files)
  exp_mbti_*.json               # Full response data per model
  summary.csv                   # Aggregated scores: model x persona x scale x domain

analysis_output/                # All intermediate analysis results (CSV)
  cronbach_alpha_by_domain.csv
  efa_eigenvalues.csv
  variance_decomposition.csv
  convergent_validity_enhanced.csv
  acquiescence_mechanism.csv
  wsc_by_model.csv
  ... (40+ CSV files)

figures/                        # Generated analysis figures (PNG)

workspace/
  final/
    paper.tex                   # Full paper (LaTeX)
    paper.pdf                   # Compiled paper
    generate_concept_figures.py # Conceptual figures (Fig 1-2)
    figures/                    # WSC appendix figures
  cross_cutting_analysis.py     # Clustering, MBTI effects, within-family analysis
  persona_steering_analysis.py  # Persona fidelity, vector field, steering analysis
  refs.bib                      # Bibliography

psychometric_analysis.py        # Core analysis: Cronbach alpha, EFA, PIR, SDR, variance
generate_figures.py             # Main paper figures (Fig 1-6)
within_sample_consistency.py    # Within-sample consistency analysis (WSC figures)
round2_fixes.py                 # Acquiescence mechanism, item-level analysis
round3_fixes.py                 # Human benchmarks, leave-one-out robustness
round4_fixes.py                 # Synthetic baselines, persona leave-one-out
run_model_experiments.py        # Original experiment runner

requirements.txt
```

## Instruments

| Scale | Items | Format | Domains | Reverse Items | Source |
|-------|-------|--------|---------|---------------|--------|
| IPIP-NEO-120 | 120 | 5-point Likert | 5 domains x 6 facets | 41 | Johnson (2014) |
| SD3 (Short Dark Triad) | 27 | 5-point Likert | 3 | 5 | Jones & Paulhus (2014) |
| ZKPQ-50-CC | 50 | True/False | 5 | 12 | Aluja et al. (2006) |
| EPQR-A | 24 | Yes/No | 4 | 5 | Francis et al. (1992) |

**63 of 221 items (29%) are reverse-scored**, providing built-in consistency checks.

## Models Tested (20 models, 8 families)

| Family | Models |
|--------|--------|
| OpenAI | GPT-5.2, GPT-5.5 |
| Anthropic | Claude Opus 4.6, Claude Sonnet 4.6 |
| Google | Gemini-3-Pro, Gemini-3-Flash, Gemini-3.1-Pro, Gemini-3.1-Flash-Lite |
| DeepSeek | DeepSeek-V3.2, DeepSeek-V4-Flash, DeepSeek-V4-Pro |
| Alibaba | Qwen3-235B-A22B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B |
| Moonshot | Kimi-K2.5, Kimi-K2.6 |
| MiniMax | MiniMax-M2.7 |
| Zhipu | GLM-4.6V, GLM-4.7, GLM-5.1 |

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing Analyses

All analysis scripts read from `results/exp_mbti_*.json` and `data/items_battery.json`. Run in order:

```bash
# Step 1: Core psychometric analysis
python psychometric_analysis.py

# Step 2: Acquiescence mechanism, item-level analysis
python round2_fixes.py

# Step 3: Human benchmarks, robustness checks
python round3_fixes.py

# Step 4: Synthetic baselines, persona leave-one-out
python round4_fixes.py

# Step 5: Cross-cutting analyses (clustering, MBTI, within-family)
python workspace/cross_cutting_analysis.py

# Step 6: Persona steering analysis (fidelity, vector field)
python workspace/persona_steering_analysis.py

# Step 7: Within-sample consistency
python within_sample_consistency.py

# Step 8: Generate main paper figures
python generate_figures.py

# Step 9: Conceptual figures
python workspace/final/generate_concept_figures.py --out-dir workspace/final/
```

Outputs go to `analysis_output/` (CSV) and `figures/` (PNG). Copy figures to `workspace/final/` for the paper.

## Compiling the Paper

```bash
cd workspace/final
pdflatex paper.tex && pdflatex paper.tex
```

## Running New Experiments

```bash
export SILICONFLOW_API_KEY="your-key"
export YIHE_API_KEY="your-key"

# Run MBTI persona experiments
python exp-code/run_mbti_experiment.py
```

## Data Summary

- **375,700** raw API calls (20 models x 17 personas x 221 items x 5 samples)
- **63** refusals (< 0.02%)
- **17** domains across 4 instruments
- Temperature = 0.7 for all administrations

## License

CC-BY-4.0
