# Response Style Variation Across Language Model Families

Large-scale psychometric comparison of 15 model families (13 Chinese AI vendors + 2 international) on standardized personality inventories.

**Paper:** EMNLP 2026 Findings
**Title:** Response Style Variation Across Language Model Families: A Large-Scale Psychometric Comparison of 15 Model Families

## Overview

We administer 61 Likert-scale items spanning 9 psychometric dimensions to one model from each of 15 model families via unified APIs, collecting 10,980 responses (12 seeds per model). We find that model families differ substantially in their response distributions (median pairwise Cohen's *d* = 1.08), HEXACO-Honesty-Humility is the most discriminative dimension, and LLM inter-dimension correlation structure diverges from human personality baselines.

## Repository Structure

```
├── paper/                          # Paper source and compiled PDF
│   ├── emnlp2026_improved.tex      # LaTeX source
│   ├── emnlp2026_improved.pdf      # Compiled paper
│   ├── references.bib              # Bibliography
│   └── acl.sty, acl_natbib.bst     # EMNLP style files
├── figures/                        # Publication figures (PDF)
│   ├── fig1-7_*.pdf                # Main figures
│   ├── appendix_*.pdf              # Appendix figures
│   ├── gen_*.py                    # Figure generation scripts
│   └── paper_plot_style.py         # Plotting style config
├── data/
│   └── results.json                # Merged experiment results (all studies)
├── run_vendor_experiments.py       # Experiment runner (SiliconFlow + YiHe API)
├── analyze_vendor_design.py        # Statistical analysis (ANOVA, OLR, FDR, ICC)
├── create_pca_figure.py            # PCA visualization
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Analysis

The analysis script operates on the merged results JSON:

```bash
python analyze_vendor_design.py --input data/results.json
```

This produces:
- OLS ANOVA with FDR-corrected *p*-values and partial η²
- Ordered Logistic Regression with architecture covariate
- Pairwise Cohen's *d* effect sizes (Welch's *t*-test, FDR-corrected)
- Cronbach's α and ICC(1,1) reliability
- Convergent validity check against human baselines
- Acquiescence bias analysis (PC1 correction)
- Alignment artifact classification with threshold sensitivity

## Running Experiments

Experiments require API access:

```bash
export SILICONFLOW_API_KEY="your-key"
# Optional: YiHe API for international models
export YIHE_API_KEY="your-key"

python run_vendor_experiments.py
```

See the script header for model configuration (Study 1: 15 families, Study 2: within-family evolution, Study 3: international models).

## Data Format

`data/results.json` is a JSON array where each element represents one (model, seed) observation:

```json
{
  "model_id": "Qwen/Qwen3.5-397B-A17B",
  "vendor": "Qwen",
  "arch": "MoE",
  "study": 1,
  "seed": 0,
  "items": {
    "bfi_extraversion": [3, 3, 5, 1, 3, 3, 4, 3],
    ...
  },
  "bfi.extraversion": 3.0,
  "bfi.agreeableness": 3.0,
  "hexaco_h": 1.0,
  ...
}
```

**9 Dimensions:** BFI Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness; HEXACO Honesty-Humility; Collectivism; Intuition; Uncertainty Avoidance.

**Studies:**
- Study 1 (15 models): Cross-family comparison via SiliconFlow API
- Study 2 (16 models): Within-family evolution (Qwen, DeepSeek, GLM)
- Study 3 (6 models): International providers (OpenAI, Anthropic, Gemini, Grok) via YiHe API

## Citation

```bibtex
@inproceedings{anonymous2026response,
  title={Response Style Variation Across Language Model Families: A Large-Scale Psychometric Comparison of 15 Model Families},
  author={Anonymous},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026}
}
```

## License

MIT
