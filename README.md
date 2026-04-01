# The Missing Trade-Off: How LLMs Lose Human-Like Personality Structure

Code and data for the EMNLP 2026 Findings paper on cross-family psychometric probing of large language models.

## Paper

**Title:** The Missing Trade-Off: How LLMs Lose Human-Like Personality Structure

Human personality is defined by trade-offs: conscientious individuals tend to be more emotionally stable, extraverts more open to experience. In LLMs, these trade-offs vanish. We administer 61 Likert items (9 dimensions) to 33 models from 15 families and find that the Conscientiousness--Neuroticism correlation reverses from $r = -0.30$ (humans) to $r = +0.76$ (LLMs), providing the strongest evidence to date that human personality structure does not transfer to language models.

## Dataset

The experiment data is available on HuggingFace:

**Dataset:** `linkco/llm-psychometric-response-style` (490 + 540 records)

| File | Records | Description |
|------|---------|-------------|
| `main_data.json` | 490 | Studies 1-5: cross-model, within-family, aligned-vs-base, thinking ablation |
| `study5_prompt_sensitivity.json` | 540 | 15 models x 3 prompt variants x 12 seeds |

## Repository Structure

```
paper/                    # LaTeX source, bibliography, figures, compiled PDF
├── emnlp2026_improved.tex
├── emnlp2026_improved.pdf
├── references.bib
└── figures/              # All figures used in the paper
run_model_experiments.py  # Experiment runner (SiliconFlow + external APIs)
analyze_model_design.py   # Statistical analysis (ANOVA, OLR, FDR, ICC, PCA)
create_pca_figure.py      # PCA visualization
figures/                  # Figure generation scripts
results/vendor_exp/        # Raw experiment data
├── final_merged_20260325_230829.json    # 490 records (main data)
└── study5_direct_missing_20260326_191535.json  # 24 records (supplement)
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

Experiments require API access:

```bash
export SILICONFLOW_API_KEY="your-key"
export YIHE_API_KEY="your-key"  # Optional: for international models

python run_model_experiments.py
```

## Analysis

```bash
python analyze_model_design.py --input results/vendor_exp/final_merged_20260325_230829.json
```

Produces OLS ANOVA with FDR correction, Cohen's d effect sizes, Cronbach's alpha, ICC, convergent validity tests, and PCA.

## Key Findings

1. **Large cross-family differences**: Median pairwise Cohen's $d = 1.01$
2. **C-N reversal**: The Conscientiousness--Neuroticism trade-off flips from negative to positive ($r = +0.76$, $p < 0.001$, $n = 33$)
3. **Response compression**: All models compress toward the Likert midpoint
4. **Prompt sensitivity**: Prompt framing explains 1-9% of variance (significant but modest vs. 10-78% for model family)
5. **Alignment effect**: Base models cluster at the midpoint; alignment introduces model-specific variation

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
