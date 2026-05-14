# LLM Personality Consistency: Do Language Models Exhibit Internally Consistent Personality Traits?

Research on whether LLMs maintain internally consistent personality profiles when responding to psychometric instruments with built-in consistency checks (reverse-scored items, positively/negatively correlated subscales).

---

## Research Question

Validated psychometric scales contain **internal consistency checks**: some items are positively keyed (agree = high score), others are reverse-scored (agree = low score). In humans, a respondent who scores high on "I worry about things" should score low on "I rarely feel anxious" — these are negatively correlated items measuring the same construct.

**Do LLMs respect these consistency constraints?**

Our hypothesis: LLMs tend toward a "moderate, prosocial" response style driven by alignment, producing responses that lack the internal consistency seen in real human personality profiles. A model might simultaneously:
- Agree with "I am always honest" (prosocial alignment)
- Also agree with "I can be manipulative when needed" (because it seems reasonable)
- These are contradictory — a genuinely honest person would disagree with the second statement

---

## Psychometric Battery

We use **4 validated psychometric scales** totaling **221 items** across **17 dimensions**:

| Scale | Items | Format | Dimensions | Reverse Items | Source |
|-------|-------|--------|------------|---------------|--------|
| IPIP-NEO-120 | 120 | 5-point Likert | 5 domains x 6 facets | 41 | Johnson (2014), public domain |
| SD3 (Short Dark Triad) | 27 | 5-point Likert | 3 (Machiavellianism, Narcissism, Psychopathy) | 5 | Jones & Paulhus (2014) |
| ZKPQ-50-CC | 50 | True/False | 5 | 12 | Aluja et al. (2006) |
| EPQR-A | 24 | Yes/No | 4 (Psychoticism, Extraversion, Neuroticism, Lie) | 5 | Francis et al. (1992) |

Key property: **63 out of 221 items (29%) are reverse-scored**, creating built-in consistency checks. If a model truly "has" a personality trait, its responses to forward and reverse items should be internally consistent after scoring.

See [`data/BATTERY_SPECIFICATION.md`](data/BATTERY_SPECIFICATION.md) for full scale documentation.

---

## Experiment Design

### Conditions

Each model is tested under **17 persona conditions**:
- **Default**: No persona instruction (baseline)
- **16 MBTI types**: ISTJ, ISFJ, INFJ, INTJ, ISTP, ISFP, INFP, INTP, ESTP, ESFP, ENFP, ENTP, ESTJ, ESFJ, ENFJ, ENTJ

### Protocol

- All 221 items administered at **temperature = 0.7**
- Items are framed as self-referential statements with appropriate response scales
- Each model x persona combination produces a full response profile

### Models Tested (18 models)

| # | Model | Architecture |
|---|-------|-------------|
| 1 | Qwen3.5-397B-A17B | MoE (397B) |
| 2 | Qwen3.5-122B-A10B | MoE (122B) |
| 3 | Qwen3-235B-A22B | MoE (235B) |
| 4 | DeepSeek-V3.2 | MoE (671B) |
| 5 | DeepSeek-V4-Flash | MoE |
| 6 | DeepSeek-V4-Pro | MoE |
| 7 | GLM-4.6V | - |
| 8 | Kimi-K2.5 | MoE (1.1T) |
| 9 | Kimi-K2.6 | MoE |
| 10 | MiniMax-M2.7 | MoE (230B) |
| 11 | Gemini-3-Pro-Preview | - |
| 12 | Gemini-3-Flash-Preview | - |
| 13 | Gemini-3.1-Pro-Preview | - |
| 14 | Gemini-3.1-Flash-Lite | - |
| 15 | GPT-5.2 | - |
| 16 | GPT-5.5 | - |
| 17 | Claude Opus 4.6 | - |
| 18 | Claude Sonnet 4.6 | - |

---

## Data

### Results

| File | Description |
|------|-------------|
| `results/exp_mbti_*.json` | Full experiment results per model (18 files) |
| `results/summary.csv` | Aggregated scores: model x persona x scale x domain |

### Scale Data

| File | Description |
|------|-------------|
| `data/items_battery.json` | All 221 items with metadata (scale, domain, facet, keyed direction, item text) |
| `data/BATTERY_SPECIFICATION.md` | Detailed scale documentation with all items, scoring rules, and reverse-item mappings |
| `data/build_battery.py` | Script to build items_battery.json from source scales |

---

## Repository Structure

```
data/
  BATTERY_SPECIFICATION.md   # Full scale documentation
  items_battery.json          # 221 items with scoring metadata
  build_battery.py            # Battery construction script
exp-code/
  run_mbti_experiment.py      # MBTI persona experiment runner
  postprocess_mbti.py         # Post-processing for MBTI results
results/
  exp_mbti_*.json             # Raw experiment results (18 models)
  summary.csv                 # Aggregated scores
run_model_experiments.py      # Main experiment runner (V4.0)
requirements.txt              # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
export SILICONFLOW_API_KEY="your-key"

# Run default (no persona) experiment
python run_model_experiments.py

# Run MBTI persona experiments
python exp-code/run_mbti_experiment.py
```

## Analysis Plan

1. **Within-scale consistency**: For each scale, check if forward and reverse items produce consistent scores after reverse-scoring. A perfectly consistent respondent should show high correlation between forward and reverse subscores.

2. **Cross-scale trait consistency**: If a model scores high on Neuroticism (IPIP-NEO-120), it should also score high on Neuroticism-Anxiety (ZKPQ-50-CC) — these measure overlapping constructs.

3. **Dark Triad vs. prosocial alignment**: SD3 measures Machiavellianism, Narcissism, and Psychopathy. LLMs aligned to be helpful/harmless should score low — but do their responses to individual items show internal contradictions?

4. **Lie scale detection**: EPQR-A includes a Lie scale (social desirability items). High Lie scores in LLMs would indicate response distortion toward social acceptability.

5. **Persona consistency**: Under MBTI personas, do models produce internally consistent profiles that match the theoretical MBTI trait predictions?

## License

CC-BY-4.0
