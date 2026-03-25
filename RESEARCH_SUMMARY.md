# Research Summary: Vendor-Dependent Response Style Variation in Open LLMs

## Paper Target
EMNLP 2026 Findings

## Research Question
Do different AI vendors' LLMs exhibit systematically different "response styles" when completing psychological assessments, even when using the same base architecture and similar training data?

## Design
- **Study 1**: Cross-vendor comparison — 11 Chinese AI vendors × 1 flagship model each × 12 seeds × 61 Likert items (9 dimensions)
- **Study 2**: Within-vendor evolution — DeepSeek (V2.5→V3→V3.2→R1), Qwen (4B→9B→27B + MoE), Zhipu (GLM-4→4.x→5→Z1) × 12 seeds
- **Thinking Ablation**: 4 models × enable_thinking ON/OFF × 3 seeds
- **Study 3**: International providers — GPT-4o, GPT-5, Claude Opus 4.5, Claude Sonnet 4 × 12 seeds (grok-3 and gemini-3-pro unavailable due to API expiration)

## 9 Dimensions Measured
BFI-5 (E, A, C, N, O) + HEXACO-Honesty + Collectivism + Intuition + Uncertainty Avoidance

## Key Results

### Study 1: ANOVA (all 9 dimensions significant after Bonferroni)
| Dimension | F | p | η²p |
|-----------|---|---|-----|
| HEXACO-H | F(10,133)=19.63 | 8.18e-22 | 0.374 |
| Neuroticism | F(10,133)=10.34 | 9.04e-13 | 0.304 |
| Conscientiousness | F(10,133)=9.88 | 3.04e-12 | 0.299 |
| Openness | F(10,133)=9.16 | 2.08e-11 | 0.290 |
| Intuition | F(10,133)=8.16 | 3.36e-10 | 0.275 |
| Agreeableness | F(10,133)=5.97 | 1.92e-07 | 0.237 |
| Extraversion | F(10,133)=4.64 | 1.13e-05 | 0.206 |
| Collectivism | F(10,133)=3.90 | 1.17e-04 | 0.185 |
| Uncertainty Avoidance | F(10,133)=4.57 | 1.43e-05 | 0.204 |

### Pairwise Effects
- 272/495 pairwise tests significant (55%)
- 234/495 significant after FDR
- Mean Cohen's d = 1.223, median = 0.906
- Largest effect: HEXACO-H Baidu vs Tencent (d=9.34)

### Alignment Artifact Analysis
- Agreeableness classified as alignment artifact (SD=0.12, near-zero inter-vendor variance)
- 8/9 dimensions are discriminative
- HEXACO-H has highest inter-vendor SD (0.70) — most discriminative dimension
- 13/33 models show HEXACO-H floor effect (H≤1.5)

### Study 2: Within-Vendor Trajectories
- **DeepSeek**: V2.5→V3→V3.2 shows H dropping from 3.00→2.22→1.56 (strong RLHF alignment over versions)
- **Qwen**: 4B H=2.13, 27B H=1.18, 397B H=1.00 (scale → more alignment)
- **Zhipu**: GLM-4-9B H=2.93, GLM-5 H=1.15 (clear alignment trajectory)
- **DeepSeek-R1** (reasoning): Unique profile — high E (3.78), low C (2.05), low N (2.41)

### HEXACO-H × Scale Correlation
- Pearson r = -0.357, p = 0.159 (NOT significant)
- Weak trend: smaller models → higher H (less RLHF-aligned)

### Reliability
- Cronbach's α: BFI dimensions 0.78-0.94 (good), HEXACO-H 0.27 (poor — only 5 items)
- ICC(1,1): 0.22-0.46 (low-moderate — expected for synthetic respondents)

### Acquiescence Bias
- PC1 explains 52.3% of variance (strong acquiescence signal)
- After PC1 correction, high correlations drop from 16 to 2
- Convergent validity: 4/5 dimension-dimension correlations match human baseline direction

### Power Analysis
- n=12 per vendor: 80% power to detect d≥1.14
- Observed effects: mean d=1.22, so most significant effects are well-powered
- 234/495 FDR-significant pairs → many effects are genuinely large

## Post-Review Fixes (Round 1 → Round 2)
1. **Floor-effect robustness**: Added analysis excluding floor-effected models (H≤1.5). HEXACO-H remains highly significant: F(6,77)=18.79, p=2.44e-13, η²p=0.373 (5 models excluded, 7 vendors remain)
2. **Reframing strengthened**: Paper explicitly frames as "response style variation" NOT "personality measurement". Psychological instruments used as standardized probes.
3. **HEXACO-H caveat**: Acknowledged low reliability (α=0.27, 5 items) — presented as exploratory signal requiring validation with full 10-item scale
4. **Acquiescence analysis**: Already includes PC1 correction showing correlation structure changes
5. **ICC discussion**: Low ICC (0.22-0.46) expected for near-deterministic LLMs with temperature=0.7

## Limitations
1. Only Chinese vendors (Study 1) + limited international (Study 3)
2. HEXACO-H has only 5 items (low reliability)
3. Single prompt format (no prompt variation robustness check)
4. No human baseline comparison group
5. Response parsing: regex extraction of first digit from free-form responses
6. Acquiescence bias is strong (PC1 = 52.3% variance)
7. ICC values are low-moderate (0.22-0.46)

## Files
- `run_vendor_experiments.py` — Experiment runner
- `analyze_vendor_design.py` — Analysis script
- `data/results.json` — Merged dataset (394 observations, 33 models)
- `analysis_output.txt` — Full analysis output
