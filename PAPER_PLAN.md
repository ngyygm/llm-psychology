# Paper Plan: EMNLP 2026 Findings

## Title
**Proposed**: "Response Style Variation Across AI Vendors: Evidence from Cross-Vendor Psychometric Probes"

## Key Changes from Previous Version
The previous version (emnlp2026_improved.tex) had a different framing ("Language Models Do Not Share Human Personality Structure") and included HEXACO-H as a primary dimension. This version addresses all review feedback from the 3-round auto-review loop (3→5→6/10).

### Critical Changes
1. **Reframing**: "Response style variation" not "personality structure breakdown"
2. **HEXACO-H demoted**: Removed from primary ANOVA; reported as descriptive observation only (floor effect in 5/12 Study 1 models)
3. **Primary analysis**: 8 dimensions (BFI-5 + Collectivism + Intuition + UA), all α > 0.70
4. **Updated numbers**: 11 Chinese vendors (not 15), 144 Study 1 obs, 192 Study 2 obs
5. **Study 2 expanded**: Now 16 models across 3 vendors with clear trajectory analysis
6. **Alignment artifact analysis**: Agreeableness classified as artifact (SD=0.12)
7. **Reviewer concerns addressed**: Neutral ICC, single-prompt caveat, generalizability discussion

### Section Structure (4 pages + references)
1. **Abstract** (~180 words)
2. **Introduction** (~0.8 page) — response style variation, alignment training hypothesis
3. **Related Work** (~0.5 page) — LLM personality, response biases, our contribution
4. **Method** (~0.8 page) — instruments, model selection, protocol, analysis
5. **Experiments** (~1.5 pages) — Study 1 ANOVA, pairwise effects, Study 2 trajectories, HEXACO-H observation
6. **Conclusion + Limitations** (~0.4 page)

### Figures (reuse existing + 1 new)
- Fig 1: Radar profiles (reuse fig1_radar_combined.pdf — regenerate with 11 vendors, 8 dims)
- Fig 2: Cohen's d heatmap (reuse fig4_cohen_d_heatmap.pdf — regenerate)
- Fig 3: Inter-dimension correlation (reuse fig7_inter_dim_corr.pdf — regenerate for 8 dims)
- Fig 4: Study 2 trajectories (reuse fig_study2_trajectories_combined.pdf)

### Tables
- Table 1: Instruments (61 items, 9 dimensions — keep as-is, note HEXACO-H is exploratory)
- Table 2: Study 1 ANOVA results (8 dimensions)
- Table 3: Study 2 trajectory summary
- Table 4: Reliability (8 primary dimensions)

### Key Numbers (updated)
- Study 1: 11 vendors × 12 seeds = 132 observations
- All 8 dimensions: Bonferroni significant (p < 1.4e-04)
- η²p range: 0.185 (Collectivism) to 0.304 (Neuroticism)
- 192/440 pairwise tests significant after FDR
- Median Cohen's d = 0.84
- Agreeableness: alignment artifact (inter-vendor SD = 0.12)
- HEXACO-H: 5/12 Study 1 models at floor (H≤1.5); non-floor models still differ (p=5.6e-09)
- Study 2: DeepSeek V2.5→V3.2 H drops 3.00→1.56; Qwen 4B→397B H drops 2.13→1.00

### Claims-Evidence Matrix
| Claim | Evidence | Section |
|-------|---------|---------|
| Vendors have distinct response styles | ANOVA: all 8 dims p<0.001, median d=0.84 | §5.1 |
| Effect sizes are substantial | η²p=0.19-0.30, max d=6.21 | §5.1 |
| Alignment shapes response style | Agreeableness artifact, HEXACO-H floor effects, Study 2 trajectories | §5.2-5.3 |
| Acquiescence is present but doesn't explain vendor differences | PC1=52%, but KW tests still significant | §5.4 |
| Response styles evolve within vendors | Study 2: clear trajectories for DeepSeek, Qwen, Zhipu | §5.2 |
| Instruments have adequate reliability | Cronbach's α=0.74-0.94 for all primary dims | §4.4 |

### Reviewer-Required Framing
1. "Response style" not "personality" throughout
2. HEXACO-H: "We piloted HEXACO-H (5 items, α=0.27) and observed floor effects in 42% of Study 1 models, consistent with strong RLHF alignment."
3. ICC: "ICC(1,1) values of 0.22-0.46 indicate moderate within-model consistency across random seeds, consistent with alignment-trained models producing context-dependent responses."
4. Single prompt: "We use a single standardized prompt format; response patterns may vary with alternative framings."
5. Generalizability: "Our sample focuses on Chinese AI vendors; generalization to Western vendors requires further investigation."
