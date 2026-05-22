# Figure rendering code

This directory contains the Python and shell code used by Step 2 of the
PaperOrchestra pipeline (the *plotting-agent*) to produce every figure in
the paper.

## Layout

```
figure_code/
├── README.md                           ← you are here
├── render_figs.py                      ← matplotlib generator for 18 chart figures
└── render_paperbanana_diagrams.sh      ← PaperBanana commands for the 2 diagrams
```

## How to reproduce

### 1. The 18 matplotlib figures

`render_figs.py` regenerates every chart-type figure (heatmaps, bar charts,
scatter plots, beeswarms, histograms) from the precomputed CSVs in
`code/results/`. It is fully deterministic — no LLM calls.

```bash
cd /Users/zhanshaoxiong.3/Desktop/new-exp/llm-psychology/paper-orc/figure_code
python3 render_figs.py
```

The script reads CSVs from
`/Users/zhanshaoxiong.3/Desktop/new-exp/llm-psychology/code/results/` and
writes 300-DPI PNGs to
`/Users/zhanshaoxiong.3/Desktop/new-exp/llm-psychology/paper-orc/figures/`.

Output figures (ordered by section appearance):

| Figure ID | Section | Type |
|---|---|---|
| `fig_alpha_vs_reverse_density` | §4.2 Internal consistency | scatter + regression |
| `fig_acquiescence_directional_asymmetry` | §4.3 Item-level acquiescence | grouped bar |
| `fig_efa_loadings_heatmap` | §4.4 Factor collapse | annotated heatmap |
| `fig_efa_loo_robustness` | App A | bar + LOO scatter |
| `fig_variance_decomposition_likert_binary` | App A | stacked bars |
| `fig_item_plasticity_distribution` | §4.8 Persona stress | beeswarm |
| `fig_persona_separation_degree` | App A | sorted bar with error |
| `fig_within_sample_consistency_by_model` | App D | sorted bar + line |
| `fig_persona_fidelity_confusion_heatmap` | §4.8 | 16×16 heatmap |
| `fig_measurement_invariance_histogram` | App A | histogram + KDE |
| `fig_default_persona_classification` | App A | grouped bar |
| `fig_pir_default_vs_mbti` | App A | paired bar + per-model bars |
| `fig_target_adherence_heatmap` | App A | 16×2 heatmap |
| `fig_factorial_mbti_main_effects` | App A | 4×17 heatmap |
| `fig_cross_scale_coherence_by_construct` | App A | grouped bar |
| `fig_alpha_per_model_heatmap` | App A | 17×5 heatmap |
| `fig_alpha_synthetic_baseline_calibration` | App B (DIF) | grouped bar |
| `fig_refusal_breakdown` | App C | bar + 2 pies |

The script's global style block (`plt.rcParams.update(...)`) is the
academic-paper preset used uniformly across all figures: serif fonts (Times
New Roman / DejaVu Serif fallback), 8 pt body, 7 pt legend/tick, 0.6 pt axis
linewidth, 300 DPI export, muted print-safe palette
(`BLUE / RED / GREEN / ORANGE / PURPLE / GOLD / GRAY / TEAL`).

### 2. The 2 conceptual diagrams (PaperBanana)

`render_paperbanana_diagrams.sh` invokes the PaperBanana backbone
(Zhu et al., 2026 — see `~/paper-orchestra/skills/plotting-agent/references/paperbanana-cookbook.md`)
to produce the two block-diagram figures:

| Figure ID | Section | Aspect | Critic rounds |
|---|---|---|---|
| `fig_causal_chain_overview` | §1 teaser | 16:9 | 1 |
| `fig_validation_battery_schema` | App A | 16:9 | 1 |

```bash
chmod +x render_paperbanana_diagrams.sh
./render_paperbanana_diagrams.sh
```

Each call goes through PaperBanana's Retriever → Planner → Stylist →
Visualizer → Critic loop (1 critic round here). The captions passed via
`--caption` are the same ones recorded in
`workspace/figures/captions.json`.

## Data dependencies

`render_figs.py` reads the following CSVs (paths are absolute in the script):

```
code/results/
├── cronbach_alpha_by_domain.csv          (§4.2)
├── cronbach_alpha_by_persona.csv         (App A heatmap)
├── acquiescence_mechanism.csv            (§4.3)
├── synthetic_baselines.csv               (App B)
├── efa_domain_loadings.csv               (§4.4)
├── efa_eigenvalues.csv                   (App A LOO robustness)
├── leave_one_out_robustness.csv          (App A)
├── variance_decomposition.csv            (App A)
├── item_plasticity.csv                   (§4.8)
├── persona_target_adherence.csv          (App A: PSD + adherence)
├── persona_fidelity_confusion.csv        (§4.8)
├── persona_invariance.csv                (App A)
├── default_bias_index.csv                (App A)
├── pir_by_persona.csv                    (App A)
├── pir_by_model_domain.csv               (App A right panel)
├── factorial_mbti_effects.csv            (App A)
├── cross_scale_persona_coherence_summary.csv (App A)
└── wsc_by_model.csv                      (App D)
```

Refusal counts for `fig_refusal_breakdown` are encoded directly in the
script (sourced from `experimental_log.md` §2.15) since the per-cell refusal
log was not exported as a single CSV.

## Reproducibility notes

- Random seed for the beeswarm jitter is fixed via
  `np.random.default_rng(7)` so layouts are byte-stable across re-runs.
- PNG output is 300 DPI with `bbox_inches='tight'` and `pad_inches=0.08`;
  this matches the figure-aspect specs in `outline.json` precisely.
- Two figures rely on the `TwoSlopeNorm` colour normalisation
  (`fig_efa_loadings_heatmap`, `fig_factorial_mbti_main_effects`,
  `fig_alpha_per_model_heatmap`) — change of vmin/vmax shifts colour
  semantics, so keep those values stable when re-rendering.
- One Unicode glyph (`≪`, U+226A "MUCH LESS-THAN") is missing from
  Times New Roman; it produces a benign UserWarning during savefig and
  does not affect the rendered figures (which use ASCII alternatives).
