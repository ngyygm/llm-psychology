#!/usr/bin/env bash
# render_paperbanana_diagrams.sh
# ---------------------------------------------------------------------------
# Step 2 of the PaperOrchestra pipeline (App. F.1, Plotting Agent).
# Renders the two conceptual diagrams (plot_type == "diagram") via the
# PaperBanana backbone (Zhu et al., 2026) which runs a Retriever → Planner →
# Stylist → Visualizer → Critic loop. Used for figures whose semantics are
# better expressed as a publication-grade conceptual block diagram than as a
# matplotlib chart.
#
# Prerequisites:
#   - paper-orchestra skill installed at ~/paper-orchestra
#   - GEMINI / nano-banana / corresponding API key configured
#
# Usage: ./render_paperbanana_diagrams.sh
# ---------------------------------------------------------------------------

set -euo pipefail

WORKSPACE="/Users/zhanshaoxiong.3/Desktop/new-exp/llm-psychology/paper-orc"
PB_HOME="$HOME/paper-orchestra"

mkdir -p "$WORKSPACE/figures"

# ============================================================
# Figure 1: fig_causal_chain_overview (16:9 diagram)
# ============================================================
cd "$PB_HOME" && python3 skills/plotting-agent/scripts/paperbanana_render.py \
  --figure-id fig_causal_chain_overview \
  --caption "A four-stage causal chain that explains why every standard psychometric validity check fails on large language models. Stage 1 (Item-level acquiescence): alignment training produces an agree-impulse with directional asymmetry — IPIP normal-personality items Delta = +0.29, SD3 dark-trait items Delta = -0.33. Stage 2 (Domain-level): reverse-keyed items behave like forward items so Cronbach alpha tracks reverse-item density at Spearman rho = -1.0 across the IPIP Big Five. Stage 3 (Structural): exploratory factor analysis collapses the expected 5 Big Five factors into 3 universally across 20 LLMs (Tucker phi = 0.993). Stage 4 (Model-level): inter-model variance is below 1% on Likert and binary scales, so cross-model personality comparisons are statistically meaningless. A side panel labels the diagnostic battery: 4 instruments, 17 domains, 20 LLMs x 17 personas x k=5 = 375,700 API calls. Arrows connect the stages from left to right." \
  --content-file "$WORKSPACE/inputs/idea.md" \
  --task diagram \
  --aspect-ratio 16:9 \
  --max-critic-rounds 1 \
  --out "$WORKSPACE/figures/fig_causal_chain_overview.png"

# ============================================================
# Figure 2: fig_validation_battery_schema (16:9 diagram)
# ============================================================
cd "$PB_HOME" && python3 skills/plotting-agent/scripts/paperbanana_render.py \
  --figure-id fig_validation_battery_schema \
  --caption "Diagnostic schema for the six-check psychometric validation battery plus persona stress test applied to 20 LLMs on a 221-item battery (IPIP-NEO-120, SD3, ZKPQ-50-CC, EPQR-A; 17 domains; 63 reverse-keyed items). Top row: Inputs (4 instruments x 17 domains; 20 LLMs across 8 vendors; 17 conditions = Default + 16 MBTI personas; k=5 samples per cell, T=0.7). Middle row: Six classical validity checks (Internal Consistency / Cronbach alpha, Factor Structure / EFA, Reverse-Item Consistency / PIR, Variance Decomposition, Convergent Validity, Measurement Invariance). Each check displays a red FAIL badge for LLMs with a one-line failure summary (alpha -> Spearman rho=-1 with reverse density; EFA -> 5->3 collapse, Tucker phi=0.993; PIR=0.468; Model SS<1%; 7/8 sign-match but weak r; 0/320 strong invariance). Bottom row: Persona Stress Test (PSD, fidelity, target adherence, plasticity, factorial MBTI main effects, within-sample stability) reporting 99.7% recovery vs 0.839 cross-scale coherence -> compliance without validity." \
  --content-file "$WORKSPACE/inputs/experimental_log.md" \
  --task diagram \
  --aspect-ratio 16:9 \
  --max-critic-rounds 1 \
  --out "$WORKSPACE/figures/fig_validation_battery_schema.png"

echo "Both PaperBanana diagrams rendered to $WORKSPACE/figures/"
