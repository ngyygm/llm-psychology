#!/usr/bin/env python3
"""Generate 3x3 grid: all 9 dimensions vs total parameters (appendix figure).
Rules match regen_figures.py Fig 6 (Study 1 only, same MODEL_PARAMS dict).
"""

import sys
sys.path.insert(0, '/home/linkco/exa/llm-psychology/figures')
from paper_plot_style import *

import json, glob, numpy as np, pandas as pd
from scipy import stats
from matplotlib.patches import Patch

print("Loading data...")
s1 = load_study1_data()

# 15 representative models (same as regen_figures.py)
REPR_MODELS = [
    "Qwen/Qwen3.5-397B-A17B",
    "Pro/deepseek-ai/DeepSeek-V3.2",
    "Pro/zai-org/GLM-5",
    "Pro/moonshotai/Kimi-K2.5",
    "baidu/ERNIE-4.5-300B-A47B",
    "tencent/Hunyuan-A13B-Instruct",
    "ByteDance-Seed/Seed-OSS-36B-Instruct",
    "internlm/internlm2_5-7b-chat",
    "inclusionAI/Ring-flash-2.0",
    "stepfun-ai/Step-3.5-Flash",
    "ascend-tribe/pangu-pro-moe",
    "Kwaipilot/KAT-Dev",
    "Pro/MiniMaxAI/MiniMax-M2.5",
    "gpt-5",
    "claude-opus-4-5-20251101",
]
s1 = s1[s1['model'].isin(REPR_MODELS)]

# Total parameters (Billions) — same mapping as regen_figures.py MODEL_PARAMS
# Values are TOTAL params, not active params
MODEL_TOTAL_PARAMS = {
    # Study 1 models — disclosed in Table 2
    "Qwen/Qwen3.5-397B-A17B": 397,
    "Pro/deepseek-ai/DeepSeek-V3.2": 671,
    "Pro/zai-org/GLM-5": 744,
    "Pro/moonshotai/Kimi-K2.5": 1100,
    "baidu/ERNIE-4.5-300B-A47B": 300,
    "tencent/Hunyuan-A13B-Instruct": 13,
    "ByteDance-Seed/Seed-OSS-36B-Instruct": 36,
    "internlm/internlm2_5-7b-chat": 7,
    # Undisclosed — set to 0 to skip in regression, but plot as open circles
    "inclusionAI/Ring-flash-2.0": 100,
    "stepfun-ai/Step-3.5-Flash": 197,
    "ascend-tribe/pangu-pro-moe": 72,
    "Kwaipilot/KAT-Dev": 32,
    "Pro/MiniMaxAI/MiniMax-M2.5": 230,
    # Truly undisclosed
    "gpt-5": 0,
    "claude-sonnet-4-20250514": 0,
    "claude-opus-4-5-20251101": 0,
    "gemini-3-pro": 0,
}

VENDOR_TO_FAMILY = {
    "Baidu": "ERNIE", "ByteDance": "Seed", "DeepSeek": "DeepSeek",
    "Huawei": "Pangu", "inclusionAI": "Ring", "InternLM": "InternLM",
    "Kwaipilot": "KAT", "MiniMax": "MiniMax", "Moonshot": "Kimi",
    "Qwen": "Qwen", "StepFun": "Step", "Tencent": "Hunyuan", "Zhipu": "GLM",
    "OpenAI": "GPT", "Anthropic": "Claude", "Gemini": "Gemini", "Grok": "Grok",
}
FAMILY_COLORS = {VENDOR_TO_FAMILY[v]: c for v, c in C_VENDOR_COLORS.items() if v in VENDOR_TO_FAMILY}

# Compute model-level means
model_means = s1.groupby("model")[DIMENSIONS].mean()

# Build lists — same logic as regen_figures.py Fig 6
# Skip models with undisclosed params (params = 0)
params_list = []
family_list = []
model_ids = []

for model_id in model_means.index:
    params = MODEL_TOTAL_PARAMS.get(model_id, 0)
    if params <= 0:
        continue
    vdata = s1[s1["model"] == model_id]
    vname = vdata["vendor"].iloc[0] if len(vdata) > 0 else ""
    family = VENDOR_TO_FAMILY.get(vname, vname)
    params_list.append(params)
    family_list.append(family)
    model_ids.append(model_id)

print(f"Models with disclosed total params: {len(model_ids)}")

def short_name(model_id):
    return model_id.split("/")[-1].replace("-Instruct", "").replace("-0414", "")

DIM_TITLES = [
    'Extraversion', 'Agreeableness', 'Conscientiousness',
    'Neuroticism', 'Openness', 'HEXACO-H',
    'Collectivism', 'Intuition', 'Uncertainty Avoidance',
]

fig, axes = plt.subplots(3, 3, figsize=(10, 8.5))

for idx, dim in enumerate(DIMENSIONS):
    ax = axes[idx // 3][idx % 3]

    # Get dimension values for models with disclosed params
    dim_vals = []
    p_vals = []
    f_vals = []
    for j, mid in enumerate(model_ids):
        if mid in model_means.index and not np.isnan(model_means.loc[mid, dim]):
            dim_vals.append(model_means.loc[mid, dim])
            p_vals.append(params_list[j])
            f_vals.append(family_list[j])

    dim_vals = np.array(dim_vals)
    p_vals = np.array(p_vals)

    # Scatter colored by family
    unique_fams = sorted(set(f_vals))
    for fam in unique_fams:
        fidx = [j for j, ff in enumerate(f_vals) if ff == fam]
        color = FAMILY_COLORS.get(fam, '#999999')
        ax.scatter([p_vals[j] for j in fidx], [dim_vals[j] for j in fidx],
                   color=color, s=40, edgecolors='black', linewidths=0.4, zorder=3)

    # Regression line
    if len(p_vals) >= 3:
        slope, intercept, r_val, p_val, _ = stats.linregress(p_vals, dim_vals)
        x_range = np.linspace(min(p_vals) * 0.8, max(p_vals) * 1.1, 50)
        ax.plot(x_range, slope * x_range + intercept, 'k--', linewidth=0.8, alpha=0.5)

        if p_val < 0.05:
            sig = f'p<0.05' if p_val >= 0.01 else ('p<0.01' if p_val >= 0.001 else 'p<0.001')
            label = f'r = {r_val:.2f} ({sig})'
        else:
            label = f'r = {r_val:.2f} (n.s.)'

        txt_color = '#C44E52' if p_val < 0.05 else '#666666'
        ax.text(0.97, 0.95, label, transform=ax.transAxes, fontsize=7.5,
                va='top', ha='right', color=txt_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    ax.set_title(DIM_TITLES[idx], fontsize=9, fontweight='bold', pad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if idx // 3 == 2:
        ax.set_xlabel('Total Parameters (Billions)', fontsize=8)
    else:
        ax.set_xlabel('')

    if idx % 3 == 0:
        ax.set_ylabel('Score', fontsize=8)

    ax.tick_params(labelsize=7)

# Legend — one row at the bottom, all families
unique_fams_all = sorted(set(family_list))
legend_handles = [Patch(facecolor=FAMILY_COLORS.get(fam, '#999999'),
                        edgecolor='black', linewidth=0.4, label=fam)
                  for fam in unique_fams_all]
fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), fontsize=6.5,
           frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0)

plt.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig, 'appendix_9dim_params_grid')
print("Done!")
