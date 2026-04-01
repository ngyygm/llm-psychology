#!/usr/bin/env python3
"""Regenerate all main-body figures with consistent style.
- Combined radar (15 families, one per model)
- PCA scatter (all models, Study 1 + Study 2)
- SD bar chart (15 representative models)
- Cohen's d heatmap (15 representative models)
- HEXACO-H models dot plot (15 representative models)
- HEXACO-H vs total params scatter (all models, architecture colors)
- Inter-dim correlation matrix (15 representative models)
"""

import sys
sys.path.insert(0, '/home/linkco/exa/llm-psychology/figures')
from paper_plot_style import *

import json, glob, numpy as np, pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 15 representative models (one per family, as in paper Table 2)
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

# Architecture map (matches paper Table 2)
ARCH_MAP = {
    "Qwen/Qwen3.5-397B-A17B": "MoE",
    "Pro/deepseek-ai/DeepSeek-V3.2": "MoE",
    "Pro/zai-org/GLM-5": "MoE",
    "Pro/moonshotai/Kimi-K2.5": "MoE",
    "baidu/ERNIE-4.5-300B-A47B": "MoE",
    "tencent/Hunyuan-A13B-Instruct": "Dense",
    "ByteDance-Seed/Seed-OSS-36B-Instruct": "Dense",
    "internlm/internlm2_5-7b-chat": "Dense",
    "inclusionAI/Ring-flash-2.0": "MoE",
    "stepfun-ai/Step-3.5-Flash": "MoE",
    "ascend-tribe/pangu-pro-moe": "MoE",
    "Kwaipilot/KAT-Dev": "Dense",
    "Pro/MiniMaxAI/MiniMax-M2.5": "MoE",
    "gpt-5": "Undisclosed",
    "claude-opus-4-5-20251101": "Undisclosed",
    # Study 2 models
    "Qwen/Qwen3.5-4B": "Dense",
    "ollama:qwen3.5:9b": "Dense",
    "Qwen/Qwen3.5-27B": "Dense",
    "Qwen/Qwen3.5-35B-A3B": "MoE",
    "Qwen/Qwen3.5-122B-A10B": "MoE",
    "deepseek-ai/DeepSeek-V2.5": "Dense",
    "deepseek-ai/DeepSeek-V3": "MoE",
    "deepseek-ai/DeepSeek-V3.2": "MoE",
    "deepseek-ai/DeepSeek-R1": "MoE",
    "THUDM/GLM-4-9B-0414": "Dense",
    "THUDM/GLM-4-32B-0414": "Dense",
    "zai-org/GLM-4.5-Air": "Dense",
    "zai-org/GLM-4.6": "Dense",
    "THUDM/GLM-Z1-32B-0414": "Dense",
}

# Total parameters in billions (matches paper Table 2)
TOTAL_PARAMS = {
    # Study 1 representative
    "Qwen/Qwen3.5-397B-A17B": 397,
    "Pro/deepseek-ai/DeepSeek-V3.2": 671,
    "Pro/zai-org/GLM-5": 744,
    "Pro/moonshotai/Kimi-K2.5": 1100,
    "baidu/ERNIE-4.5-300B-A47B": 300,
    "tencent/Hunyuan-A13B-Instruct": 13,
    "ByteDance-Seed/Seed-OSS-36B-Instruct": 36,
    "internlm/internlm2_5-7b-chat": 7,
    "inclusionAI/Ring-flash-2.0": 100,
    "stepfun-ai/Step-3.5-Flash": 197,
    "ascend-tribe/pangu-pro-moe": 72,
    "Kwaipilot/KAT-Dev": 32,
    "Pro/MiniMaxAI/MiniMax-M2.5": 230,
    # Study 2
    "Qwen/Qwen3.5-4B": 4,
    "ollama:qwen3.5:9b": 9,
    "Qwen/Qwen3.5-27B": 27,
    "Qwen/Qwen3.5-35B-A3B": 35,
    "Qwen/Qwen3.5-122B-A10B": 122,
    "deepseek-ai/DeepSeek-V2.5": 21,  # Active params (21B active out of 236B total)
    "deepseek-ai/DeepSeek-V3": 671,
    "deepseek-ai/DeepSeek-V3.2": 671,
    "deepseek-ai/DeepSeek-R1": 671,
    "THUDM/GLM-4-9B-0414": 9,
    "THUDM/GLM-4-32B-0414": 32,
    "zai-org/GLM-4.5-Air": 10,
    "zai-org/GLM-4.6": 12,
    "THUDM/GLM-Z1-32B-0414": 32,
}

print("Loading data...")
s1 = load_study1_data()
s2 = load_study2_data()
# Filter Study 1 to only 15 representative models
s1 = s1[s1['model'].isin(REPR_MODELS)]
all_chat = pd.concat([s1, s2])
print(f"Study 1: {len(s1)} obs ({s1['model'].nunique()} models), Study 2: {len(s2)} obs")

DIM_LABELS = [DIM_SHORT[d] for d in DIMENSIONS]
N_DIMS = len(DIMENSIONS)

# Model name -> Model family name (readers know models, not companies)
MODEL_TO_FAMILY = {
    "Baidu": "ERNIE", "ByteDance": "Seed", "DeepSeek": "DeepSeek",
    "Huawei": "Pangu", "inclusionAI": "Ring", "InternLM": "InternLM",
    "Kwaipilot": "KAT", "MiniMax": "MiniMax", "Moonshot": "Kimi",
    "Qwen": "Qwen", "StepFun": "Step", "Tencent": "Hunyuan", "Zhipu": "GLM",
    "OpenAI": "GPT", "Anthropic": "Claude", "Gemini": "Gemini", "Grok": "Grok",
}
# Family -> color (inherit from model colors)
FAMILY_COLORS = {MODEL_TO_FAMILY[v]: c for v, c in C_MODEL_COLORS.items() if v in MODEL_TO_FAMILY}

# ============================
# Fig 1: Combined Radar — all 15 model families in ONE plot
# ============================
print("\n--- Fig 1: Combined Radar ---")

all_models = sorted(s1['model'].unique())
angles = np.linspace(0, 2 * np.pi, N_DIMS, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

for idx, model in enumerate(all_models):
    vdata = s1[s1['model'] == model]
    means = [vdata[d].mean() for d in DIMENSIONS]
    means_plot = means + means[:1]
    family = MODEL_TO_FAMILY.get(model, model)
    color = FAMILY_COLORS.get(family, COLORS[idx % 10])
    ax.plot(angles, means_plot, 'o-', linewidth=1.4, color=color,
            markersize=3.5, label=family, alpha=0.85)
    ax.fill(angles, means_plot, alpha=0.06, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(DIM_LABELS, fontsize=8)
ax.set_ylim(1, 4.2)
ax.set_yticks([2, 3, 4])
ax.set_yticklabels(['2', '3', '4'], fontsize=7, color='gray')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), fontsize=7,
          framealpha=0.95, edgecolor='#cccccc', ncol=5)
plt.tight_layout()
save_fig(fig, 'fig1_radar_combined')


# ============================
# Fig 2: PCA — all models (Study 1 + Study 2), consistent style
# ============================
print("\n--- Fig 2: PCA Scatter ---")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For PCA, use ALL study 1 models (not just 15 repr) + Study 2
s1_all = load_study1_data()
s1_all = s1_all[s1_all['model'] != 'gemini-3-pro']  # exclude 1-seed model
all_for_pca = pd.concat([s1_all, s2]).drop_duplicates(subset=['model', 'seed'])

# Build model-level means (all models)
model_means = all_for_pca.groupby("model")[DIMENSIONS].mean()
model_map = all_for_pca.groupby("model")["model"].first()

valid = model_means.dropna()
X = valid.values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Map model ID -> model family name (what users actually know)
def model_family(model_id):
    mid = model_id.split("/")[-1].lower()
    if "qwen" in mid:          return "Qwen"
    if "deepseek" in mid:       return "DeepSeek"
    if "glm" in mid:            return "GLM"
    if "ernie" in mid:          return "ERNIE"
    if "hunyuan" in mid:        return "Hunyuan"
    if "seed" in mid:           return "Seed"
    if "internlm" in mid:       return "InternLM"
    if "ring" in mid:           return "Ring"
    if "step" in mid:           return "Step"
    if "pangu" in mid:          return "Pangu"
    if "kat" in mid:            return "KAT"
    if "minimax" in mid:        return "MiniMax"
    if "kimi" in mid:           return "Kimi"
    if "gpt" in mid:            return "GPT"
    if "claude" in mid:         return "Claude"
    if "gemini" in mid:         return "Gemini"
    if "grok" in mid:           return "Grok"
    return mid[:10]

# Short model names for annotation
MODEL_SHORT = {}
for m in valid.index:
    name = m.split("/")[-1]
    name = name.replace("-Instruct", "").replace("-0414", "")
    MODEL_SHORT[m] = name

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot each model family once (no duplicate legend entries)
family_map = {m: model_family(m) for m in valid.index}
unique_families = sorted(set(family_map.values()))
for fam in unique_families:
    fidx = [i for i, m in enumerate(valid.index) if family_map[m] == fam]
    color = FAMILY_COLORS.get(fam, '#999999')
    ax.scatter(X_pca[fidx, 0], X_pca[fidx, 1], c=color, s=120, alpha=0.85,
               edgecolors='black', linewidths=0.6, zorder=5, label=fam)

# Annotate with short names, use adjustText to avoid overlaps
from adjustText import adjust_text
texts = []
for i, m in enumerate(valid.index):
    texts.append(ax.text(X_pca[i, 0], X_pca[i, 1], MODEL_SHORT[m],
                         fontsize=5.5, ha='center', va='bottom', alpha=0.9, zorder=6))
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='#999999', lw=0.4),
            only_move={'points': 'y', 'texts': 'xy'}, force_text=(0.5, 0.5),
            force_points=(0.3, 0.3), lim=30)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), fontsize=7,
          framealpha=0.9, edgecolor='#cccccc', ncol=5)
ax.grid(True, alpha=0.15, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig, 'fig2_pca')

print(f"  PCA: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%, "
      f"Total={sum(pca.explained_variance_ratio_[:2])*100:.1f}%")


# ============================
# Fig 3: Inter-Model SD Bar Chart
# ============================
print("\n--- Fig 3: Inter-Model SD ---")

model_means_all = s1.groupby("model")[DIMENSIONS].mean()
dim_sds = model_means_all.std()
sorted_dims = sorted(DIMENSIONS, key=lambda d: dim_sds[d], reverse=True)
bar_colors = [C_ALIGNMENT if dim_sds[d] < 0.15 else C_DISCRIMINATIVE for d in sorted_dims]

fig, ax = plt.subplots(figsize=(5.5, 3))
labels = [DIM_AXIS[d] for d in sorted_dims]
bars = ax.bar(labels, [dim_sds[d] for d in sorted_dims], color=bar_colors,
              edgecolor='black', linewidth=0.5)
for bar, d in zip(bars, sorted_dims):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{dim_sds[d]:.2f}', ha='center', va='bottom', fontsize=8)

ax.axhline(y=0.15, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.text(len(labels) - 0.5, 0.16, 'Threshold', fontsize=7, color='gray', ha='right')
ax.set_ylabel('Inter-Model SD', fontsize=10)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
legend_elements = [Patch(facecolor=C_DISCRIMINATIVE, edgecolor='black', label='Discriminative'),
                   Patch(facecolor=C_ALIGNMENT, edgecolor='black', label='Alignment Artifact')]
ax.legend(handles=legend_elements, frameon=False, fontsize=8)
plt.tight_layout()
save_fig(fig, 'fig3_intermodel_sd')


# ============================
# Fig 4: Cohen's d Heatmap
# ============================
print("\n--- Fig 4: Cohen's d Heatmap ---")

models_sorted = sorted(s1['model'].unique())
n_m = len(models_sorted)
d_matrix = np.zeros((n_m, n_m))

for dim in DIMENSIONS:
    d_dim = np.zeros((n_m, n_m))
    for i in range(n_m):
        for j in range(i + 1, n_m):
            g1 = s1[s1['model'] == models_sorted[i]][dim].dropna().values
            g2 = s1[s1['model'] == models_sorted[j]][dim].dropna().values
            if len(g1) < 2 or len(g2) < 2:
                continue
            pooled_sd = np.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + (len(g2)-1)*np.var(g2, ddof=1))
                                / (len(g1) + len(g2) - 2))
            if pooled_sd == 0:
                continue
            d_dim[i, j] = d_dim[j, i] = abs(np.mean(g1) - np.mean(g2)) / pooled_sd
    d_matrix = np.maximum(d_matrix, d_dim)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(d_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(3, d_matrix.max()))
short_labels = [MODEL_TO_FAMILY.get(v, v[:10]) for v in models_sorted]
ax.set_xticks(range(n_m))
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(n_m))
ax.set_yticklabels(short_labels, fontsize=7)
for i in range(n_m):
    for j in range(n_m):
        color = 'white' if d_matrix[i, j] > 1.5 else 'black'
        ax.text(j, i, f'{d_matrix[i, j]:.1f}', ha='center', va='center', fontsize=6, color=color)
plt.colorbar(im, ax=ax, label='max |d| across 9 dims', shrink=0.8)
plt.tight_layout()
save_fig(fig, 'fig4_cohen_d_heatmap')


# ============================
# Fig 5: HEXACO-H dot plot
# ============================
print("\n--- Fig 5: HEXACO-H Dot Plot ---")

model_h = s1.groupby("model_id")["hexaco_h"].agg(["mean", "std", "count"])
model_h = model_h.sort_values("mean", ascending=False)

fig, ax = plt.subplots(figsize=(5, 3.5))
y_pos = np.arange(len(model_h))
ax.errorbar(model_h["mean"], y_pos, xerr=model_h["std"],
            fmt='o', color=C_DISCRIMINATIVE, markersize=7, capsize=3,
            elinewidth=1, markeredgecolor='black', markeredgewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([MODEL_TO_FAMILY.get(v, v) for v in model_h.index], fontsize=8)
ax.set_xlabel('HEXACO-H Score (1-5)', fontsize=10)
ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
ax.text(1.02, len(model_h) - 0.3, 'Floor', fontsize=7, color='gray')
ax.set_xlim(0.8, model_h["mean"].max() + 0.5)
plt.tight_layout()
save_fig(fig, 'fig5_hexaco_h_models')


# ============================
# Fig 6: HEXACO-H vs Total Params scatter (all models, architecture colors)
# ============================
print("\n--- Fig 6: H vs Total Params ---")

# Combine all models for Figure 6 (same as PCA dataset)
combined = all_for_pca
model_h = combined.groupby("model")["hexaco_h"].agg(["mean", "std", "count"]).dropna()

params_list, h_list, arch_list = [], [], []
model_ids_with_params = []
for model_id, row in model_h.iterrows():
    params = TOTAL_PARAMS.get(model_id, 0)
    if params > 0:
        params_list.append(params)
        h_list.append(row["mean"])
        arch_list.append(ARCH_MAP.get(model_id, "Unknown"))
        model_ids_with_params.append(model_id)

# Also track undisclosed models (no x-axis position, shown separately)
undisclosed_h = []
undisclosed_families = []
for model_id, row in model_h.iterrows():
    if model_id not in TOTAL_PARAMS:
        undisclosed_h.append(row["mean"])
        vdata = combined[combined["model"] == model_id]
        mname = vdata["model"].iloc[0] if len(vdata) > 0 else ""
        undisclosed_families.append(MODEL_TO_FAMILY.get(mname, model_id))

from scipy import stats
if len(params_list) >= 3:
    r_linear, p_linear = stats.pearsonr(params_list, h_list)
    log_params = np.log10(params_list)
    r_log, p_log = stats.pearsonr(log_params, h_list)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Color by architecture type
    arch_colors = {"Dense": "#4C72B0", "MoE": "#DD8452", "Undisclosed": "#999999"}
    arch_markers = {"Dense": "o", "MoE": "s", "Undisclosed": "D"}
    for arch_type in ["Dense", "MoE"]:
        fidx = [j for j, a in enumerate(arch_list) if a == arch_type]
        if not fidx:
            continue
        ax.scatter([params_list[j] for j in fidx], [h_list[j] for j in fidx],
                   c=arch_colors[arch_type], s=60, marker=arch_markers[arch_type],
                   edgecolors='black', linewidths=0.5, zorder=3)

    # Regression line (log scale)
    slope, intercept, _, _, _ = stats.linregress(log_params, h_list)
    x_range = np.linspace(min(params_list), max(params_list), 50)
    ax.plot(x_range, slope * np.log10(x_range) + intercept, 'k--', linewidth=1, alpha=0.5)

    # Undisclosed models shown on right margin
    for i, (h_val, fam) in enumerate(zip(undisclosed_h, undisclosed_families)):
        ax.scatter([], [], c=arch_colors["Undisclosed"], s=60, marker=arch_markers["Undisclosed"],
                   edgecolors='black', linewidths=0.5)
        ax.annotate(fam, (max(params_list) * 1.15, h_val),
                    fontsize=6, va='center', color='#555555', alpha=0.8)

    # Legend
    legend_handles = [
        Patch(facecolor=arch_colors["Dense"], edgecolor='black', label=f'Dense (n={sum(1 for a in arch_list if a=="Dense")})'),
        Patch(facecolor=arch_colors["MoE"], edgecolor='black', label=f'MoE (n={sum(1 for a in arch_list if a=="MoE")})'),
    ]
    if undisclosed_h:
        legend_handles.append(Patch(facecolor=arch_colors["Undisclosed"], edgecolor='black',
                                    label=f'Undisclosed (n={len(undisclosed_h)})'))

    ax.set_xlabel('Total Parameters (Billions)', fontsize=10)
    ax.set_ylabel('HEXACO-H Score', fontsize=10)
    ax.set_xscale('log')
    ax.legend(handles=legend_handles, fontsize=7, loc='upper right', frameon=False)
    sig_label = f'p={p_log:.3f}' if p_log >= 0.001 else f'p<0.001'
    ax.text(0.03, 0.03, f'r = {r_log:.2f} (log scale, {sig_label})',
            transform=ax.transAxes, fontsize=7.5, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    save_fig(fig, 'fig6_hexaco_h_params')
    print(f"  r_linear={r_linear:.3f} (p={p_linear:.4f}), r_log={r_log:.3f} (p={p_log:.4f}), n={len(params_list)}+{len(undisclosed_h)} undisclosed")


# ============================
# Fig 7: Inter-Dimension Correlation Matrix
# ============================
print("\n--- Fig 7: Inter-Dim Correlation ---")

corr = s1.groupby("model")[DIMENSIONS].mean().corr()
short_dims = ["E", "A", "C", "N", "O", "H", "Col", "Int", "UA"]

fig, ax = plt.subplots(figsize=(4.5, 3.8))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax.set_xticks(range(len(short_dims)))
ax.set_xticklabels(short_dims, fontsize=8)
ax.set_yticks(range(len(short_dims)))
ax.set_yticklabels(short_dims, fontsize=8)
for i in range(N_DIMS):
    for j in range(N_DIMS):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
# no colorbar
plt.tight_layout()
save_fig(fig, 'fig7_inter_dim_corr')

print("\nAll figures regenerated!")
