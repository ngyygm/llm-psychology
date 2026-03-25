#!/usr/bin/env python3
"""Generate all publication-quality figures for EMNLP 2026 paper.

Usage: python3 figures/gen_all_figures.py
"""

import sys
sys.path.insert(0, '/home/linkco/exa/llm-psychology/figures')
from paper_plot_style import *

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# ============== Load Data ==============
print("Loading data...")
s1 = load_study1_data()
s2 = load_study2_data()
print(f"Study 1: {len(s1)} obs, {s1['model'].nunique()} models")
print(f"Study 2: {len(s2)} obs, {s2['model'].nunique()} models")

# Model-level means for Study 1
s1_model_means = s1.groupby("model")[DIMENSIONS].agg(["mean", "std", "count"])
s1_vendor_means = s1.groupby("vendor")[DIMENSIONS].agg(["mean", "std"])


# ============== Fig 1: Radar Chart — 3 Representative Vendors ==============
print("\n--- Fig 1: Radar Chart ---")

# Show all available vendors as small multiples
all_vendors = sorted(s1['vendor'].unique())
n_vendors = len(all_vendors)
print(f"  All vendors: {all_vendors} (n={n_vendors})")

# 5-column wide grid to fit two-column appendix page
ncols = 5
nrows = (n_vendors + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 1.8 * nrows),
                         subplot_kw=dict(polar=True))
axes = axes.flatten()

categories = [DIM_SHORT[d] for d in DIMENSIONS]
n_cats = len(categories)
angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
angles += angles[:1]

# Compute global min/max for consistent axis
global_mins, global_maxs = [], []
for vendor in all_vendors:
    vdata = s1[s1['vendor'] == vendor]
    if len(vdata) > 0:
        for d in DIMENSIONS:
            global_mins.append(vdata[d].mean())
            global_maxs.append(vdata[d].mean())

for idx, vendor in enumerate(all_vendors):
    ax = axes[idx]
    vdata = s1[s1['vendor'] == vendor]
    if len(vdata) == 0:
        ax.set_visible(False)
        continue

    means = [vdata[d].mean() for d in DIMENSIONS]
    means_plot = means + means[:1]
    color = C_VENDOR_COLORS.get(vendor, COLORS[idx % 10])

    ax.plot(angles, means_plot, 'o-', linewidth=1.5, color=color, markersize=3)
    ax.fill(angles, means_plot, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=5)
    ax.set_ylim(1, 5)
    ax.set_yticks([2, 3, 4])
    ax.set_yticklabels(['2', '3', '4'], fontsize=5, color='gray')
    ax.set_title(vendor, fontsize=7, pad=6)

# Hide unused subplots
for idx in range(n_vendors, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
save_fig(fig, 'fig1_radar_profiles')


# ============== Fig 2: Inter-Vendor SD Bar Chart ==============
print("\n--- Fig 2: Inter-Vendor SD Bar Chart ---")

model_means_all = s1.groupby("model")[DIMENSIONS].mean()
dim_sds = model_means_all.std()

# Sort dimensions by SD descending
sorted_dims = sorted(DIMENSIONS, key=lambda d: dim_sds[d], reverse=True)

# Classify dimensions (SD < 0.15 = alignment artifact)
bar_colors = [C_ALIGNMENT if dim_sds[d] < 0.15 else C_DISCRIMINATIVE for d in sorted_dims]

fig, ax = plt.subplots(figsize=(6, 3.5))
labels = [DIM_AXIS[d] for d in sorted_dims]
bars = ax.bar(labels, [dim_sds[d] for d in sorted_dims], color=bar_colors,
              edgecolor='black', linewidth=0.5)

# Add value labels
for bar, val in zip(bars, [dim_sds[d] for d in DIMENSIONS]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=FONT_SIZE - 2)

# Add threshold line
ax.axhline(y=0.15, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.text(len(labels) - 0.5, 0.16, 'Threshold', fontsize=FONT_SIZE - 3, color='gray', ha='right')

ax.set_ylabel('Inter-Vendor SD')
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=FONT_SIZE - 2)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=C_DISCRIMINATIVE, edgecolor='black', label='Discriminative'),
                   Patch(facecolor=C_ALIGNMENT, edgecolor='black', label='Alignment Artifact')]
ax.legend(handles=legend_elements, frameon=False, fontsize=FONT_SIZE - 2)
plt.tight_layout()
save_fig(fig, 'fig2_intervendor_sd')


# ============== Fig 3: HEXACO-H Dot Plot by Vendor ==============
print("\n--- Fig 3: HEXACO-H Dot Plot ---")

vendor_h = s1.groupby("vendor")["hexaco_h"].agg(["mean", "std", "count"])
vendor_h = vendor_h.sort_values("mean", ascending=False)

fig, ax = plt.subplots(figsize=(5.5, 4))
y_pos = np.arange(len(vendor_h))

# Color by discriminative vs alignment
h_sd = dim_sds.get("hexaco_h", 0)
dot_color = C_DISCRIMINATIVE if h_sd >= 0.15 else C_ALIGNMENT

ax.errorbar(vendor_h["mean"], y_pos, xerr=vendor_h["std"],
            fmt='o', color=dot_color, markersize=8, capsize=3,
            elinewidth=1, markeredgecolor='black', markeredgewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(vendor_h.index, fontsize=FONT_SIZE - 1)
ax.set_xlabel('HEXACO-H Score (1-5)')
ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
ax.text(1.02, len(vendor_h) - 0.3, 'Floor (1.0)', fontsize=FONT_SIZE - 3, color='gray')

# Add n per vendor
for i, (vendor, row) in enumerate(vendor_h.iterrows()):
    ax.text(vendor_h["mean"].max() + 0.1, i, f'n={int(row["count"])}',
            fontsize=FONT_SIZE - 3, va='center', color='gray')

ax.set_xlim(0.8, vendor_h["mean"].max() + 0.6)
plt.tight_layout()
save_fig(fig, 'fig3_hexaco_h_vendors')


# ============== Fig 4: H × Active Params Scatter ==============
print("\n--- Fig 4: H × Active Params Scatter ---")

MODEL_PARAMS = {
    "Qwen/Qwen3.5-397B-A17B": 17, "Qwen/Qwen3.5-4B": 4,
    "Qwen/Qwen3.5-27B": 27, "Qwen/Qwen3.5-35B-A3B": 3,
    "Qwen/Qwen3.5-122B-A10B": 10,
    "ollama:qwen3.5:9b": 9,
    "Pro/deepseek-ai/DeepSeek-V3.2": 37, "deepseek-ai/DeepSeek-V3": 37,
    "deepseek-ai/DeepSeek-V2.5": 21, "deepseek-ai/DeepSeek-R1": 37,
    "baidu/ERNIE-4.5-300B-A47B": 47,
    "tencent/Hunyuan-A13B-Instruct": 13,
    "ByteDance-Seed/Seed-OSS-36B-Instruct": 36,
    "internlm/internlm2_5-7b-chat": 7,
    "Pro/zai-org/GLM-5": 32,
    "Pro/moonshotai/Kimi-K2.5": 32,
}

# Combine study 1 and study 2 for more data points
all_chat = pd.concat([s1, s2])
model_h = all_chat.groupby("model")["hexaco_h"].agg(["mean", "std", "count"])
model_h = model_h.dropna()

params_list = []
h_list = []
vendor_list = []
for model_id, row in model_h.iterrows():
    params = MODEL_PARAMS.get(model_id, 0)
    if params > 0:
        params_list.append(params)
        h_list.append(row["mean"])
        # Get vendor
        vdata = all_chat[all_chat["model"] == model_id]
        vendor_list.append(vdata["vendor"].iloc[0] if len(vdata) > 0 else model_id.split("/")[0])

from scipy import stats
if len(params_list) >= 3:
    r_val, p_val = stats.pearsonr(params_list, h_list)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Color by vendor
    unique_vendors = list(set(vendor_list))
    for i, v in enumerate(unique_vendors):
        vidx = [j for j, vv in enumerate(vendor_list) if vv == v]
        ax.scatter([params_list[j] for j in vidx], [h_list[j] for j in vidx],
                   c=C_VENDOR_COLORS.get(v, COLORS[i % 10]), label=v, s=60,
                   edgecolors='black', linewidths=0.5, zorder=3)

    # Regression line
    slope, intercept, _, _, _ = stats.linregress(params_list, h_list)
    x_range = np.linspace(min(params_list) - 2, max(params_list) + 2, 50)
    ax.plot(x_range, slope * x_range + intercept, 'k--', linewidth=1, alpha=0.5)

    sig_label = f'p={p_val:.3f}' if p_val >= 0.001 else f'p<0.001'
    ax.text(0.05, 0.95, f'r = {r_val:.2f} ({sig_label})',
            transform=ax.transAxes, fontsize=FONT_SIZE - 1, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Active Parameters (Billions)')
    ax.set_ylabel('HEXACO-H Score')
    ax.legend(fontsize=FONT_SIZE - 3, loc='lower right', ncol=2, frameon=False)
    plt.tight_layout()
    save_fig(fig, 'fig4_hexaco_h_params')
    print(f"  r={r_val:.3f}, p={p_val:.4f}, n={len(params_list)}")
else:
    print(f"  SKIP: Only {len(params_list)} models with known params")


# ============== Fig 5: Pairwise Cohen's d Heatmap ==============
print("\n--- Fig 5: Pairwise Cohen's d Heatmap ---")

vendors_sorted = sorted(s1['vendor'].unique())
# Use study 1 only for cross-vendor comparison
n_v = len(vendors_sorted)

# Compute pairwise d for each dimension
d_matrix = np.zeros((n_v, n_v))  # max |d| across dimensions
d_by_dim = {}

for dim in DIMENSIONS:
    d_dim = np.zeros((n_v, n_v))
    for i in range(n_v):
        for j in range(i + 1, n_v):
            g1 = s1[s1['vendor'] == vendors_sorted[i]][dim].dropna().values
            g2 = s1[s1['vendor'] == vendors_sorted[j]][dim].dropna().values
            if len(g1) < 2 or len(g2) < 2:
                d_dim[i, j] = d_dim[j, i] = 0
                continue
            pooled_sd = np.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + (len(g2)-1)*np.var(g2, ddof=1))
                                / (len(g1) + len(g2) - 2))
            if pooled_sd == 0:
                d_dim[i, j] = d_dim[j, i] = 0
                continue
            d = abs(np.mean(g1) - np.mean(g2)) / pooled_sd
            d_dim[i, j] = d_dim[j, i] = d
    d_by_dim[dim] = d_dim
    d_matrix = np.maximum(d_matrix, d_dim)

# Plot heatmap of max |d|
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(d_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(3, d_matrix.max()))

# Labels
short_vendors = [v[:12] for v in vendors_sorted]
ax.set_xticks(range(n_v))
ax.set_xticklabels(short_vendors, rotation=45, ha='right', fontsize=FONT_SIZE - 2)
ax.set_yticks(range(n_v))
ax.set_yticklabels(short_vendors, fontsize=FONT_SIZE - 2)

# Annotate cells
for i in range(n_v):
    for j in range(n_v):
        color = 'white' if d_matrix[i, j] > 1.5 else 'black'
        ax.text(j, i, f'{d_matrix[i, j]:.1f}', ha='center', va='center',
                fontsize=FONT_SIZE - 3, color=color)

plt.colorbar(im, ax=ax, label='max |d| across 9 dimensions', shrink=0.8)
plt.tight_layout()
save_fig(fig, 'fig5_cohen_d_heatmap')


# ============== Fig 6: Inter-Dimension Correlation Matrix ==============
print("\n--- Fig 6: Inter-Dimension Correlation Matrix ---")

corr = s1.groupby("model")[DIMENSIONS].mean().corr()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
short_dims = ["E", "A", "C", "N", "O", "H", "Col", "Int", "UA"]

im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax.set_xticks(range(len(short_dims)))
ax.set_xticklabels(short_dims, fontsize=FONT_SIZE - 1)
ax.set_yticks(range(len(short_dims)))
ax.set_yticklabels(short_dims, fontsize=FONT_SIZE - 1)

# Annotate
for i in range(len(DIMENSIONS)):
    for j in range(len(DIMENSIONS)):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=FONT_SIZE - 2, color=color)

plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
plt.tight_layout()
save_fig(fig, 'fig6_inter_dim_corr')


# ============== Appendix Figures: Study 2 Trajectories ==============
print("\n--- Appendix Fig A1: Qwen Scale Trajectories ---")

# Identify Qwen models in Study 2
qwen_models = {
    'Qwen Dense': {
        'Qwen/Qwen3.5-4B': '4B',
        'ollama:qwen3.5:9b': '9B',
        'Qwen/Qwen3.5-27B': '27B',
    },
    'Qwen MoE': {
        'Qwen/Qwen3.5-35B-A3B': '35B\n(3B act.)',
        'Qwen/Qwen3.5-122B-A10B': '122B\n(10B act.)',
        'Qwen/Qwen3.5-397B-A17B': '397B\n(17B act.)',
    }
}

for series_name, models_map in qwen_models.items():
    fig, axes = plt.subplots(3, 3, figsize=(8, 7))
    axes = axes.flatten()

    available_models = [m for m in models_map.keys() if m in s2['model'].values]

    if len(available_models) < 2:
        print(f"  SKIP {series_name}: only {len(available_models)} models")
        continue

    for dim_idx, dim in enumerate(DIMENSIONS):
        ax = axes[dim_idx]
        for model_id in available_models:
            model_data = s2[s2['model'] == model_id]
            if len(model_data) == 0:
                continue
            # Get parameter count for x-axis
            params = MODEL_PARAMS.get(model_id, 0)
            label = models_map[model_id]
            mean_val = model_data[dim].mean()
            std_val = model_data[dim].std()
            ax.errorbar(params, mean_val, yerr=std_val, fmt='o',
                        markersize=6, capsize=2, elinewidth=1,
                        markeredgecolor='black', markeredgewidth=0.5,
                        label=label)

        ax.set_title(DIM_SHORT[dim], fontsize=FONT_SIZE - 1)
        ax.set_xlabel('Active Params (B)', fontsize=FONT_SIZE - 3)
        ax.set_ylabel('Score', fontsize=FONT_SIZE - 3)
        if dim_idx == 0:
            ax.legend(fontsize=FONT_SIZE - 4, frameon=False)

    fig.suptitle(f'Study 2: {series_name} Trajectories', fontsize=FONT_SIZE + 1, y=1.01)
    plt.tight_layout()
    safe_name = series_name.replace(' ', '_').lower()
    save_fig(fig, f'fig_a1_qwen_{safe_name}')


# DeepSeek evolution
print("\n--- Appendix Fig A2: DeepSeek Evolution ---")
deepseek_models = {
    'deepseek-ai/DeepSeek-V2.5': 'V2.5',
    'deepseek-ai/DeepSeek-V3': 'V3',
    'deepseek-ai/DeepSeek-V3.2': 'V3.2',
    'Pro/deepseek-ai/DeepSeek-V3.2': 'V3.2\n(Pro)',
    'deepseek-ai/DeepSeek-R1': 'R1',
}

all_chat = pd.concat([s1, s2])
all_models = set(all_chat['model'].unique())
available_ds = [m for m in deepseek_models.keys() if m in all_models]
# Deduplicate (prefer non-Pro version)
seen_versions = set()
deduped_ds = []
for m in available_ds:
    ver = deepseek_models[m].split('\n')[0]
    if ver not in seen_versions:
        seen_versions.add(ver)
        deduped_ds.append(m)
available_ds = deduped_ds

if len(available_ds) >= 2:
    ds_colors = [C_VENDOR_COLORS.get('DeepSeek', COLORS[1])]

    fig, axes = plt.subplots(3, 3, figsize=(8, 7))
    axes = axes.flatten()

    for dim_idx, dim in enumerate(DIMENSIONS):
        ax = axes[dim_idx]
        for i, model_id in enumerate(available_ds):
            model_data = all_chat[all_chat['model'] == model_id]
            if len(model_data) == 0:
                continue
            label = deepseek_models[model_id]
            mean_val = model_data[dim].mean()
            std_val = model_data[dim].std()
            ax.errorbar(i, mean_val, yerr=std_val, fmt='o',
                        markersize=6, capsize=2, elinewidth=1,
                        markeredgecolor='black', markeredgewidth=0.5, color=ds_colors[0])

        ax.set_xticks(range(len(available_ds)))
        ax.set_xticklabels([deepseek_models[m] for m in available_ds],
                           fontsize=FONT_SIZE - 2, rotation=45)
        ax.set_title(DIM_SHORT[dim], fontsize=FONT_SIZE - 1)
        ax.set_ylabel('Score', fontsize=FONT_SIZE - 3)

    fig.suptitle('Study 2: DeepSeek Evolution', fontsize=FONT_SIZE + 1, y=1.01)
    plt.tight_layout()
    save_fig(fig, 'fig_a2_deepseek_evolution')
else:
    print(f"  SKIP: only {len(available_ds)} DeepSeek models")


# Zhipu evolution
print("\n--- Appendix Fig A3: Zhipu Evolution ---")
zhipu_models = {
    'THUDM/GLM-4-9B-0414': 'GLM-4\n9B',
    'THUDM/GLM-4-32B-0414': 'GLM-4\n32B',
    'zai-org/GLM-4.5-Air': 'GLM-4.5\nAir',
    'zai-org/GLM-4.6': 'GLM-4.6',
    'Pro/zai-org/GLM-5': 'GLM-5',
    'THUDM/GLM-Z1-32B-0414': 'GLM-Z1\n32B',
}

all_models = set(all_chat['model'].unique())
available_zp = [m for m in zhipu_models if m in all_models]

if len(available_zp) >= 2:
    fig, axes = plt.subplots(3, 3, figsize=(8, 7))
    axes = axes.flatten()
    zp_color = C_VENDOR_COLORS.get('Zhipu', COLORS[2])

    for dim_idx, dim in enumerate(DIMENSIONS):
        ax = axes[dim_idx]
        for i, model_id in enumerate(available_zp):
            model_data = all_chat[all_chat['model'] == model_id]
            if len(model_data) == 0:
                continue
            mean_val = model_data[dim].mean()
            std_val = model_data[dim].std()
            ax.errorbar(i, mean_val, yerr=std_val, fmt='o',
                        markersize=6, capsize=2, elinewidth=1,
                        markeredgecolor='black', markeredgewidth=0.5, color=zp_color)

        ax.set_xticks(range(len(available_zp)))
        ax.set_xticklabels([zhipu_models[m].replace('\n', ' ') for m in available_zp],
                           fontsize=FONT_SIZE - 2, rotation=45)
        ax.set_title(DIM_SHORT[dim], fontsize=FONT_SIZE - 1)
        ax.set_ylabel('Score', fontsize=FONT_SIZE - 3)

    fig.suptitle('Study 2: Zhipu Evolution', fontsize=FONT_SIZE + 1, y=1.01)
    plt.tight_layout()
    save_fig(fig, 'fig_a3_zhipu_evolution')
else:
    print(f"  SKIP: only {len(available_zp)} Zhipu models available ({available_zp})")


print("\n\nAll figures generated successfully!")
