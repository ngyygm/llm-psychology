#!/usr/bin/env python3
"""
PCA Visualization of LLM Response Styles
Reduces 9 personality dimensions to 2D and plots all models.

Usage:
  python3 create_pca_figure.py                           # Auto-load latest merged data
  python3 create_pca_figure.py --input results/vendor_exp/final_merged_*.json
"""

import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ============== DIMENSIONS ==============

DIMENSIONS = [
    "bfi.extraversion",
    "bfi.agreeableness",
    "bfi.conscientiousness",
    "bfi.neuroticism",
    "bfi.openness",
    "hexaco_h",
    "collectivism",
    "intuition",
    "uncertainty_avoidance",
]

DIM_LABELS = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "HEXACO-H",
    "Collectivism",
    "Intuition",
    "UA",
]

# Vendor colors (13 vendors + gray for others)
VENDOR_COLORS = {
    "Qwen":      "#E53935",  # red
    "DeepSeek":  "#1E88E5",  # blue
    "Zhipu":     "#7CB342",  # green
    "Moonshot":  "#FB8C00",  # orange
    "Baidu":     "#8E24AA",  # purple
    "Tencent":   "#00ACC1",  # cyan
    "ByteDance": "#FFB300",  # amber
    "InternLM":  "#5C6BC0",  # indigo
    "inclusionAI": "#26A69A", # teal
    "StepFun":   "#EC407A",  # pink
    "Huawei":    "#78909C",  # blue-grey
    "Kwaipilot": "#8D6E63",  # brown
    "MiniMax":   "#3949AB",  # deep-purple
}

# Study marker shapes
STUDY_MARKERS = {
    "1": "o",       # Study 1: circle
    "2": "s",       # Study 2: square
    "ablation": "D", # Thinking ablation: diamond
}

# Model short names for plot labels
MODEL_SHORT = {
    "Qwen/Qwen3.5-397B-A17B": "Qwen3.5-397B",
    "Qwen/Qwen3.5-122B-A10B": "Qwen3.5-122B",
    "Qwen/Qwen3.5-35B-A3B": "Qwen3.5-35B",
    "Qwen/Qwen3.5-27B": "Qwen3.5-27B",
    "Qwen/Qwen3.5-4B": "Qwen3.5-4B",
    "ollama:qwen3.5:9b": "Qwen3.5-9B",
    "Pro/deepseek-ai/DeepSeek-V3.2": "DS-V3.2-Pro",
    "deepseek-ai/DeepSeek-V3.2": "DS-V3.2",
    "deepseek-ai/DeepSeek-V3": "DS-V3",
    "deepseek-ai/DeepSeek-V2.5": "DS-V2.5",
    "deepseek-ai/DeepSeek-R1": "DS-R1",
    "Pro/zai-org/GLM-5": "GLM-5-Pro",
    "zai-org/GLM-4.6": "GLM-4.6",
    "zai-org/GLM-4.5-Air": "GLM-4.5",
    "THUDM/GLM-4-9B-0414": "GLM-4-9B",
    "THUDM/GLM-4-32B-0414": "GLM-4-32B",
    "THUDM/GLM-Z1-32B-0414": "GLM-Z1-32B",
    "Pro/moonshotai/Kimi-K2.5": "Kimi-K2.5",
    "moonshotai/Kimi-K2-Instruct-0905": "Kimi-K2",
    "baidu/ERNIE-4.5-300B-A47B": "ERNIE-4.5",
    "tencent/Hunyuan-A13B-Instruct": "Hunyuan-13B",
    "ByteDance-Seed/Seed-OSS-36B-Instruct": "Seed-36B",
    "internlm/internlm2_5-7b-chat": "InternLM-7B",
    "inclusionAI/Ring-flash-2.0": "Ring-Flash",
    "stepfun-ai/Step-3.5-Flash": "Step-3.5",
    "ascend-tribe/pangu-pro-moe": "Pangu-MoE",
    "Kwaipilot/KAT-Dev": "KAT-Dev",
    "Pro/MiniMaxAI/MiniMax-M2.5": "MiniMax-M2.5",
}


def short_name(model_id: str) -> str:
    return MODEL_SHORT.get(model_id, model_id.split("/")[-1])


def load_data(input_pattern: str) -> list:
    """Load all result JSON files."""
    all_results = []
    for f in sorted(glob.glob(input_pattern)):
        if "checkpoint" in Path(f).name:
            continue
        with open(f) as fh:
            all_results.extend(json.load(fh))
    return all_results


def compute_model_means(results: list) -> list:
    """Average across seeds for each (model, thinking_mode) combination."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        mid = r.get("model_id", r.get("model", ""))
        tm = r.get("thinking_mode", "chat")
        groups[(mid, tm)].append(r)

    model_data = []
    for (mid, tm), obs in groups.items():
        vendor = obs[0].get("vendor", mid.split("/")[0])
        study = obs[0].get("study", 0)
        subgroup = obs[0].get("subgroup", "")
        means = {}
        for dim in DIMENSIONS:
            vals = [r[dim] for r in obs if dim in r]
            means[dim] = np.mean(vals) if vals else np.nan
        model_data.append({
            "model_id": mid,
            "short_name": short_name(mid),
            "vendor": vendor,
            "study": study,
            "thinking_mode": tm,
            "subgroup": subgroup,
            "n_obs": len(obs),
            **means,
        })
    return model_data


def plot_pca(model_data: list, output_path: str = "results/vendor_exp/pca_response_styles.pdf"):
    """Create PCA scatter plot of all models."""
    # Build matrix: rows=models, cols=dimensions
    valid = [m for m in model_data if not any(np.isnan(m[d]) for d in DIMENSIONS)]
    if not valid:
        print("No valid data to plot!")
        return

    X = np.array([[m[d] for d in DIMENSIONS] for m in valid])

    # Standardize and PCA
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Group by vendor
    vendors = sorted(set(m["vendor"] for m in valid))

    for vendor in vendors:
        vendor_models = [(m, i) for i, m in enumerate(valid) if m["vendor"] == vendor]
        color = VENDOR_COLORS.get(vendor, "#999999")

        for m, i in vendor_models:
            # Marker by study/type
            if m["thinking_mode"] == "thinking":
                marker = "D"
                alpha = 0.6
                size = 60
            elif m["study"] == 1:
                marker = "o"
                alpha = 0.85
                size = 80
            else:
                marker = "s"
                alpha = 0.7
                size = 60

            x, y = X_pca[i]
            ax.scatter(x, y, c=color, marker=marker, s=size, alpha=alpha,
                       edgecolors='black', linewidths=0.5, zorder=5, label=vendor if vendor_models.index((m, i)) == 0 else None)

            # Label
            label = m["short_name"]
            # Thinking ablation: append tag
            if m["thinking_mode"] == "thinking":
                label += " [T]"

            ax.annotate(label, (x, y), fontsize=5.5, ha='left', va='bottom',
                        xytext=(4, 3), textcoords='offset points',
                        alpha=0.8, zorder=6)

    # Axis labels
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=13)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=13)
    ax.set_title("PCA of LLM Response Styles Across 9 Personality Dimensions",
                 fontsize=14, fontweight='bold', pad=15)

    # Legend: vendors only (one per vendor)
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate labels while preserving order
    seen = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    ax.legend(unique_h, unique_l, loc='upper right', fontsize=8,
              framealpha=0.9, edgecolor='gray', ncol=2)

    # Marker legend (manual)
    from matplotlib.lines import Line2D
    custom_markers = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=8, label='Study 1 (cross-vendor)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=8, label='Study 2 (evolution)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=8, label='Thinking ablation [T]'),
    ]
    ax.legend(handles=custom_markers, loc='lower right', fontsize=8,
              framealpha=0.9, edgecolor='gray')

    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved to {output_path}")
    print(f"Saved to {output_path.replace('.pdf', '.png')}")

    # Print PCA summary
    print(f"\nPCA Summary ({len(valid)} models):")
    print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}% variance")
    print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}% variance")
    print(f"  Total: {sum(pca.explained_variance_ratio_[:2])*100:.1f}%")

    # Print loadings
    print(f"\nDimension loadings:")
    for dim, label in zip(DIMENSIONS, DIM_LABELS):
        l1, l2 = pca.components_[0][DIMENSIONS.index(dim)], pca.components_[1][DIMENSIONS.index(dim)]
        print(f"  {label:20s} PC1={l1:+.3f}  PC2={l2:+.3f}")

    return pca, X_pca


def plot_pca_by_vendor(model_data: list, output_path: str = "results/vendor_exp/pca_by_vendor.pdf"):
    """Separate PCA plots per vendor for Study 2 evolution trajectories."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    fig.patch.set_facecolor('white')

    # Vendors with Study 2 data
    study2_vendors = sorted(set(m["vendor"] for m in model_data if m["study"] == 2))
    top_vendors = [v for v in ["Qwen", "DeepSeek", "Zhipu"] if v in study2_vendors]

    for idx, vendor in enumerate(top_vendors[:4]):
        ax = axes[idx // 2][idx % 2]
        ax.set_facecolor('white')

        vendor_data = [m for m in model_data if m["vendor"] == vendor]
        if not vendor_data:
            ax.set_visible(False)
            continue

        X = np.array([[m[d] for d in DIMENSIONS] for m in vendor_data])
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)

        color = VENDOR_COLORS.get(vendor, "#999999")

        # Plot Study 1 model as reference (large circle)
        # Plot Study 2 models with size/order indicating scale
        for i, m in enumerate(vendor_data):
            x, y = X_pca[i]
            label = m["short_name"]
            sg = m.get("subgroup", "")
            tm = m.get("thinking_mode", "chat")

            if m["study"] == 1:
                marker, size, alpha = "o", 120, 1.0
                label = label + " (S1)"
            elif tm == "reasoning":
                marker, size, alpha = "D", 80, 0.7
                label = label + " [R]"
            elif tm == "thinking":
                marker, size, alpha = "^", 70, 0.7
                label = label + " [T]"
            else:
                marker, size, alpha = "s", 70, 0.8

            ax.scatter(x, y, c=color, marker=marker, s=size, alpha=alpha,
                       edgecolors='black', linewidths=0.5, zorder=5)
            ax.annotate(label, (x, y), fontsize=7, ha='left', va='bottom',
                        xytext=(4, 3), textcoords='offset points', alpha=0.9)

        # Draw arrows between consecutive models in same subgroup
        subgroups = sorted(set(m.get("subgroup", "") for m in vendor_data if m["study"] == 2 and m.get("thinking_mode", "chat") != "reasoning"))
        for sg in subgroups:
            sg_data = [m for m in vendor_data
                       if m.get("subgroup") == sg and m.get("thinking_mode", "chat") != "reasoning"]
            if len(sg_data) > 1:
                for j in range(len(sg_data) - 1):
                    m1 = sg_data[j]
                    m2 = sg_data[j + 1]
                    idx1 = vendor_data.index(m1)
                    idx2 = vendor_data.index(m2)
                    ax.annotate("", xy=(X_pca[idx2, 0], X_pca[idx2, 1]),
                                xytext=(X_pca[idx1, 0], X_pca[idx1, 1]),
                                arrowprops=dict(arrowstyle='->', color=color, alpha=0.5, lw=1.5))

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
        ax.set_title(f"{vendor} — Response Style Trajectory", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved vendor trajectories to {output_path}")


def plot_pca_biplot(model_data: list, output_path: str = "results/vendor_exp/pca_biplot.pdf"):
    """PCA biplot showing both model positions and dimension loadings."""
    valid = [m for m in model_data if not any(np.isnan(m[d]) for d in DIMENSIONS)]
    X = np.array([[m[d] for d in DIMENSIONS] for m in valid])

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    fig, ax = plt.subplots(1, 1, figsize=(13, 9), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot models colored by vendor
    for vendor in sorted(set(m["vendor"] for m in valid)):
        vendor_idx = [i for i, m in enumerate(valid) if m["vendor"] == vendor]
        color = VENDOR_COLORS.get(vendor, "#999999")
        ax.scatter(X_pca[vendor_idx, 0], X_pca[vendor_idx, 1],
                   c=color, s=60, alpha=0.8, edgecolors='black', linewidths=0.3,
                   label=vendor)

    # Plot dimension loading vectors
    scale = 2.0  # scale factor for visibility
    for dim, label in zip(DIMENSIONS, DIM_LABELS):
        j = DIMENSIONS.index(dim)
        ax.annotate("", xy=(pca.components_[0][j] * scale, pca.components_[1][j] * scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2, alpha=0.7))
        ax.text(pca.components_[0][j] * scale * 1.15, pca.components_[1][j] * scale * 1.15,
                label, fontsize=8, ha='center', va='center', color='#333333', fontweight='bold')

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=13)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=13)
    ax.set_title("PCA Biplot — Response Styles & Dimension Loadings", fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9, edgecolor='gray', ncol=2)
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved biplot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA visualization of LLM response styles")
    parser.add_argument("--input", type=str, default="results/vendor_exp/*.json",
                        help="Glob pattern for result JSON files")
    args = parser.parse_args()

    results = load_data(args.input)
    print(f"Loaded {len(results)} observations from {args.input}")

    model_data = compute_model_means(results)
    print(f"Computed means for {len(model_data)} (model, thinking_mode) groups")
    for m in model_data:
        print(f"  {m['short_name']:25s} vendor={m['vendor']:12s} study={m['study']} "
              f"mode={m['thinking_mode']:8s} n={m['n_obs']}")

    if model_data:
        plot_pca(model_data)

        valid_full = [m for m in model_data if not any(np.isnan(m[d]) for d in DIMENSIONS)]
        if len(valid_full) >= 3:
            plot_pca_biplot(model_data)

        # Only plot vendor trajectories if we have Study 2 data with dimensions
        study2_valid = [m for m in valid_full if m["study"] == 2]
        if len(study2_valid) > 1:
            plot_pca_by_vendor(model_data)
