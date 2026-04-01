#!/usr/bin/env python3
"""Generate publication-quality figures for EMNLP 2026 paper.

Figures:
  1. Radar profiles (Study 1, selected models, 8 primary dimensions)
  2. Cohen's d heatmap (Study 1 pairwise effects, 8 dimensions)
  3. Inter-dimension correlation matrix (8 primary dimensions)
  4. Study 2 within-model trajectories (DeepSeek, Qwen, Zhipu)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "results.json"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,  # TrueType fonts for PDF
    "ps.fonttype": 42,
})

# 8 primary dimensions (HEXACO-H excluded from primary analysis)
DIM_KEYS = [
    "bfi.extraversion", "bfi.agreeableness", "bfi.conscientiousness",
    "bfi.neuroticism", "bfi.openness", "collectivism", "intuition",
    "uncertainty_avoidance",
]
DIM_LABELS = [
    "Extra.", "Agreeab.", "Conscien.", "Neurotic.", "Openness",
    "Collectiv.", "Intuition", "UA",
]
DIM_SHORT = ["E", "A", "C", "N", "O", "Col", "Int", "UA"]

# All 11 models — colors chosen for distinctness and accessibility
MODEL_COLORS = {
    "Baidu":      "#1f77b4",
    "ByteDance":  "#ff7f0e",
    "DeepSeek":   "#2ca02c",
    "Huawei":     "#d62728",
    "InternLM":   "#9467bd",
    "Kwaipilot":  "#8c564b",
    "MiniMax":    "#e377c2",
    "Moonshot":   "#7f7f7f",
    "StepFun":    "#bcbd22",
    "Tencent":    "#17becf",
    "inclusionAI":"#aec7e8",
}

STUDY2_COLORS = {
    "DeepSeek": {"color": "#2ca02c", "marker": "o"},
    "Qwen":     {"color": "#ff7f0e", "marker": "s"},
    "Zhipu":    {"color": "#1f77b4", "marker": "^"},
}

# Models to highlight in radar (most distinct profiles)
RADAR_HIGHLIGHT = ["MiniMax", "InternLM", "DeepSeek", "Baidu", "ByteDance",
                    "Moonshot", "Huawei", "StepFun", "Tencent", "Kwaipilot",
                    "inclusionAI"]


# ── Load & prepare data ───────────────────────────────────────────────────
def load_data():
    with open(DATA_FILE) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["subgroup"] = df.apply(_derive_subgroup, axis=1)
    return df


def _derive_subgroup(row):
    if row["study"] != 2:
        return ""
    mid = row["model_id"]
    if "DeepSeek-V2" in mid: return "DeepSeek"
    if "DeepSeek-V3" in mid and "R1" not in mid: return "DeepSeek"
    if "DeepSeek-R1" in mid: return "DeepSeek"
    if "Qwen" in mid or "qwen" in mid.lower(): return "Qwen"
    if "GLM" in mid or "zai" in mid.lower(): return "Zhipu"
    return ""


def get_study1_model_means(df):
    s1 = df[df["study"] == 1].copy()
    model_means = s1.groupby("model_id")[DIM_KEYS].mean()
    model_means = model_means.loc[model_means.index.isin(MODEL_COLORS.keys())]
    model_means = model_means.sort_index()
    return model_means


def compute_cohens_d(df):
    s1 = df[df["study"] == 1]
    models = sorted(s1["model"].unique())
    n_models = len(models)
    n_dims = len(DIM_KEYS)

    d_matrix = np.full((n_dims, n_models, n_models), np.nan)
    for di, dim in enumerate(DIM_KEYS):
        for i, v1 in enumerate(models):
            for j, v2 in enumerate(models):
                if i >= j:
                    continue
                g1 = s1[s1["model"] == v1][dim].dropna()
                g2 = s1[s1["model"] == v2][dim].dropna()
                if len(g1) < 2 or len(g2) < 2:
                    continue
                pooled_sd = np.sqrt(
                    ((len(g1) - 1) * g1.std()**2 + (len(g2) - 1) * g2.std()**2)
                    / (len(g1) + len(g2) - 2)
                )
                if pooled_sd == 0:
                    continue
                d = (g1.mean() - g2.mean()) / pooled_sd
                d_matrix[di, i, j] = d
                d_matrix[di, j, i] = -d

    return d_matrix, models


def compute_dim_correlations(df):
    model_means = df.groupby("model_id")[DIM_KEYS].mean()
    corr = model_means.corr()
    return corr


def get_study2_trajectories(df):
    s2 = df[(df["study"] == 2) & (df["thinking_mode"] == "chat")].copy()

    family_models = {
        "DeepSeek": [
            "deepseek-ai/DeepSeek-V2.5",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-V3.2",
            "deepseek-ai/DeepSeek-R1",
        ],
        "Qwen": [
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-27B",
            "Qwen/Qwen3.5-397B-A17B",
        ],
        "Zhipu": [
            "THUDM/GLM-4-9B-0414",
            "THUDM/GLM-4-32B-0414",
            "zai-org/GLM-4.5-Air",
            "zai-org/GLM-4.6",
            "Pro/zai-org/GLM-5",
        ],
    }

    family_model_labels = {
        "DeepSeek": ["V2.5", "V3", "V3.2", "R1"],
        "Qwen": ["4B", "27B", "397B"],
        "Zhipu": ["9B", "32B", "4.5", "4.6", "5"],
    }

    trajectories = {}
    for model, models in family_models.items():
        means = []
        for mid in models:
            recs = s2[s2["model_id"] == mid]
            if len(recs) == 0:
                matches = s2[s2["model_id"].str.contains(mid.split("/")[-1], na=False)]
                if len(matches) > 0:
                    recs = matches
            if len(recs) > 0:
                means.append(recs[DIM_KEYS].mean())
        if means:
            trajectories[model] = {
                "means": pd.DataFrame(means),
                "labels": family_model_labels[model][:len(means)],
            }

    return trajectories


# ── Figure 1: Radar Profiles ──────────────────────────────────────────────
def fig1_radar_profiles(df):
    """Radar chart showing model profiles across 8 dimensions."""
    model_means = get_study1_model_means(df)
    n_dims = len(DIM_KEYS)

    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    # Select 6 most distinctive models for clarity
    # Based on ANOVA effect sizes, pick those with extreme profiles
    highlight_models = ["MiniMax", "InternLM", "DeepSeek", "Baidu",
                         "ByteDance", "Huawei"]
    other_models = [v for v in model_means.index if v not in highlight_models]

    fig, ax = plt.subplots(figsize=(3.2, 3.2), subplot_kw=dict(polar=True))

    # Plot "other" models first (lighter, background)
    for model in other_models:
        values = model_means.loc[model, DIM_KEYS].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=0.5, color="#cccccc", alpha=0.4, zorder=1)
        ax.fill(angles, values, alpha=0.02, color="#cccccc")

    # Plot highlighted models
    for model in highlight_models:
        if model not in model_means.index:
            continue
        values = model_means.loc[model, DIM_KEYS].values.tolist()
        values += values[:1]
        color = MODEL_COLORS[model]
        ax.plot(angles, values, linewidth=1.0, color=color, alpha=0.85,
                label=model, zorder=2)
        ax.fill(angles, values, alpha=0.06, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_SHORT, fontsize=8, fontweight="bold")
    ax.set_ylim(1.8, 3.6)
    ax.set_yticks([2.0, 2.5, 3.0, 3.5])
    ax.set_yticklabels(["2.0", "2.5", "3.0", "3.5"], fontsize=6, color="#888888")
    ax.spines["polar"].set_linewidth(0.5)
    ax.grid(linewidth=0.3, alpha=0.5)

    # Legend outside, compact
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=6,
              frameon=True, framealpha=0.9, edgecolor="#cccccc",
              handletextpad=0.3, borderpad=0.3)

    plt.tight_layout()
    out = FIG_DIR / "fig1_radar_profiles.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Figure 2: Cohen's d Heatmap ───────────────────────────────────────────
def fig2_cohen_d_heatmap(df):
    """Heatmap of max pairwise Cohen's d across model pairs."""
    d_matrix, models = compute_cohens_d(df)

    n_v = len(models)
    max_d = np.zeros((n_v, n_v))
    for i in range(n_v):
        for j in range(n_v):
            if i != j:
                max_d[i, j] = np.nanmax(np.abs(d_matrix[:, i, j]))

    v_labels = [v if v in MODEL_COLORS else v for v in models]

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    mask = np.eye(n_v, dtype=bool)

    # Use sequential colormap with proper range
    cmap = plt.colormaps["YlOrRd"]
    norm = Normalize(vmin=0, vmax=7.0)

    hm = sns.heatmap(max_d, mask=mask, annot=True, fmt=".1f",
                     xticklabels=v_labels, yticklabels=v_labels,
                     cmap=cmap, norm=norm, ax=ax,
                     linewidths=0.5, linecolor="white",
                     cbar_kws={"label": "max $|d|$", "shrink": 0.8, "aspect": 20},
                     annot_kws={"fontsize": 6.5, "fontweight": "bold"})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title("Maximum Pairwise Cohen's $d$ Across Dimensions", fontsize=9, pad=10)

    plt.tight_layout()
    out = FIG_DIR / "fig2_cohen_d_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Figure 3: Inter-dimension Correlation ─────────────────────────────────
def fig3_inter_dim_corr(df):
    """Correlation matrix of 8 primary dimensions (model-level)."""
    corr = compute_dim_correlations(df)

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    hm = sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                     xticklabels=DIM_LABELS, yticklabels=DIM_LABELS,
                     cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
                     linewidths=0.5, linecolor="white",
                     cbar_kws={"shrink": 0.8, "label": "$r$", "aspect": 20},
                     annot_kws={"fontsize": 7.5})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title("Inter-Dimension Correlations", fontsize=9, pad=8)

    plt.tight_layout()
    out = FIG_DIR / "fig3_inter_dim_corr.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Figure 4: Study 2 Trajectories ────────────────────────────────────────
def fig4_study2_trajectories(df):
    """Line plots showing within-model dimension trajectories."""
    trajectories = get_study2_trajectories(df)

    traj_dims = [
        ("bfi.neuroticism", "Neuroticism", "N"),
        ("bfi.conscientiousness", "Conscientiousness", "C"),
        ("intuition", "Intuition", "Int"),
        ("bfi.openness", "Openness", "O"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=False)

    for ai, (dim_key, dim_name, dim_label) in enumerate(traj_dims):
        ax = axes[ai]
        for model, traj in trajectories.items():
            meta = STUDY2_COLORS[model]
            means = traj["means"][dim_key].values
            xs = range(len(means))
            ax.plot(xs, means, marker=meta["marker"], color=meta["color"],
                    linewidth=1.3, markersize=5, label=model,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(traj["labels"], fontsize=6)

        ax.set_title(dim_label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean" if ai == 0 else "", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(1.5, 3.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.3)

        if ai == 3:
            ax.legend(fontsize=6, frameon=True, framealpha=0.9,
                      edgecolor="#cccccc", loc="best")

    plt.tight_layout()
    out = FIG_DIR / "fig4_study2_trajectories.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── LaTeX includes ────────────────────────────────────────────────────────
def write_latex_includes(figures):
    lines = []
    lines.append("% Auto-generated LaTeX figure includes")
    lines.append("")
    lines.append("% Figure 1: Radar profiles")
    lines.append(r"\begin{figure}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \includegraphics[width=0.45\textwidth]{figures/fig1_radar_profiles.pdf}")
    lines.append(r"  \caption{Response style profiles across 8 psychometric dimensions for selected models (gray: remaining models). Scores on a 1--5 Likert scale. Models show distinct profiles, with MiniMax and InternLM exhibiting the most divergent patterns.}")
    lines.append(r"  \label{fig:radar}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append("% Figure 2: Cohen's d heatmap")
    lines.append(r"\begin{figure}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \includegraphics[width=0.50\textwidth]{figures/fig2_cohen_d_heatmap.pdf}")
    lines.append(r"  \caption{Maximum pairwise Cohen's $d$ across all 8 dimensions for each model pair. All pairs exhibit at least one large effect ($d \geq 0.8$); the largest is Baidu vs.\ MiniMax ($d=6.2$ on Intuition).}")
    lines.append(r"  \label{fig:cohens_d}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append("% Figure 3: Inter-dimension correlation")
    lines.append(r"\begin{figure}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \includegraphics[width=0.40\textwidth]{figures/fig3_inter_dim_corr.pdf}")
    lines.append(r"  \caption{Inter-dimension correlations at the model level ($n=33$). The strong C--N correlation ($r=0.83$) reflects acquiescence bias rather than a substantive construct relationship (PC1 explains 52\% of variance).}")
    lines.append(r"  \label{fig:corr}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append("% Figure 4: Study 2 trajectories")
    lines.append(r"\begin{figure*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \includegraphics[width=0.85\textwidth]{figures/fig4_study2_trajectories.pdf}")
    lines.append(r"  \caption{Within-model response style trajectories across model versions for DeepSeek (V2.5$\rightarrow$R1), Qwen (4B$\rightarrow$397B), and Zhipu (GLM-4$\rightarrow$GLM-5). Each panel shows a different psychometric dimension. Models exhibit distinct evolutionary patterns, suggesting alignment training shapes response styles over time.}")
    lines.append(r"  \label{fig:trajectories}")
    lines.append(r"\end{figure*}")
    lines.append("")

    out = FIG_DIR / "latex_includes.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} records loaded")

    print("\nGenerating figures...")
    figs = []
    figs.append(fig1_radar_profiles(df))
    figs.append(fig2_cohen_d_heatmap(df))
    figs.append(fig3_inter_dim_corr(df))
    figs.append(fig4_study2_trajectories(df))

    print("\nWriting LaTeX includes...")
    write_latex_includes(figs)

    print("\nDone! All figures saved to paper/figures/")
