#!/usr/bin/env python3
"""
Within-Sample Consistency Analysis (5-repetition)
=================================================
Measures how consistent LLMs are when answering the same personality
questionnaire item multiple times under identical conditions (same model,
same persona, same item, temperature=0.7).

Metrics computed for each (model, persona, item) triple:
  - sd:               standard deviation across repetitions
  - range:            max - min across repetitions
  - exact_agreement:  1 if all repetitions give identical answer, 0 otherwise
  - mode_agreement:   fraction of repetitions equal to the modal answer
  - mean_score:       mean scored_value across repetitions

Analysis levels (macro → micro):
  1. Overall portrait
  2. By model
  3. By persona (Default vs 16 MBTI)
  4. By scale / domain
  5. By individual item
  6. Interaction: model × persona
  7. Interaction: model × domain
  8. Interaction: persona × domain
  9. Interaction: model × persona × domain (anomaly detection)

Figures (EMNLP style):
  - Fig WSC-1: Overall consistency portrait (violin + bar)
  - Fig WSC-2: Model-level consistency (bee swarm)
  - Fig WSC-3: Persona-level consistency (bee swarm)
  - Fig WSC-4: Domain-level consistency (grouped violin)
  - Fig WSC-5: Item-level: top/bottom unstable items (horizontal bar)
  - Fig WSC-6: Model × persona dendrogram + line profiles
  - Fig WSC-7: Model × domain dendrogram + line profiles
  - Fig WSC-8: Persona × domain dendrogram + line profiles
  - Fig WSC-9: Anomaly detection: worst (model, persona, domain) cells
"""
from __future__ import annotations

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis_output")
FIG_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# EMNLP-compatible style
PALETTE = sns.color_palette("husl", 20)
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

# ============================================================================
#                          DATA LOADING
# ============================================================================

def load_all_results() -> dict:
    results = {}
    for fpath in sorted(RESULTS_DIR.glob("exp_mbti_*.json")):
        model_name = fpath.stem.replace("exp_mbti_", "")
        with open(fpath) as f:
            results[model_name] = json.load(f)
    return results


def build_consistency_frame(all_results: dict) -> pd.DataFrame:
    """Build a DataFrame with one row per (model, persona, item) triple.
    Each row contains consistency metrics across repetitions."""
    rows = []
    for model_name, model_data in all_results.items():
        for persona, pdata in model_data["results_by_persona"].items():
            for rec in pdata["responses"]:
                samples = rec.get("samples", [])
                if not samples:
                    continue

                scored_vals = [s["scored_value"] for s in samples
                               if s.get("scored_value") is not None]
                parsed_vals = [s["parsed_value"] for s in samples
                               if s.get("parsed_value") is not None]

                if not scored_vals:
                    continue

                n = len(scored_vals)
                arr = np.array(scored_vals, dtype=float)
                sd = float(np.std(arr)) if n > 1 else 0.0
                rng = float(arr.max() - arr.min())
                mean_sc = float(np.mean(arr))

                # Exact agreement: all identical
                exact = 1.0 if n <= 1 or np.all(arr == arr[0]) else 0.0

                # Mode agreement: fraction equal to modal value
                counter = Counter(scored_vals)
                mode_count = counter.most_common(1)[0][1]
                mode_agree = mode_count / n if n > 0 else 1.0

                rows.append({
                    "model": model_name,
                    "persona": persona,
                    "item_id": rec["item_id"],
                    "scale": rec["scale"],
                    "domain": rec["domain"],
                    "facet": rec.get("facet"),
                    "keyed": rec["keyed"],
                    "response_format": rec["response_format"],
                    "item_text": rec.get("item_text", ""),
                    "n_samples": n,
                    "sd": sd,
                    "range": rng,
                    "exact_agreement": exact,
                    "mode_agreement": mode_agree,
                    "mean_score": mean_sc,
                })

    df = pd.DataFrame(rows)
    # Add convenience columns
    df["is_default"] = df["persona"] == "Default"
    df["is_likert"] = df["response_format"] == "likert_5"
    df["is_binary"] = df["response_format"].isin(["true_false", "yes_no"])
    return df


# ============================================================================
#                      ANALYSIS FUNCTIONS
# ============================================================================

def model_family(model: str) -> str:
    m = model.lower()
    if "gpt" in m: return "OpenAI"
    if "claude" in m: return "Anthropic"
    if "gemini" in m: return "Google"
    if "deepseek" in m: return "DeepSeek"
    if "qwen" in m: return "Alibaba"
    if "glm" in m: return "Zhipu"
    if "kimi" in m: return "Moonshot"
    if "minimax" in m: return "MiniMax"
    return "Other"


def compute_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}

    # Table 1: By model
    t1 = df.groupby("model").agg(
        mean_sd=("sd", "mean"),
        median_sd=("sd", "median"),
        mean_mode_agree=("mode_agreement", "mean"),
        exact_agree_rate=("exact_agreement", "mean"),
        mean_range=("range", "mean"),
        n_items=("sd", "count"),
    ).round(4).sort_values("mean_sd")
    t1.insert(0, "family", t1.index.map(model_family))
    tables["by_model"] = t1

    # Table 2: By persona
    t2 = df.groupby("persona").agg(
        mean_sd=("sd", "mean"),
        mean_mode_agree=("mode_agreement", "mean"),
        exact_agree_rate=("exact_agreement", "mean"),
        mean_range=("range", "mean"),
    ).round(4).sort_values("mean_sd")
    tables["by_persona"] = t2

    # Table 3: By domain
    t3 = df.groupby(["scale", "domain"]).agg(
        mean_sd=("sd", "mean"),
        mean_mode_agree=("mode_agreement", "mean"),
        exact_agree_rate=("exact_agreement", "mean"),
        n_items=("sd", "count"),
    ).round(4).sort_values("mean_sd")
    tables["by_domain"] = t3

    # Table 4: Top 20 most unstable items
    item_stats = df.groupby(["item_id", "scale", "domain", "item_text"]).agg(
        mean_sd=("sd", "mean"),
        max_sd=("sd", "max"),
        mean_mode_agree=("mode_agreement", "mean"),
    ).round(4).sort_values("mean_sd", ascending=False)
    tables["top_unstable_items"] = item_stats.head(20)

    # Table 5: Top 20 most stable items
    tables["top_stable_items"] = item_stats.tail(20).sort_values("mean_sd")

    return tables


# ============================================================================
#                      FIGURE GENERATION
# ============================================================================

def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {name}")


def fig_overall_portrait(df: pd.DataFrame):
    """Fig WSC-1: Overall consistency distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))

    # Panel A: SD distribution by response format
    ax = axes[0]
    for i, fmt in enumerate(["likert_5", "true_false", "yes_no"]):
        subset = df[df["response_format"] == fmt]["sd"]
        if len(subset) > 0:
            parts = ax.violinplot(subset.values, positions=[i], showmeans=True,
                                  showmedians=True, widths=0.7)
            for pc in parts["bodies"]:
                pc.set_alpha(0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Likert\n(1-5)", "True/\nFalse", "Yes/\nNo"])
    ax.set_ylabel("Within-item SD")
    ax.set_title("(a) SD by Response Format")

    # Panel B: Exact agreement rate by format
    ax = axes[1]
    agree_by_fmt = df.groupby("response_format")["exact_agreement"].mean()
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]
    bars = ax.bar(range(len(agree_by_fmt)), agree_by_fmt.values, color=colors[:len(agree_by_fmt)],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(agree_by_fmt)))
    ax.set_xticklabels([f.replace("_", "\n") for f in agree_by_fmt.index], fontsize=7)
    ax.set_ylabel("Exact Agreement Rate")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, agree_by_fmt.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=7)
    ax.set_title("(b) Exact Agreement Rate")

    # Panel C: SD distribution (all Likert)
    ax = axes[2]
    likert_sd = df[df["is_likert"]]["sd"]
    ax.hist(likert_sd, bins=30, color=PALETTE[3], edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.axvline(likert_sd.mean(), color="red", linestyle="--", linewidth=1, label=f"mean={likert_sd.mean():.3f}")
    ax.set_xlabel("Within-item SD")
    ax.set_ylabel("Count")
    ax.set_title("(c) Likert SD Distribution")
    ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, "fig_wsc1_overall_portrait.png")


def fig_model_beeswarm(df: pd.DataFrame):
    """Fig WSC-2: Model-level consistency (bee swarm)."""
    models_order = df.groupby("model")["sd"].mean().sort_values().index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, max(3, len(models_order) * 0.28)))

    # Panel A: SD bee swarm per model
    ax = axes[0]
    sns.stripplot(data=df, y="model", x="sd", order=models_order,
                  size=1.5, alpha=0.15, color=PALETTE[0], jitter=0.3, ax=ax)
    # Add mean markers
    means = df.groupby("model")["sd"].mean().reindex(models_order)
    ax.scatter(means.values, range(len(models_order)), color="red", s=25,
               zorder=5, marker="D", label="mean")
    ax.set_xlabel("Within-item SD")
    ax.set_ylabel("")
    ax.set_title("(a) SD Distribution per Model")
    ax.legend(fontsize=7, loc="lower right")

    # Panel B: Exact agreement rate per model
    ax = axes[1]
    agree = df.groupby("model")["exact_agreement"].mean().reindex(models_order)
    colors = [PALETTE[model_family(m) == "OpenAI" and 0 or
                       model_family(m) == "Anthropic" and 1 or
                       model_family(m) == "Google" and 2 or 3] for m in models_order]
    colors = []
    family_palette = {"OpenAI": PALETTE[0], "Anthropic": PALETTE[1], "Google": PALETTE[2],
                      "DeepSeek": PALETTE[3], "Alibaba": PALETTE[4], "Zhipu": PALETTE[5],
                      "Moonshot": PALETTE[6], "MiniMax": PALETTE[7], "Other": PALETTE[8]}
    colors = [family_palette.get(model_family(m), PALETTE[8]) for m in models_order]
    ax.barh(range(len(models_order)), agree.values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels(models_order)
    ax.set_xlabel("Exact Agreement Rate")
    ax.set_xlim(0, 1.05)
    ax.set_title("(b) Exact Agreement per Model")

    fig.tight_layout()
    _save(fig, "fig_wsc2_model_beeswarm.png")


def fig_persona_beeswarm(df: pd.DataFrame):
    """Fig WSC-3: Persona-level consistency."""
    personas_order = df.groupby("persona")["sd"].mean().sort_values().index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5))

    # Panel A: SD bee swarm per persona
    ax = axes[0]
    sns.stripplot(data=df, y="persona", x="sd", order=personas_order,
                  size=1.5, alpha=0.12, color=PALETTE[0], jitter=0.3, ax=ax)
    means = df.groupby("persona")["sd"].mean().reindex(personas_order)
    ax.scatter(means.values, range(len(personas_order)), color="red", s=25,
               zorder=5, marker="D", label="mean")
    # Highlight Default
    if "Default" in personas_order:
        idx = personas_order.index("Default")
        ax.axhspan(idx - 0.5, idx + 0.5, color="gold", alpha=0.3)
    ax.set_xlabel("Within-item SD")
    ax.set_ylabel("")
    ax.set_title("(a) SD Distribution per Persona")
    ax.legend(fontsize=7)

    # Panel B: Mode agreement per persona
    ax = axes[1]
    agree = df.groupby("persona")["mode_agreement"].mean().reindex(personas_order)
    bar_colors = ["gold" if p == "Default" else PALETTE[4] for p in personas_order]
    ax.barh(range(len(personas_order)), agree.values, color=bar_colors,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(personas_order)))
    ax.set_yticklabels(personas_order)
    ax.set_xlabel("Mode Agreement Rate")
    ax.set_title("(b) Mode Agreement per Persona")

    fig.tight_layout()
    _save(fig, "fig_wsc3_persona_beeswarm.png")


def fig_domain_violin(df: pd.DataFrame):
    """Fig WSC-4: Domain-level consistency (grouped violin)."""
    # Only Likert items have meaningful SD (binary SD is bimodal 0/0.4)
    likert = df[df["is_likert"]].copy()
    domains_order = likert.groupby("domain")["sd"].mean().sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # Panel A: Likert domains
    ax = axes[0]
    sns.violinplot(data=likert, y="domain", x="sd", order=domains_order,
                   inner="box", cut=0, ax=ax, palette="Set2")
    ax.set_xlabel("Within-item SD")
    ax.set_ylabel("")
    ax.set_title("(a) Likert Domains")

    # Panel B: Binary items agreement
    ax = axes[1]
    binary = df[df["is_binary"]].copy()
    bin_domains = binary.groupby("domain")["exact_agreement"].mean().sort_values().index.tolist()
    sns.violinplot(data=binary, y="domain", x="exact_agreement", order=bin_domains,
                   inner="box", cut=0, ax=ax, palette="Set3")
    ax.set_xlabel("Exact Agreement Rate")
    ax.set_ylabel("")
    ax.set_title("(b) Binary Domains (Agreement)")

    fig.tight_layout()
    _save(fig, "fig_wsc4_domain_violin.png")


def fig_item_stability(df: pd.DataFrame):
    """Fig WSC-5: Top/bottom items by consistency."""
    likert = df[df["is_likert"]].copy()
    item_mean = likert.groupby(["item_id", "domain"]).agg(
        mean_sd=("sd", "mean"),
        item_text=("item_text", "first"),
    ).reset_index()

    top_unstable = item_mean.nlargest(15, "mean_sd")
    top_stable = item_mean.nsmallest(15, "mean_sd")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4))

    # Panel A: Most unstable
    ax = axes[0]
    colors = sns.color_palette("Reds_r", len(top_unstable))
    texts = [f"{r.item_id}: {r.item_text[:30]}..." if len(r.item_text) > 30
             else f"{r.item_id}: {r.item_text}" for _, r in top_unstable.iterrows()]
    ax.barh(range(len(top_unstable)), top_unstable["mean_sd"].values, color=colors)
    ax.set_yticks(range(len(top_unstable)))
    ax.set_yticklabels(texts, fontsize=6)
    ax.set_xlabel("Mean SD across repetitions")
    ax.set_title("(a) 15 Most Unstable Items")
    ax.invert_yaxis()

    # Panel B: Most stable
    ax = axes[1]
    colors = sns.color_palette("Greens", len(top_stable))
    texts = [f"{r.item_id}: {r.item_text[:30]}..." if len(r.item_text) > 30
             else f"{r.item_id}: {r.item_text}" for _, r in top_stable.iterrows()]
    ax.barh(range(len(top_stable)), top_stable["mean_sd"].values, color=colors)
    ax.set_yticks(range(len(top_stable)))
    ax.set_yticklabels(texts, fontsize=6)
    ax.set_xlabel("Mean SD across repetitions")
    ax.set_title("(b) 15 Most Stable Items")
    ax.invert_yaxis()

    fig.tight_layout()
    _save(fig, "fig_wsc5_item_stability.png")


def fig_model_persona_heatmap(df: pd.DataFrame):
    """Fig WSC-6: Model × Persona clustering dendrogram + line profiles."""
    pivot = df.groupby(["model", "persona"])["sd"].mean().unstack("persona")

    model_order = df.groupby("model")["sd"].mean().sort_values().index
    pivot = pivot.reindex(model_order)
    pivot_filled = pivot.fillna(pivot.mean(axis=0))

    # Hierarchical clustering on models
    Z = linkage(pdist(pivot_filled.values, metric="correlation"), method="average")

    fig = plt.figure(figsize=(7.2, max(3, len(model_order) * 0.35)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.5], wspace=0.05)

    # Panel A: Dendrogram
    ax1 = fig.add_subplot(gs[0])
    dn = dendrogram(Z, labels=pivot_filled.index.tolist(), orientation="left",
                    ax=ax1, color_threshold=0.7 * max(Z[:, 2]))
    ax1.set_xlabel("Distance (1 - corr)")
    ax1.set_ylabel("")
    ax1.set_title("(a) Clustering", fontsize=9)
    ax1.tick_params(axis="y", labelsize=6)

    # Panel B: Line profiles ordered by dendrogram
    ax2 = fig.add_subplot(gs[1])
    dendro_order = dn["leaves"]
    family_palette = {"OpenAI": PALETTE[0], "Anthropic": PALETTE[1], "Google": PALETTE[2],
                      "DeepSeek": PALETTE[3], "Alibaba": PALETTE[4], "Zhipu": PALETTE[5],
                      "Moonshot": PALETTE[6], "MiniMax": PALETTE[7], "Other": PALETTE[8]}
    x = range(pivot_filled.shape[1])
    for idx in dendro_order:
        model_name = pivot_filled.index[idx]
        color = family_palette.get(model_family(model_name), PALETTE[8])
        ax2.plot(x, pivot_filled.iloc[idx].values, alpha=0.7, linewidth=1.2,
                 color=color, marker="o", markersize=2.5, label=model_name)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot_filled.columns, rotation=90, fontsize=5)
    ax2.set_ylabel("Mean Within-item SD")
    ax2.set_title("(b) Persona Consistency Profile", fontsize=9)
    # Add legend outside
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=5, loc="upper right", ncol=2,
               framealpha=0.7, labelspacing=0.3)

    fig.tight_layout()
    _save(fig, "fig_wsc6_model_persona_heatmap.png")


def fig_model_domain_heatmap(df: pd.DataFrame):
    """Fig WSC-7: Model × Domain clustering dendrogram + line profiles."""
    pivot = df.groupby(["model", "domain"])["sd"].mean().unstack("domain")

    model_order = df.groupby("model")["sd"].mean().sort_values().index
    pivot = pivot.reindex(model_order)
    domain_order = df.groupby("domain")["sd"].mean().sort_values(ascending=False).index
    pivot = pivot[domain_order]
    pivot_filled = pivot.fillna(pivot.mean(axis=0))

    # Hierarchical clustering on models by domain profile
    Z = linkage(pdist(pivot_filled.values, metric="correlation"), method="average")

    fig = plt.figure(figsize=(7.2, max(3, len(model_order) * 0.35)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.5], wspace=0.05)

    # Panel A: Dendrogram
    ax1 = fig.add_subplot(gs[0])
    dn = dendrogram(Z, labels=pivot_filled.index.tolist(), orientation="left",
                    ax=ax1, color_threshold=0.7 * max(Z[:, 2]))
    ax1.set_xlabel("Distance (1 - corr)")
    ax1.set_ylabel("")
    ax1.set_title("(a) Clustering", fontsize=9)
    ax1.tick_params(axis="y", labelsize=6)

    # Panel B: Line profiles
    ax2 = fig.add_subplot(gs[1])
    dendro_order = dn["leaves"]
    family_palette = {"OpenAI": PALETTE[0], "Anthropic": PALETTE[1], "Google": PALETTE[2],
                      "DeepSeek": PALETTE[3], "Alibaba": PALETTE[4], "Zhipu": PALETTE[5],
                      "Moonshot": PALETTE[6], "MiniMax": PALETTE[7], "Other": PALETTE[8]}
    x = range(pivot_filled.shape[1])
    for idx in dendro_order:
        model_name = pivot_filled.index[idx]
        color = family_palette.get(model_family(model_name), PALETTE[8])
        ax2.plot(x, pivot_filled.iloc[idx].values, alpha=0.7, linewidth=1.2,
                 color=color, marker="o", markersize=2.5, label=model_name)
    ax2.set_xticks(x)
    ax2.set_xticklabels(domain_order, rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("Mean Within-item SD")
    ax2.set_title("(b) Domain Consistency Profile", fontsize=9)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=5, loc="upper right", ncol=2,
               framealpha=0.7, labelspacing=0.3)

    fig.tight_layout()
    _save(fig, "fig_wsc7_model_domain_heatmap.png")


def fig_persona_domain_heatmap(df: pd.DataFrame):
    """Fig WSC-8: Persona × Domain clustering dendrogram + line profiles."""
    pivot = df.groupby(["persona", "domain"])["sd"].mean().unstack("domain")

    domain_order = df.groupby("domain")["sd"].mean().sort_values(ascending=False).index
    pivot = pivot[domain_order]
    pivot_filled = pivot.fillna(pivot.mean(axis=0))

    # Hierarchical clustering on personas by domain profile
    Z = linkage(pdist(pivot_filled.values, metric="correlation"), method="average")

    fig = plt.figure(figsize=(7.2, max(3, pivot_filled.shape[0] * 0.3)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.5], wspace=0.05)

    # Panel A: Dendrogram
    ax1 = fig.add_subplot(gs[0])
    dn = dendrogram(Z, labels=pivot_filled.index.tolist(), orientation="left",
                    ax=ax1, color_threshold=0.7 * max(Z[:, 2]))
    ax1.set_xlabel("Distance (1 - corr)")
    ax1.set_ylabel("")
    ax1.set_title("(a) Clustering", fontsize=9)
    ax1.tick_params(axis="y", labelsize=6)

    # Panel B: Line profiles
    ax2 = fig.add_subplot(gs[1])
    dendro_order = dn["leaves"]
    # Use a color per persona: Default=gold, others=cycling palette
    persona_colors = {}
    for i, p in enumerate(pivot_filled.index):
        persona_colors[p] = "gold" if p == "Default" else PALETTE[i % len(PALETTE)]
    x = range(pivot_filled.shape[1])
    for idx in dendro_order:
        persona_name = pivot_filled.index[idx]
        color = persona_colors.get(persona_name, PALETTE[8])
        lw = 2.0 if persona_name == "Default" else 1.0
        ax2.plot(x, pivot_filled.iloc[idx].values, alpha=0.7, linewidth=lw,
                 color=color, marker="o", markersize=2.5, label=persona_name)
    ax2.set_xticks(x)
    ax2.set_xticklabels(domain_order, rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("Mean Within-item SD")
    ax2.set_title("(b) Domain Consistency Profile", fontsize=9)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=5, loc="upper right", ncol=2,
               framealpha=0.7, labelspacing=0.3)

    fig.tight_layout()
    _save(fig, "fig_wsc8_persona_domain_heatmap.png")


def fig_anomaly_detection(df: pd.DataFrame):
    """Fig WSC-9: Anomaly cells — worst (model, persona, domain) combos."""
    # Aggregate to (model, persona, domain) level
    agg = df.groupby(["model", "persona", "domain"]).agg(
        mean_sd=("sd", "mean"),
        mean_agree=("exact_agreement", "mean"),
    ).reset_index()

    # Find anomalies: cells with SD > overall mean + 1.5 * IQR
    overall_sd = agg["mean_sd"]
    q1, q3 = overall_sd.quantile(0.25), overall_sd.quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    anomalies = agg[agg["mean_sd"] > threshold].sort_values("mean_sd", ascending=False).head(25)

    if anomalies.empty:
        print("  No anomalies detected (all cells within 1.5*IQR).")
        # Still generate the figure showing the distribution
        fig, ax = plt.subplots(figsize=(7.2, 3))
        agg["mean_sd"].hist(bins=50, ax=ax, color=PALETTE[4], edgecolor="black", linewidth=0.3)
        ax.axvline(threshold, color="red", linestyle="--", label=f"anomaly threshold={threshold:.3f}")
        ax.set_xlabel("Mean SD (model × persona × domain)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Cell-level SD (no anomalies)")
        ax.legend()
        fig.tight_layout()
        _save(fig, "fig_wsc9_anomaly.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, max(3, len(anomalies) * 0.18)))

    # Panel A: Top anomalies horizontal bar
    ax = axes[0]
    labels = [f"{r.model} / {r.persona}\n/ {r.domain}" for _, r in anomalies.iterrows()]
    ax.barh(range(len(anomalies)), anomalies["mean_sd"].values,
            color=PALETTE[3], edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(anomalies)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean SD")
    ax.set_title("(a) Worst (Model, Persona, Domain) Cells")
    ax.invert_yaxis()

    # Panel B: Overall cell distribution with threshold
    ax = axes[1]
    ax.hist(agg["mean_sd"], bins=60, color=PALETTE[4], edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1, label=f"threshold={threshold:.3f}")
    ax.set_xlabel("Mean SD")
    ax.set_ylabel("Count")
    ax.set_title("(b) Cell-level SD Distribution")
    ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, "fig_wsc9_anomaly.png")


# ============================================================================
#                        CSV TABLE EXPORTS
# ============================================================================

def save_tables(tables: dict[str, pd.DataFrame]):
    for name, tdf in tables.items():
        path = OUTPUT_DIR / f"wsc_{name}.csv"
        tdf.to_csv(path)
        print(f"  Saved {path.name}")


def save_main_dataframe(df: pd.DataFrame):
    path = OUTPUT_DIR / "wsc_item_level_consistency.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {path.name}")


# ============================================================================
#                              MAIN
# ============================================================================

def main():
    print("Loading results...")
    all_results = load_all_results()
    print(f"  {len(all_results)} models loaded")

    print("\nBuilding consistency frame...")
    df = build_consistency_frame(all_results)
    print(f"  {len(df)} (model, persona, item) observations")
    print(f"  Models: {df['model'].nunique()}, Personas: {df['persona'].nunique()}")
    print(f"  Items: {df['item_id'].nunique()}, Mean samples: {df['n_samples'].mean():.1f}")
    print(f"  Overall mean SD: {df['sd'].mean():.4f}")
    print(f"  Overall exact agreement: {df['exact_agreement'].mean():.4f}")

    print("\nComputing summary tables...")
    tables = compute_summary_tables(df)
    save_tables(tables)
    save_main_dataframe(df)

    print("\nGenerating figures...")
    fig_overall_portrait(df)
    fig_model_beeswarm(df)
    fig_persona_beeswarm(df)
    fig_domain_violin(df)
    fig_item_stability(df)
    fig_model_persona_heatmap(df)
    fig_model_domain_heatmap(df)
    fig_persona_domain_heatmap(df)
    fig_anomaly_detection(df)

    # Print summary for paper
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"Total observations: {len(df)}")
    print(f"Likert mean SD: {df[df.is_likert]['sd'].mean():.4f}")
    print(f"Binary exact agreement: {df[df.is_binary]['exact_agreement'].mean():.4f}")
    print(f"Likert exact agreement: {df[df.is_likert]['exact_agreement'].mean():.4f}")

    # Most/least consistent models
    model_summary = tables["by_model"]
    print(f"\nMost consistent model: {model_summary.index[0]} (mean SD={model_summary.iloc[0]['mean_sd']:.4f})")
    print(f"Least consistent model: {model_summary.index[-1]} (mean SD={model_summary.iloc[-1]['mean_sd']:.4f})")

    # Default vs MBTI
    default_sd = df[df["is_default"]]["sd"].mean()
    mbti_sd = df[~df["is_default"]]["sd"].mean()
    print(f"\nDefault persona SD: {default_sd:.4f}")
    print(f"MBTI persona SD: {mbti_sd:.4f}")
    print(f"Ratio (MBTI/Default): {mbti_sd / default_sd:.3f}")

    print("\nAll outputs saved.")


if __name__ == "__main__":
    main()
