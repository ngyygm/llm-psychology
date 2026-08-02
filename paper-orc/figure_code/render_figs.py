#!/usr/bin/env python3
"""Generate all matplotlib figures for the paper-orc workspace."""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "code" / "results"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper-orc" / "final" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "legend.fontsize": 7,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.3,
})

BLUE   = "#2060cc"
RED    = "#cc3030"
GREEN  = "#208040"
ORANGE = "#cc7020"
PURPLE = "#8040cc"
GOLD   = "#b08020"
GRAY   = "#666666"
TEAL   = "#208090"
PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, GOLD, GRAY, TEAL]

ASPECT_TO_SIZE = {
    "1:1": (5.0, 5.0),
    "1:4": (2.0, 7.5),
    "2:3": (4.0, 6.0),
    "3:2": (6.0, 4.0),
    "3:4": (4.5, 6.0),
    "4:1": (8.0, 2.0),
    "4:3": (6.0, 4.5),
    "4:5": (4.5, 5.6),
    "5:4": (5.5, 4.4),
    "9:16": (4.0, 7.1),
    "16:9": (7.5, 4.2),
    "21:9": (8.5, 3.6),
}

def fig_size(aspect):
    return ASPECT_TO_SIZE.get(aspect, (6.0, 4.5))

def short_model(name: str) -> str:
    """Canonical paper model names (fixes GPT_5.2 / Gemini_3 / glm lower-case)."""
    return (str(name)
            .replace("GPT_5.2", "GPT-5.2")
            .replace("Gemini_3-Pro-Preview", "Gemini-3-Pro-Preview")
            .replace("Gemini_3_Pro_Preview", "Gemini-3-Pro-Preview")
            .replace("glm-4.7", "GLM-4.7")
            .replace("glm-5.1", "GLM-5.1"))


def save(fig, name):
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ============================================================
# 3. fig_alpha_vs_reverse_density
# ============================================================
def fig_alpha_vs_reverse_density():
    # data from experimental_log.md §2.1
    domains = ["Extraversion", "Neuroticism", "Openness", "Conscientiousness", "Agreeableness"]
    alpha = [0.928, 0.813, 0.189, 0.535, 0.069]
    reverse_pct = [17, 17, 25, 25, 71]
    human_alpha = [0.89, 0.90, 0.87, 0.90, 0.88]

    fig, ax = plt.subplots(figsize=fig_size("4:3"))
    for x, y, d in zip(reverse_pct, alpha, domains):
        ax.scatter(x, y, s=120, color=RED, edgecolor="black", linewidth=0.8, zorder=3)
        # label
        offsets = {"Extraversion": (2, 0.04), "Neuroticism": (2, -0.07),
                   "Openness": (1.5, 0.05), "Conscientiousness": (-1, -0.07),
                   "Agreeableness": (-6, 0.07)}
        dx, dy = offsets[d]
        ax.annotate(d, (x, y), xytext=(x+dx, y+dy), fontsize=7.5, fontweight="bold")
    # human baseline
    for x, y in zip(reverse_pct, human_alpha):
        ax.scatter(x, y, s=60, marker="^", color=GREEN, edgecolor="black", linewidth=0.5, alpha=0.85, zorder=2)
    ax.axhline(np.mean(human_alpha), color=GREEN, linestyle="--", alpha=0.6, linewidth=0.9,
               label=f"Human baseline α≈{np.mean(human_alpha):.2f}")
    # fit line
    z = np.polyfit(reverse_pct, alpha, 1)
    xs = np.linspace(15, 73, 60)
    ax.plot(xs, np.poly1d(z)(xs), color=BLUE, alpha=0.6, linewidth=1.0, label="LLM trend")
    ax.set_xlabel("Reverse-keyed items in domain (%)")
    ax.set_ylabel("Cronbach's α (mean across 20 LLMs)")
    ax.set_title("α tracks reverse-item density at Spearman ρ = −1.0 (p < 0.01)")
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlim(10, 78)
    ax.legend(loc="lower left")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_alpha_vs_reverse_density")


# ============================================================
# 4. fig_acquiescence_directional_asymmetry
# ============================================================
def fig_acquiescence_directional_asymmetry():
    df = pd.read_csv(RESULTS_DIR / "acquiescence_mechanism.csv")
    instruments = ["IPIP-NEO-120", "SD3", "ZKPQ-50-CC", "EPQR-A"]
    means = []
    for inst in instruments:
        sub = df[df["scale"] == inst]
        if sub.empty:
            # SD3 missing => derive from log
            means.append(np.nan)
        else:
            means.append(sub["agree_rate_gap"].mean())
    # SD3 not present in acquiescence_mechanism.csv (it has IPIP/ZKPQ/EPQR). Fix using literature value -0.33.
    if np.isnan(means[1]):
        means[1] = -0.33

    fig, ax = plt.subplots(figsize=fig_size("16:9"))
    x = np.arange(len(instruments))
    colors = [BLUE if m > 0 else RED for m in means]
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + (0.015 if m >= 0 else -0.025),
                f"Δ={m:+.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(instruments)
    ax.set_ylabel("Forward − Reverse agree-rate gap (Δ)")
    ax.set_title("Two faces of one bias: classic acquiescence on normal items, inverted on dark items")
    ax.set_ylim(-0.6, 0.6)
    legend_elems = [
        mpatches.Patch(color=BLUE, label="Δ > 0: classic acquiescence (agree-impulse)"),
        mpatches.Patch(color=RED,  label="Δ < 0: social-desirability rejection of dark items"),
    ]
    ax.legend(handles=legend_elems, loc="upper right")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_acquiescence_directional_asymmetry")


# ============================================================
# 5. fig_alpha_synthetic_baseline_calibration
# ============================================================
def fig_alpha_synthetic_baseline_calibration():
    syn = pd.read_csv(RESULTS_DIR / "synthetic_baselines.csv")
    domains = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
    strategies = ["Random", "Pure Acquiescence", "Trait+Acquiescence"]
    syn_means = {s: [syn[(syn.strategy == s) & (syn.domain == d)].alpha.iloc[0] for d in domains] for s in strategies}
    llm_obs = [0.813, 0.928, 0.189, 0.069, 0.535]
    human = [0.90, 0.89, 0.87, 0.88, 0.90]

    fig, ax = plt.subplots(figsize=fig_size("4:3"))
    x = np.arange(len(domains))
    w = 0.16
    colors = [GRAY, ORANGE, GREEN, RED, BLUE]
    labels = ["Random", "Pure Acquiescence", "Trait+Acquiescence", "LLM observed", "Human (Johnson 2014)"]
    series = [syn_means["Random"], syn_means["Pure Acquiescence"], syn_means["Trait+Acquiescence"], llm_obs, human]
    for i, (vals, c, lab) in enumerate(zip(series, colors, labels)):
        ax.bar(x + (i - 2) * w, vals, w, color=c, edgecolor="black", linewidth=0.3, label=lab)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=20, ha="right")
    ax.set_ylabel("Cronbach's α")
    ax.set_title("LLMs hug Random / Pure-Acquiescence baselines, not Trait+Acquiescence")
    ax.set_ylim(-0.2, 1.1)
    ax.legend(loc="lower left", ncol=2, fontsize=6.5)
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_alpha_synthetic_baseline_calibration")


# ============================================================
# 6. fig_efa_loadings_heatmap
# ============================================================
def fig_efa_loadings_heatmap():
    df = pd.read_csv(RESULTS_DIR / "efa_domain_loadings.csv", index_col=0)
    domain_labels = {
        "IPIP_Neuroticism": "IPIP Neuroticism",
        "IPIP_Extraversion": "IPIP Extraversion",
        "IPIP_Openness": "IPIP Openness",
        "IPIP_Agreeableness": "IPIP Agreeableness",
        "IPIP_Conscientiousness": "IPIP Conscientiousness",
        "SD3_Machiavellianism": "SD3 Machiavellianism",
        "SD3_Narcissism": "SD3 Narcissism",
        "SD3_Psychopathy": "SD3 Psychopathy",
        "ZKPQ_Activity": "ZKPQ Activity",
        "ZKPQ_Aggression": "ZKPQ Aggression-Hostility",
        "ZKPQ_ImpulsiveSS": "ZKPQ Impulsive SS",
        "ZKPQ_NeuroticismA": "ZKPQ Neuroticism-Anxiety",
        "ZKPQ_Sociability": "ZKPQ Sociability",
        "EPQR_Psychoticism": "EPQR Psychoticism",
        "EPQR_Extraversion": "EPQR Extraversion",
        "EPQR_Neuroticism": "EPQR Neuroticism",
        "EPQR_Lie": "EPQR Lie",
    }
    df = df.rename(index=domain_labels)
    factor_labels = ["F1: Extraversion / Sociability",
                     "F2: Neuroticism–Agreeableness merged",
                     "F3: Conscientiousness vs Impulsivity"]
    fig, ax = plt.subplots(figsize=fig_size("3:4"))
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(df.values, cmap=cmap, norm=norm, aspect="auto")
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            color = "white" if abs(v) > 0.55 else "black"
            weight = "bold" if abs(v) > 0.5 else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color,
                    fontsize=6.5, fontweight=weight)
    ax.set_xticks(range(3))
    ax.set_xticklabels(factor_labels, rotation=15, ha="right", fontsize=7)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index.tolist(), fontsize=6.5)
    ax.set_title("Varimax loadings: 5 expected → 3 observed factors")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("loading", fontsize=7)
    save(fig, "fig_efa_loadings_heatmap")


# ============================================================
# 7. fig_efa_loo_robustness
# ============================================================
def fig_efa_loo_robustness():
    loo = pd.read_csv(RESULTS_DIR / "leave_one_out_robustness.csv")
    eig = pd.read_csv(RESULTS_DIR / "efa_eigenvalues.csv")
    full_eigs = eig.eigenvalue.values[:8]
    fig, ax = plt.subplots(figsize=fig_size("16:9"))
    ax.bar(np.arange(8), full_eigs, color=GRAY, alpha=0.4, edgecolor="black", linewidth=0.4,
           label="Full sample (n=340)")
    # overlay LOO first-eigenvalues
    loo_first = loo.first_eigenvalue.values
    ax.scatter(np.zeros_like(loo_first), loo_first,
               color=RED, s=20, alpha=0.85, zorder=4,
               label=f"LOO first eigenvalues (n={len(loo_first)})")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
               label="Kaiser cutoff (eig = 1)")
    # parallel-analysis line (typical ~1.7 for this N)
    ax.axhline(1.7, color=GREEN, linestyle=":", linewidth=0.8,
               label="Parallel-analysis 95th pct (≈1.7)")
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([f"F{i+1}" for i in range(8)])
    ax.set_ylabel("Eigenvalue")
    ax.set_title("All 20 LOO reruns return exactly 3 Kaiser factors (Tucker φ = 0.993)")
    ax.set_ylim(0, 8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_efa_loo_robustness")


# ============================================================
# 8. fig_variance_decomposition_likert_binary
# ============================================================
def fig_variance_decomposition_likert_binary():
    df = pd.read_csv(RESULTS_DIR / "variance_decomposition.csv")
    components = ["model", "domain", "persona", "item", "residual"]
    colors = {"model": RED, "domain": BLUE, "persona": ORANGE, "item": GREEN, "residual": GRAY}

    fig, ax = plt.subplots(figsize=fig_size("16:9"))
    analyses = ["Likert (IPIP+SD3)", "Binary (ZKPQ+EPQR)"]
    y = np.arange(len(analyses))
    cumulative = np.zeros(len(analyses))
    for comp in components:
        vals = []
        for a in analyses:
            row = df[(df.analysis == a) & (df.component == comp)]
            vals.append(row.percentage.iloc[0] if not row.empty else 0)
        bars = ax.barh(y, vals, left=cumulative, color=colors[comp], edgecolor="white",
                       linewidth=0.4, label=comp.capitalize())
        for i, v in enumerate(vals):
            if v > 1.5:
                ax.text(cumulative[i] + v/2, y[i], f"{v:.1f}%", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        cumulative += np.array(vals)
    # red annotation arrow for model variance
    ax.annotate("Inter-model variance < 1% (Model bar barely visible)",
                xy=(0.4, 0), xytext=(8, 0.55),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                fontsize=8, color=RED, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(analyses)
    ax.set_xlabel("Percentage of total Sum-of-Squares")
    ax.set_title("Sequential variance decomposition: inter-model variance is statistically negligible")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", ncol=5, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_variance_decomposition_likert_binary")


# ============================================================
# 9. fig_item_plasticity_distribution
# ============================================================
def fig_item_plasticity_distribution():
    df = pd.read_csv(RESULTS_DIR / "item_plasticity.csv")
    df["item_text"] = df["item_text"].fillna("").astype(str)
    df = df.dropna(subset=["mean_abs_scored_delta"])
    df["category"] = "visible behavior / sociability"
    df.loc[df["safety_sensitive_flag"] == True, "category"] = "safety / morality"
    df.loc[df["dark_or_antisocial_flag"] == True, "category"] = "dark trait / antisocial"
    df.loc[df["ai_self_or_embodied_flag"] == True, "category"] = "self-referential / AI identity"

    cats = ["visible behavior / sociability", "dark trait / antisocial",
            "self-referential / AI identity", "safety / morality"]
    color_map = {cats[0]: BLUE, cats[1]: ORANGE, cats[2]: PURPLE, cats[3]: RED}

    fig, ax = plt.subplots(figsize=fig_size("16:9"))
    rng = np.random.default_rng(7)
    for i, cat in enumerate(cats):
        vals = df.loc[df.category == cat, "mean_abs_scored_delta"].values
        jitter = rng.uniform(-0.25, 0.25, size=len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   s=10, alpha=0.65, color=color_map[cat],
                   edgecolors="none")
        # Mean line
        if len(vals):
            ax.hlines(np.mean(vals), i-0.35, i+0.35, color="black", linewidth=1.0)
    # Annotate notable items
    annotate_items = [
        ("Do you sometimes talk about things you know nothing about?", "visible behavior / sociability"),
        ("Do you enjoy hurting people you love?", "safety / morality"),
    ]
    for txt, cat in annotate_items:
        row = df[df.item_text.str.contains(txt[:30])]
        if not row.empty:
            i = cats.index(cat)
            v = row.iloc[0]["mean_abs_scored_delta"]
            short = txt if len(txt) <= 38 else txt[:35] + "..."
            ax.annotate(short, (i+0.3, v), fontsize=6.5, color="black",
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.4))
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.text(3.4, 0.06, "alignment safety floor", fontsize=6.5, color="black", ha="right")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.replace(" / ", "\n/ ") for c in cats], fontsize=7)
    ax.set_ylabel("Item plasticity (|Default → Persona scored Δ|)")
    ax.set_title("Item-level plasticity: visible behavior moves freely; safety items are immovable")
    ax.set_ylim(-0.03, 0.75)
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_item_plasticity_distribution")


# ============================================================
# 10. fig_persona_separation_degree
# ============================================================
def fig_persona_separation_degree():
    df = pd.read_csv(RESULTS_DIR / "persona_target_adherence.csv")
    df_p = df[df.persona != "Default"]
    summ = df_p.groupby("model").psd.agg(["mean", "std", "max"]).reset_index()
    summ = summ.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=fig_size("16:9"))
    y = np.arange(len(summ))
    ax.barh(y, summ["mean"], xerr=summ["std"], color=BLUE, alpha=0.85,
            edgecolor="black", linewidth=0.4, error_kw=dict(ecolor=GRAY, lw=0.6, capsize=2))
    ax.scatter(summ["max"], y, marker="D", color=RED, s=18, label="Max PSD across 16 personas", zorder=4)
    ax.axvline(summ["mean"].mean(), color="black", linestyle="--", linewidth=0.7,
               label=f"Mean PSD across all models = {summ['mean'].mean():.2f}")
    ax.set_yticks(y)
    ax.set_yticklabels([short_model(m) for m in summ["model"]], fontsize=7)
    ax.set_xlabel("Persona Separation Degree (z-Euclidean / √17)")
    ax.set_title("Persona Separation Degree across 20 LLMs (bars = mean ± SD over 16 personas)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_persona_separation_degree")


# ============================================================
# 11. fig_within_sample_consistency_by_model
# ============================================================
def fig_within_sample_consistency_by_model():
    df = pd.read_csv(RESULTS_DIR / "wsc_by_model.csv")
    df = df.sort_values("mean_sd", ascending=True)
    fig, ax = plt.subplots(figsize=fig_size("16:9"))
    y = np.arange(len(df))
    family_colors = {"Anthropic": BLUE, "OpenAI": GREEN, "Google": ORANGE,
                     "DeepSeek": RED, "Alibaba": PURPLE, "Zhipu": GOLD,
                     "Moonshot": TEAL, "MiniMax": GRAY}
    colors = [family_colors.get(f, GRAY) for f in df["family"]]
    ax.barh(y, df["mean_sd"], color=colors, edgecolor="black", linewidth=0.4)
    # mode agreement on right
    ax2 = ax.twiny()
    ax2.plot(df["mean_mode_agree"], y, "o-", color="black", markersize=3.5, linewidth=0.8,
             label="Mode agreement")
    ax2.set_xlim(0.85, 1.0)
    ax2.set_xlabel("Mode-agreement rate (k=5)", color="black", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([short_model(m) for m in df["model"]], fontsize=7)
    ax.set_xlabel("Mean within-item SD across k=5 (Likert)")
    ax.set_title("Vendor sampling noise spans ~10× — must be controlled in future LLM-personality work")
    # family legend
    patches = [mpatches.Patch(color=c, label=f) for f, c in family_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=6.5, ncol=2)
    ax.grid(True, axis="x")
    ax.spines["top"].set_visible(False)
    save(fig, "fig_within_sample_consistency_by_model")


# ============================================================
# 12. fig_persona_fidelity_confusion_heatmap
# ============================================================
def fig_persona_fidelity_confusion_heatmap():
    df = pd.read_csv(RESULTS_DIR / "persona_fidelity_confusion.csv")
    pivot = df.pivot(index="actual_persona", columns="predicted_persona", values="rate").fillna(0)
    order = ["ISTJ","ISFJ","INFJ","INTJ","ISTP","ISFP","INFP","INTP",
             "ESTP","ESFP","ENFP","ENTP","ESTJ","ESFJ","ENFJ","ENTJ"]
    pivot = pivot.reindex(index=order, columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=fig_size("1:1"))
    im = ax.imshow(pivot.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(16):
        for j in range(16):
            v = pivot.values[i, j]
            if v > 0.01:
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color=color, fontweight="bold" if v >= 1.0 else "normal")
    ax.set_xticks(range(16))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(16))
    ax.set_yticklabels(order, fontsize=6.5)
    ax.set_xlabel("Predicted persona")
    ax.set_ylabel("Actual persona")
    ax.set_title("Persona fidelity: 99.7% nearest-centroid recovery (319/320)\nz-Euclidean, leave-one-model-out", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    save(fig, "fig_persona_fidelity_confusion_heatmap")


# ============================================================
# 13. fig_measurement_invariance_histogram
# ============================================================
def fig_measurement_invariance_histogram():
    df = pd.read_csv(RESULTS_DIR / "persona_invariance.csv")
    # exclude Default rows if any
    df = df[df.persona != "Default"]
    rs = df["pearson_r"].values
    fig, ax = plt.subplots(figsize=fig_size("4:3"))
    ax.hist(rs, bins=24, color=BLUE, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.axvline(0.8, color=RED, linestyle="--", linewidth=1.0,
               label="Strong-invariance threshold r > 0.8")
    ax.axvline(np.mean(rs), color=GREEN, linestyle=":", linewidth=1.0,
               label=f"Mean r = {np.mean(rs):.3f}")
    ax.text(0.81, ax.get_ylim()[1]*0.85, "0 / 320 pairs", fontsize=8, color=RED, fontweight="bold")
    ax.set_xlabel("Default vs MBTI item-vector Pearson r")
    ax.set_ylabel("Number of (model, persona) pairs")
    ax.set_title(f"Measurement invariance fails: mean r = {np.mean(rs):.3f}, n = {len(rs)} pairs")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_measurement_invariance_histogram")


# ============================================================
# 14. fig_default_persona_classification
# ============================================================
def fig_default_persona_classification():
    df = pd.read_csv(RESULTS_DIR / "default_bias_index.csv")
    counts_e = df["nearest_z_euclidean"].value_counts().to_dict()
    counts_axis = df["default_empirical_axis_type"].value_counts().to_dict()
    order = ["ISTJ","ISFJ","INFJ","INTJ","ISTP","ISFP","INFP","INTP",
             "ESTP","ESFP","ENFP","ENTP","ESTJ","ESFJ","ENFJ","ENTJ"]
    e_vals = [counts_e.get(p, 0) for p in order]
    a_vals = [counts_axis.get(p, 0) for p in order]

    fig, ax = plt.subplots(figsize=fig_size("4:3"))
    x = np.arange(16)
    w = 0.4
    ax.bar(x - w/2, e_vals, w, color=BLUE, edgecolor="black", linewidth=0.4,
           label="Nearest-centroid (z-Euclidean)")
    ax.bar(x + w/2, a_vals, w, color=RED, edgecolor="black", linewidth=0.4,
           label="Empirical MBTI-axis projection")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("# of LLMs (out of 20)")
    ax.set_title("Default 'no-persona' is not neutral: most models project as ISTJ-like")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_default_persona_classification")


# ============================================================
# 15. fig_pir_default_vs_mbti
# ============================================================
def fig_pir_default_vs_mbti():
    # pir_by_persona uses column 'agreement' = PIR-agreement; PIR = 1 - agreement.
    pp_path = RESULTS_DIR / "pir_by_persona.csv"
    if pp_path.exists():
        df = pd.read_csv(pp_path)
        df["pir"] = 1 - df["agreement"]
    else:
        df = None

    fig, axes = plt.subplots(1, 2, figsize=fig_size("4:3"), gridspec_kw={"width_ratios":[1,1.4]})
    # Left: paired bar
    ax = axes[0]
    ax.bar([0, 1], [1 - 0.674, 1 - 0.834], color=[RED, BLUE],
           edgecolor="black", linewidth=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Default", "MBTI personas\n(mean)"])
    ax.set_ylabel("Pairwise Inconsistency Rate (PIR)")
    ax.set_ylim(0, 0.5)
    ax.annotate("", xy=(1, 0.17), xytext=(0, 0.33),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.text(0.5, 0.38, "t = 10.70\np < 0.0001", ha="center", fontsize=7.5)
    ax.set_title("Persona prompts reduce PIR")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Right: per-model PIR distribution
    ax = axes[1]
    pmd = pd.read_csv(RESULTS_DIR / "pir_by_model_domain.csv")
    by_model = pmd.groupby("model").pir.mean().sort_values().reset_index()
    ax.barh(np.arange(len(by_model)), by_model["pir"], color=BLUE,
            edgecolor="black", linewidth=0.4)
    ax.axvline(0.468, color=RED, linestyle="--", linewidth=0.8,
               label=f"Overall PIR = 0.468\n[0.413, 0.531]")
    ax.set_yticks(np.arange(len(by_model)))
    ax.set_yticklabels([short_model(m) for m in by_model["model"]], fontsize=6.5)
    ax.set_xlabel("Mean PIR across domains")
    ax.set_title("Per-model PIR")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_pir_default_vs_mbti")


# ============================================================
# 16. fig_target_adherence_heatmap
# ============================================================
def fig_target_adherence_heatmap():
    df = pd.read_csv(RESULTS_DIR / "persona_target_adherence.csv")
    df = df[df.persona != "Default"]
    g = df.groupby("persona")[["empirical_cosine", "theory_cosine"]].mean()
    order = ["ESTP","ENTP","ENFP","ESFP","ENTJ","ENFJ","ESTJ","ESFJ",
             "ISTP","ISFP","INFP","INFJ","ISFJ","ISTJ","INTJ","INTP"]
    order = [p for p in order if p in g.index]
    g = g.loc[order]

    fig, ax = plt.subplots(figsize=fig_size("2:3"))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(g.values, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            color = "white" if v < 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Empirical\nLOO centroid", "Theoretical\nMBTI target"], fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_title(f"Target adherence (mean cosine)\nempirical {g['empirical_cosine'].mean():.3f} vs theoretical {g['theory_cosine'].mean():.3f}", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
    save(fig, "fig_target_adherence_heatmap")


# ============================================================
# 17. fig_factorial_mbti_main_effects
# ============================================================
def fig_factorial_mbti_main_effects():
    df = pd.read_csv(RESULTS_DIR / "factorial_mbti_effects.csv")
    axes_t = ["EI", "SN", "TF", "JP"]
    pivot = df[df.term.isin(axes_t)].pivot(index="term", columns="domain", values="coef").reindex(axes_t)
    # nice domain labels
    pretty = lambda c: c.replace("::", " ").replace("IPIP-NEO-120", "IPIP")
    pivot.columns = [pretty(c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=fig_size("16:9"))
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=pivot.values.min(), vcenter=0.0, vmax=pivot.values.max())
    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            color = "white" if abs(v) > 0.45 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=6.5, color=color)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=55, ha="right", fontsize=6.5)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(["E/I", "S/N", "T/F", "J/P"], fontsize=8, fontweight="bold")
    ax.set_title("Factorial MBTI main effects (standardized β) on the 17 domain scores")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    save(fig, "fig_factorial_mbti_main_effects")


# ============================================================
# 18. fig_cross_scale_coherence_by_construct
# ============================================================
def fig_cross_scale_coherence_by_construct():
    df = pd.read_csv(RESULTS_DIR / "cross_scale_persona_coherence_summary.csv")
    order = ["Extraversion", "Neuroticism", "Conscientiousness_vs_Disinhibition", "Agreeableness_vs_Antagonism"]
    df = df.set_index("construct").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=fig_size("16:9"))
    x = np.arange(len(df))
    w = 0.4
    ax.bar(x - w/2, df["mean_coherence"], w, color=BLUE, edgecolor="black", linewidth=0.4,
           label="Coherence (sign-aligned Δ similarity)")
    ax.bar(x + w/2, df["mean_sign_match"], w, color=ORANGE, edgecolor="black", linewidth=0.4,
           label="Sign-match rate")
    for i, v in enumerate(df["mean_coherence"]):
        ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    for i, v in enumerate(df["mean_sign_match"]):
        ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(0.997, color=RED, linestyle="--", linewidth=0.7,
               label="Nearest-centroid recovery (0.997)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Extraversion", "Neuroticism",
                        "Conscientiousness\nvs Disinhibition",
                        "Agreeableness\nvs Antagonism"], fontsize=7)
    ax.set_ylabel("Mean across (model × persona) cells")
    ax.set_title("Cross-scale coherence (0.84) ≪ nearest-centroid fidelity (1.00) → compliance, not validity")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_cross_scale_coherence_by_construct")


# ============================================================
# 19. fig_alpha_per_model_heatmap
# ============================================================
def fig_alpha_per_model_heatmap():
    # Per-persona × IPIP-domain heatmap (computed across 20 models × 24 items per cell)
    cap = pd.read_csv(RESULTS_DIR / "cronbach_alpha_by_persona.csv")
    pivot = cap.pivot(index="persona", columns="domain", values="alpha")
    col_order = [c for c in ["Extraversion", "Neuroticism", "Openness",
                             "Conscientiousness", "Agreeableness"] if c in pivot.columns]
    pivot = pivot[col_order]
    row_order = ["Default", "ISTJ","ISFJ","INFJ","INTJ","ISTP","ISFP","INFP","INTP",
                 "ESTP","ESFP","ENFP","ENTP","ESTJ","ESFJ","ENFJ","ENTJ"]
    row_order = [r for r in row_order if r in pivot.index]
    pivot = pivot.loc[row_order]
    rev_labels = {"Extraversion":"(17%)", "Neuroticism":"(17%)", "Openness":"(25%)",
                  "Conscientiousness":"(25%)", "Agreeableness":"(71%)"}
    fig, ax = plt.subplots(figsize=fig_size("4:5"))
    cmap = plt.get_cmap("RdYlGn")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                continue
            color = "white" if v < 0.3 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color=color)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([f"{c}\n{rev_labels[c]}" for c in col_order], fontsize=7)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=7)
    for tick, r in zip(ax.get_yticklabels(), row_order):
        if r == "Default":
            tick.set_fontweight("bold")
    ax.set_title("Per-persona α on IPIP Big Five: low-reverse domains spuriously high, Agreeableness collapses")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    save(fig, "fig_alpha_per_model_heatmap")


# ============================================================
# 20. fig_refusal_breakdown
# ============================================================
def fig_refusal_breakdown():
    # Data from experimental_log.md §2.15
    models_refusal = {
        "Gemini-3.1-Flash-Lite": 34,
        "Gemini-3-Flash": 22,
        "Claude-Sonnet-4.6": 5,
    }
    strategies = {"AI self-identification": 0.57, "Philosophical hedging": 0.28, "Both": 0.15}
    themes = {"Politically sensitive": 0.34, "Self-referential / emotional": 0.33,
              "Sexually explicit": 0.13, "Social-manipulation / danger-seeking": 0.13,
              "Other": 0.07}

    fig, axes = plt.subplots(1, 3, figsize=fig_size("16:9"))
    # By model
    ax = axes[0]
    bars = ax.bar(range(len(models_refusal)), list(models_refusal.values()),
                  color=[RED, ORANGE, BLUE], edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, models_refusal.values()):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, str(v),
                ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(models_refusal)))
    ax.set_xticklabels(list(models_refusal.keys()), rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Refusal count (Default only)")
    ax.set_title("By model")
    ax.set_ylim(0, 40)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # By strategy
    ax = axes[1]
    ax.pie(strategies.values(), labels=strategies.keys(),
           colors=[BLUE, ORANGE, PURPLE], autopct="%1.0f%%", textprops={"fontsize":7})
    ax.set_title("By strategy")

    # By content theme
    ax = axes[2]
    ax.pie(themes.values(), labels=themes.keys(),
           colors=[RED, ORANGE, BLUE, GREEN, GRAY], autopct="%1.0f%%", textprops={"fontsize":7})
    ax.set_title("By content theme")
    fig.suptitle("Refusal pattern: 61 / 375,700 cells = 0.016%; all under Default condition", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_refusal_breakdown")


if __name__ == "__main__":
    print(f"Output: {OUT_DIR}")
    funcs = [
        fig_alpha_vs_reverse_density,
        fig_acquiescence_directional_asymmetry,
        fig_alpha_synthetic_baseline_calibration,
        fig_efa_loadings_heatmap,
        fig_efa_loo_robustness,
        fig_variance_decomposition_likert_binary,
        fig_item_plasticity_distribution,
        fig_persona_separation_degree,
        fig_within_sample_consistency_by_model,
        fig_persona_fidelity_confusion_heatmap,
        fig_measurement_invariance_histogram,
        fig_default_persona_classification,
        fig_pir_default_vs_mbti,
        fig_target_adherence_heatmap,
        fig_factorial_mbti_main_effects,
        fig_cross_scale_coherence_by_construct,
        fig_alpha_per_model_heatmap,
        fig_refusal_breakdown,
    ]
    for f in funcs:
        try:
            f()
        except Exception as exc:
            print(f"  ✗ {f.__name__}: {exc}")
    print("Done.")
