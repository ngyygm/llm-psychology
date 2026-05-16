#!/usr/bin/env python3
"""Regenerate and synchronize all experiment figures with a unified style."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis_output"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
WORKSPACE = ROOT / "workspace"

for path in [
    FIGURES,
    WORKSPACE / "figures",
    WORKSPACE / "inputs" / "figures",
    WORKSPACE / "drafts",
    WORKSPACE / "final",
]:
    path.mkdir(parents=True, exist_ok=True)


# Colorblind-safe base palette plus muted category colors.
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#8A8A8A",
    "light_gray": "#D8D8D8",
    "dark": "#222222",
}

FAMILY_COLORS = {
    "OpenAI": C["orange"],
    "Anthropic": C["purple"],
    "Google": C["blue"],
    "Alibaba": C["red"],
    "DeepSeek": C["green"],
    "Moonshot": C["cyan"],
    "MiniMax": "#7F7F7F",
    "Zhipu": "#6C4F3D",
    "Other": C["gray"],
}

SCALE_COLORS = {
    "IPIP-NEO-120": "#264653",
    "SD3": "#2A9D8F",
    "ZKPQ-50-CC": "#E76F51",
    "EPQR-A": "#F4A261",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.color": "#EAEAEA",
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 260,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


IPIP_DOMAINS = [
    "Neuroticism",
    "Extraversion",
    "Openness",
    "Agreeableness",
    "Conscientiousness",
]


def save(fig: matplotlib.figure.Figure, *names: str) -> None:
    for name in names:
        fig.savefig(FIGURES / name, dpi=260, bbox_inches="tight", pad_inches=0.08)
        print(f"saved figures/{name}")
    plt.close(fig)


def short_model(name: str, max_len: int = 16) -> str:
    replacements = [
        ("Gemini_3-Pro-Preview", "Gem-3-Pro"),
        ("Gemini-3-Pro-Preview", "Gem-3-Pro"),
        ("Gemini-3-Flash-Preview", "Gem-3-Flash"),
        ("Gemini-3.1-Flash-Lite", "Gem-3.1-FL"),
        ("Gemini-3.1-Pro-Preview", "Gem-3.1-Pro"),
        ("Claude-Opus-4.6", "Claude-Opus"),
        ("Claude-Sonnet-4.6", "Claude-Sonnet"),
        ("DeepSeek-V4-Flash", "DS-V4-Flash"),
        ("DeepSeek-V4-Pro", "DS-V4-Pro"),
        ("DeepSeek-V3.2", "DS-V3.2"),
        ("Qwen3.5-397B-A17B", "Qwen3.5-397"),
        ("Qwen3.5-122B-A10B", "Qwen3.5-122"),
        ("Qwen3-235B-A22B", "Qwen3-235"),
        ("MiniMax-M2.7", "MiniMax"),
        ("GPT_5.2", "GPT-5.2"),
    ]
    out = name
    for old, new in replacements:
        out = out.replace(old, new)
    return out[:max_len]


def family(name: str) -> str:
    if "GPT" in name:
        return "OpenAI"
    if "Claude" in name:
        return "Anthropic"
    if "Gemini" in name:
        return "Google"
    if "Qwen" in name:
        return "Alibaba"
    if "DeepSeek" in name:
        return "DeepSeek"
    if "Kimi" in name:
        return "Moonshot"
    if "MiniMax" in name:
        return "MiniMax"
    if "GLM" in name:
        return "Zhipu"
    return "Other"


def short_domain(label: str) -> str:
    label = label.replace("IPIP-NEO-120::", "IPIP ")
    label = label.replace("ZKPQ-50-CC::", "ZKPQ ")
    label = label.replace("EPQR-A::", "EPQR ")
    label = label.replace("SD3::", "SD3 ")
    label = label.replace("IPIP_", "IPIP ")
    label = label.replace("ZKPQ_", "ZKPQ ")
    label = label.replace("EPQR_", "EPQR ")
    label = label.replace("SD3_", "SD3 ")
    replacements = {
        "Neuroticism-Anxiety": "N-Anx",
        "Aggression-Hostility": "Agg",
        "Impulsive_Sensation_Seeking": "ImpSS",
        "Machiavellianism": "Mach",
        "Narcissism": "Narc",
        "Psychopathy": "Psych",
        "Psychoticism": "Psych",
        "Extraversion": "Extra",
        "Neuroticism": "Neuro",
        "Agreeableness": "Agree",
        "Conscientiousness": "Consc",
        "Openness": "Open",
        "Sociability": "Social",
        "Activity": "Act",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def load_json_results() -> tuple[dict[str, dict], list[str], list[str], list[str], np.ndarray]:
    data: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("exp_mbti_*.json")):
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        data[payload.get("model_name", path.stem.replace("exp_mbti_", ""))] = payload

    models = sorted(data)
    personas = list(data[models[0]]["results_by_persona"].keys())
    first = next(iter(data[models[0]]["results_by_persona"].values()))
    domains = sorted(first["domain_scores"].keys())

    scores = np.zeros((len(models), len(personas), len(domains)))
    for i, model in enumerate(models):
        for j, persona in enumerate(personas):
            ds = data[model]["results_by_persona"][persona]["domain_scores"]
            for k, domain in enumerate(domains):
                scores[i, j, k] = ds[domain]["mean_score"]
    return data, models, personas, domains, scores


def default_ipip() -> pd.DataFrame:
    summary = pd.read_csv(RESULTS / "summary.csv")
    sub = summary[
        (summary["persona"] == "Default") & (summary["scale"] == "IPIP-NEO-120")
    ].copy()
    pivot = sub.pivot(index="model_name", columns="domain", values="mean_score")
    return pivot.reindex(columns=IPIP_DOMAINS)


def fig_factor_structure() -> None:
    eig = pd.read_csv(ANALYSIS / "efa_eigenvalues.csv")
    load = pd.read_csv(ANALYSIS / "efa_domain_loadings.csv", index_col=0)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.35),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
        constrained_layout=True,
    )

    x = np.arange(1, len(eig) + 1)
    vals = eig["eigenvalue"].to_numpy()
    ax1.plot(x, vals, marker="o", color=C["blue"], lw=1.8, ms=4.5, zorder=3)
    ax1.axhline(1.0, color=C["red"], ls="--", lw=1, label="Kaiser = 1")
    ax1.fill_between([0.5, 3.5], 1, vals[0] + 0.45, color=C["blue"], alpha=0.08)
    for i, val in enumerate(vals[:3]):
        ax1.annotate(
            f"{val:.2f}",
            (i + 1, val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color=C["blue"],
        )
    ax1.text(
        0.50,
        0.88,
        "3 factors > 1",
        transform=ax1.transAxes,
        fontsize=8,
        color=C["red"],
        fontweight="bold",
        ha="center",
        va="top",
    )
    ax1.set_title("(A) Scree plot")
    ax1.set_xlabel("Factor number")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17])
    ax1.grid(axis="y", alpha=0.8)
    ax1.legend(loc="upper right", frameon=False)

    arr = load.to_numpy()
    im = ax2.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_title("(B) Domain loadings")
    ax2.set_xticks(np.arange(arr.shape[1]))
    ax2.set_xticklabels(load.columns)
    ax2.set_yticks(np.arange(arr.shape[0]))
    ax2.set_yticklabels([short_domain(v) for v in load.index], fontsize=7)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = "white" if abs(arr[i, j]) > 0.58 else C["dark"]
            weight = "bold" if abs(arr[i, j]) > 0.5 else "normal"
            ax2.text(
                j,
                i,
                f"{arr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6.3,
                color=color,
                fontweight=weight,
            )
    cb = fig.colorbar(im, ax=ax2, shrink=0.86, pad=0.02)
    cb.set_label("Loading", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    save(fig, "fig1_factor_structure.png", "fig1_scree_loadings.png")


def fig_cronbach() -> None:
    alpha = pd.read_csv(ANALYSIS / "cronbach_alpha_by_domain.csv")
    stats_df = alpha.groupby("domain")["alpha"].agg(["mean", "std"]).reindex(IPIP_DOMAINS)
    human = pd.Series(
        {
            "Neuroticism": 0.90,
            "Extraversion": 0.89,
            "Openness": 0.87,
            "Agreeableness": 0.88,
            "Conscientiousness": 0.90,
        }
    )
    y = np.arange(len(IPIP_DOMAINS))
    fig, ax = plt.subplots(figsize=(5.4, 3.55), constrained_layout=True)
    ax.barh(y - 0.16, human.reindex(IPIP_DOMAINS), 0.32, color=C["light_gray"], label="Human norm")
    ax.barh(
        y + 0.16,
        stats_df["mean"],
        0.32,
        xerr=stats_df["std"],
        color=C["blue"],
        ecolor=C["gray"],
        capsize=2,
        label="LLM mean +/- SD",
    )
    for yi, val in enumerate(stats_df["mean"]):
        ax.text(val + 0.025, yi + 0.16, f"{val:.2f}", va="center", fontsize=7.5, fontweight="bold")
    ax.axvline(0.70, color=C["red"], ls=":", lw=1, label="0.70 threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(IPIP_DOMAINS)
    ax.set_xlabel("Cronbach alpha")
    ax.set_title("IPIP internal consistency by domain")
    ax.set_xlim(0, 1.14)
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    save(fig, "fig2_cronbach_comparison.png")


def fig_pir_sdr() -> None:
    pir = pd.read_csv(ANALYSIS / "pir_by_model_domain.csv")
    sdr = pd.read_csv(ANALYSIS / "pir_sdr_crossvalidation.csv")
    pir["label"] = pir["scale"].str.replace("-NEO-120", "", regex=False).str[:4] + " " + pir["domain"].map(short_domain)
    order = pir.groupby("label")["pir"].mean().sort_values().index.tolist()

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.0),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )
    data = [pir.loc[pir["label"] == label, "pir"].to_numpy() for label in order]
    ax1.boxplot(
        data,
        vert=False,
        patch_artist=True,
        labels=order,
        medianprops={"color": C["dark"], "linewidth": 1.1},
        boxprops={"facecolor": "#F3F3F3", "edgecolor": C["gray"]},
        whiskerprops={"color": C["gray"]},
        capprops={"color": C["gray"]},
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": C["gray"], "markeredgecolor": C["gray"], "alpha": 0.6},
    )
    means = pir.groupby("label")["pir"].mean().reindex(order)
    ax1.scatter(means, np.arange(1, len(order) + 1), s=16, color=C["blue"], zorder=3, label="Mean")
    ax1.axvline(0.5, color=C["red"], ls=":", lw=1)
    ax1.set_title("(A) PIR by domain")
    ax1.set_xlabel("Pairwise inconsistency rate")
    ax1.set_xlim(0, 1)
    ax1.tick_params(axis="y", labelsize=6.4)
    ax1.grid(axis="x")

    fams = sdr["model"].map(family)
    for fam_name, group in sdr.groupby(fams):
        ax2.scatter(
            group["sdr_composite"],
            group["mean_pir"],
            s=36,
            color=FAMILY_COLORS[fam_name],
            edgecolor="white",
            linewidth=0.6,
            label=fam_name,
            zorder=3,
        )
    corr = stats.spearmanr(sdr["sdr_composite"], sdr["mean_pir"]).correlation
    ax2.text(0.98, 0.96, f"Spearman r = {corr:.2f}", transform=ax2.transAxes, ha="right", va="top", fontsize=8)
    label_idx = sdr["mean_pir"].nlargest(2).index.tolist() + sdr["mean_pir"].nsmallest(2).index.tolist()
    for idx in sorted(set(label_idx)):
        row = sdr.loc[idx]
        offset = (5, -15) if row["mean_pir"] > 0.88 else (5, 4)
        ax2.annotate(short_model(row["model"], 11), (row["sdr_composite"], row["mean_pir"]), xytext=offset, textcoords="offset points", fontsize=6.5)
    ax2.set_title("(B) PIR vs response distortion")
    ax2.set_xlabel("SDR composite")
    ax2.set_ylabel("Mean PIR")
    ax2.grid(True)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=6.3)

    save(fig, "fig2_pir_sdr.png")


def fig_acquiescence() -> None:
    pir = pd.read_csv(ANALYSIS / "pir_by_model_domain.csv")
    acq = pd.read_csv(ANALYSIS / "acquiescence_mechanism.csv")
    domain = (
        pir.groupby(["scale", "domain"])["pir"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )
    domain["label"] = domain["scale"].str[:4] + " " + domain["domain"].map(short_domain)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.95),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
        constrained_layout=True,
    )
    ypos = np.arange(len(domain))
    colors = np.where(domain["mean"] >= 0.5, C["red"], C["blue"])
    ax1.hlines(ypos, 0, domain["mean"], color=colors, alpha=0.65, lw=1.4)
    ax1.errorbar(domain["mean"], ypos, xerr=domain["std"], fmt="none", ecolor=C["gray"], lw=0.7, capsize=2)
    ax1.scatter(domain["mean"], ypos, color=colors, s=23, zorder=3)
    ax1.axvline(0.5, color=C["red"], ls=":", lw=1)
    ax1.axvline(pir["pir"].mean(), color=C["orange"], ls="--", lw=1)
    ax1.text(pir["pir"].mean() + 0.012, len(domain) - 1.3, "overall mean", color=C["orange"], fontsize=7)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(domain["label"], fontsize=6.7)
    ax1.set_xlabel("Pairwise inconsistency rate")
    ax1.set_title("(A) Inconsistency by domain")
    ax1.set_xlim(0, 1)
    ax1.grid(axis="x")

    for fam_name, group in acq.groupby(acq["model"].map(family)):
        ax2.scatter(
            group["fwd_agree_rate"],
            group["rev_agree_rate"],
            s=42,
            color=FAMILY_COLORS[fam_name],
            edgecolor="white",
            linewidth=0.6,
            label=fam_name,
            alpha=0.9,
        )
    lim = [0, 1.02]
    ax2.plot(lim, lim, color=C["gray"], ls=":", lw=1)
    ax2.fill_between(lim, lim, [1.02, 1.02], color=C["red"], alpha=0.05)
    corr = stats.spearmanr(acq["rev_agree_rate"], acq["overall_agree_rate"]).correlation
    ax2.text(0.04, 0.96, f"rev-agree vs PIR: r = {corr:.2f}", transform=ax2.transAxes, va="top", fontsize=8)
    label_idx = acq["overall_agree_rate"].nlargest(3).index.tolist() + acq["overall_agree_rate"].nsmallest(2).index.tolist()
    for idx in sorted(set(label_idx)):
        row = acq.loc[idx]
        name = str(row["model"])
        if row["fwd_agree_rate"] > 0.86 and row["rev_agree_rate"] > 0.55 and "GPT" in name:
            offset = (-52, -14)
            ha = "right"
        elif row["fwd_agree_rate"] > 0.86 and row["rev_agree_rate"] > 0.55 and "Gemini" in name:
            offset = (-52, 10)
            ha = "right"
        elif row["fwd_agree_rate"] > 0.86 and row["rev_agree_rate"] > 0.55:
            offset = (6, -15)
            ha = "left"
        else:
            offset = (4, -11 if row["rev_agree_rate"] > 0.5 else 4)
            ha = "left"
        ax2.annotate(short_model(row["model"], 10), (row["fwd_agree_rate"], row["rev_agree_rate"]), xytext=offset, textcoords="offset points", fontsize=6.3, ha=ha)
    ax2.set_xlabel("Forward-item agree rate")
    ax2.set_ylabel("Reverse-item agree rate")
    ax2.set_title("(B) Acquiescence mechanism")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.grid(True)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=6.2)

    save(fig, "fig3_acquiescence_mechanism.png")


def fig_variance() -> None:
    var = pd.read_csv(ANALYSIS / "variance_decomposition.csv")
    components = ["model", "domain", "persona", "item", "residual"]
    labels = ["Model", "Domain", "Persona", "Item", "Residual"]
    colors = [C["red"], C["blue"], C["green"], C["orange"], "#999999"]
    pivot = (
        var.assign(analysis_short=lambda d: np.where(d["analysis"].str.contains("Likert"), "Likert", "Binary"))
        .pivot(index="analysis_short", columns="component", values="percentage")
        .reindex(["Likert", "Binary"])
        .reindex(columns=components)
    )

    fig, ax = plt.subplots(figsize=(6.4, 2.75), constrained_layout=True)
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for comp, label, color in zip(components, labels, colors):
        vals = pivot[comp].to_numpy()
        ax.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.7, label=label)
        for yi, val, start in zip(y, vals, left):
            if val >= 4.0:
                ax.text(start + val / 2, yi, f"{val:.1f}%", ha="center", va="center", fontsize=7, color="white" if comp in {"model", "domain", "item"} else C["dark"], fontweight="bold")
            elif comp == "model":
                ax.text(start + val + 1.0, yi - 0.22, f"model {val:.2f}%", ha="left", va="center", fontsize=6.7, color=C["red"], fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(["Likert\n(IPIP+SD3)", "Binary\n(ZKPQ+EPQR)"])
    ax.set_xlabel("Variance explained (%)")
    ax.set_title("Sequential variance decomposition")
    ax.set_xlim(0, 100)
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False)
    save(fig, "fig3_variance_decomposition.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.25), constrained_layout=True)
    x = np.arange(len(components))
    w = 0.34
    ax.bar(x - w / 2, pivot.loc["Likert", components], w, color=C["blue"], label="Likert (IPIP+SD3)")
    ax.bar(x + w / 2, pivot.loc["Binary", components], w, color=C["orange"], label="Binary (ZKPQ+EPQR)")
    for xi, comp in enumerate(components):
        for off, row_name in [(-w / 2, "Likert"), (w / 2, "Binary")]:
            val = float(pivot.loc[row_name, comp])
            ax.text(xi + off, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold" if val > 10 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Variance (%)")
    ax.set_ylim(0, 74)
    ax.set_title("Variance components by response format")
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.text(0.02, 0.94, "Model identity explains <1% in both formats", transform=ax.transAxes, fontsize=8, color=C["red"], fontweight="bold", va="top")
    save(fig, "fig4_variance_decomposition.png")


def fig_measurement_invariance() -> None:
    inv = pd.read_csv(ANALYSIS / "persona_invariance.csv")
    pivot = inv.pivot_table(index="model", columns="persona", values="pearson_r", aggfunc="mean")
    persona_order = pivot.mean(axis=0).sort_values(ascending=False).index.tolist()
    model_order = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[model_order, persona_order]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.25),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )
    im = ax1.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0, vmax=0.85, aspect="auto")
    ax1.set_xticks(np.arange(len(persona_order)))
    ax1.set_xticklabels(persona_order, rotation=45, ha="right", fontsize=6.4)
    ax1.set_yticks(np.arange(len(model_order)))
    ax1.set_yticklabels([short_model(m, 13) for m in model_order], fontsize=6.4)
    ax1.set_title("(A) Default vs MBTI profile correlation")
    cb = fig.colorbar(im, ax=ax1, shrink=0.74, pad=0.02)
    cb.set_label("Pearson r", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    means = inv.groupby("persona")["pearson_r"].agg(["mean", "std"]).reindex(persona_order)
    y = np.arange(len(means))
    bar_colors = plt.cm.RdYlGn((means["mean"] - means["mean"].min()) / (means["mean"].max() - means["mean"].min() + 1e-9))
    ax2.barh(y, means["mean"], xerr=means["std"], color=bar_colors, ecolor=C["gray"], capsize=2)
    ax2.axvline(0.3, color=C["orange"], ls=":", lw=1)
    ax2.axvline(0.8, color=C["green"], ls=":", lw=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(means.index, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 0.85)
    ax2.set_xlabel("Mean r with Default")
    ax2.set_title("(B) Persona ranking")
    ax2.grid(axis="x")
    save(fig, "fig4_measurement_invariance.png")


def fig_response_styles() -> None:
    styles = pd.read_csv(ANALYSIS / "response_styles.csv")
    default = styles[styles["persona"] == "Default"].copy()
    pir = pd.read_csv(ANALYSIS / "pir_sdr_crossvalidation.csv")
    lie = pd.read_csv(ANALYSIS / "sdr_by_model.csv")[["model", "lie_scale"]]
    merged = default.merge(pir, on="model", how="left").merge(lie, on="model", how="left")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.75), constrained_layout=True)
    for fam_name, group in merged.groupby(merged["model"].map(family)):
        ax1.scatter(group["acquiescence"], group["midpoint_response"], s=42, color=FAMILY_COLORS[fam_name], edgecolor="white", linewidth=0.6, label=fam_name)
        ax2.scatter(group["mean_pir"], group["lie_scale"], s=42, color=FAMILY_COLORS[fam_name], edgecolor="white", linewidth=0.6, label=fam_name)
    for idx in merged["acquiescence"].nlargest(2).index.tolist() + merged["mean_pir"].nlargest(2).index.tolist():
        row = merged.loc[idx]
        ax1.annotate(short_model(row["model"], 10), (row["acquiescence"], row["midpoint_response"]), xytext=(4, 3), textcoords="offset points", fontsize=6.2)
        ax2.annotate(short_model(row["model"], 10), (row["mean_pir"], row["lie_scale"]), xytext=(4, 3), textcoords="offset points", fontsize=6.2)
    ax1.axvline(0, color=C["gray"], ls=":", lw=1)
    ax1.set_xlabel("Acquiescence bias")
    ax1.set_ylabel("Midpoint response rate")
    ax1.set_title("(A) Response style map")
    ax1.grid(True)
    ax2.set_xlabel("Mean PIR")
    ax2.set_ylabel("Lie-scale score")
    ax2.set_title("(B) Inconsistency vs social desirability")
    ax2.grid(True)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=6.2)
    save(fig, "fig5_response_styles.png")


def fig_convergent_validity() -> None:
    conv = pd.read_csv(ANALYSIS / "convergent_validity_enhanced.csv")
    r_vals = conv["r_spearman"].astype(float).to_numpy()
    n_obs = int(conv["n_models"].iloc[0])
    se = 1 / math.sqrt(n_obs - 3)
    z = np.arctanh(np.clip(r_vals, -0.999, 0.999))
    lo = np.tanh(z - 1.96 * se)
    hi = np.tanh(z + 1.96 * se)

    labels = [p.replace(" <-> ", "\n<-> ").replace("Neuroticism-Anxiety", "N-Anxiety") for p in conv["pair"]]
    y = np.arange(len(conv))
    fig, ax = plt.subplots(figsize=(7.2, 3.45), constrained_layout=True)
    for i, row in conv.iterrows():
        color = C["green"] if bool(row["sign_match"]) else C["red"]
        alpha = 1.0 if float(row["p_spearman"]) < 0.05 else 0.55
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=color, lw=2, alpha=alpha)
        ax.scatter(r_vals[i], y[i], s=40, facecolor=color if row["p_spearman"] < 0.05 else "white", edgecolor=color, lw=1.2, zorder=3, alpha=alpha)
        mark = "*" if row["p_spearman"] < 0.05 else "ns"
        ax.text(hi[i] + 0.035, y[i], f"r={r_vals[i]:.2f} {mark}", va="center", fontsize=7)
    bad = conv.index[~conv["sign_match"].astype(bool)].tolist()
    for idx in bad:
        ax.annotate("sign reversal", (r_vals[idx], y[idx]), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=6.5, color=C["red"], fontweight="bold")
    ax.axvline(0, color=C["dark"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Spearman correlation with 95% CI")
    ax.set_title("Cross-scale convergent validity")
    ax.set_xlim(-1.0, 1.1)
    ax.grid(axis="x")
    ax.text(0.02, 0.03, "* p < .05; ns = not significant; green = expected sign", transform=ax.transAxes, fontsize=6.7, color=C["gray"])
    save(fig, "fig5_convergent_validity.png", "fig6_convergent_validity.png")


def fig_invariance_robustness() -> None:
    inv = pd.read_csv(ANALYSIS / "persona_invariance.csv")
    loo_model = pd.read_csv(ANALYSIS / "leave_one_out_robustness.csv")
    loo_persona = pd.read_csv(ANALYSIS / "leave_one_persona_out.csv")

    means = inv.groupby("persona")["pearson_r"].agg(["mean", "std"]).sort_values("mean")
    y = np.arange(len(means))
    norm = (means["mean"] - means["mean"].min()) / (means["mean"].max() - means["mean"].min() + 1e-9)
    colors = plt.cm.RdYlGn(0.18 + 0.68 * norm)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.9),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )
    ax1.barh(y, means["mean"], xerr=means["std"], color=colors, ecolor=C["gray"], capsize=2)
    ax1.axvline(0.3, color=C["orange"], ls=":", lw=1, label="Weak r=0.3")
    ax1.axvline(0.8, color=C["green"], ls=":", lw=1, label="Strong r=0.8")
    ax1.set_yticks(y)
    ax1.set_yticklabels(means.index)
    ax1.set_xlim(0, 0.85)
    ax1.set_xlabel("Mean Pearson r with Default")
    ax1.set_title("(A) Persona invariance")
    ax1.grid(axis="x")
    ax1.legend(loc="lower right", frameon=False)
    best = means.iloc[-1]
    worst = means.iloc[0]
    ax1.text(best["mean"] + 0.015, len(means) - 1, f"{means.index[-1]} {best['mean']:.2f}", va="center", fontsize=7.2, color=C["green"], fontweight="bold")
    ax1.text(worst["mean"] + 0.015, 0, f"{means.index[0]} {worst['mean']:.2f}", va="center", fontsize=7.2, color=C["red"], fontweight="bold")

    model_eigs = loo_model["first_eigenvalue"].to_numpy()
    persona_eigs = loo_persona["first_eigenvalue"].to_numpy()
    bars = ax2.barh([1, 0], [model_eigs.mean(), persona_eigs.mean()], xerr=[model_eigs.std(), persona_eigs.std()], color=[C["blue"], C["green"]], alpha=0.86, ecolor=C["gray"], capsize=3)
    ax2.axvline(1.0, color=C["red"], ls="--", lw=1, label="Kaiser = 1")
    ax2.set_yticks([1, 0])
    ax2.set_yticklabels([f"Model LOO\n(n={len(model_eigs)})", f"Persona LOO\n(n={len(persona_eigs)})"])
    ax2.set_xlabel("First eigenvalue")
    ax2.set_title("(B) Robustness")
    ax2.set_xlim(0, max(model_eigs.mean(), persona_eigs.mean()) + 1.2)
    ax2.grid(axis="x")
    ax2.legend(loc="lower right", frameon=False)
    for bar, vals in zip(bars, [model_eigs, persona_eigs]):
        ax2.text(bar.get_width() - 0.08, bar.get_y() + bar.get_height() / 2, f"{vals.mean():.2f} +/- {vals.std():.2f}\n3 factors in all runs", ha="right", va="center", fontsize=7, color="white", fontweight="bold")
    save(fig, "fig6_invariance_robustness.png")


def fig_factor_collapse() -> None:
    congr = pd.read_csv(ANALYSIS / "item_level_congruence.csv", index_col=0)
    eig = pd.read_csv(ANALYSIS / "efa_eigenvalues.csv")
    max_abs = congr.abs().max(axis=0).reindex(IPIP_DOMAINS)

    vals = eig["eigenvalue"].to_numpy()
    cum = vals.cumsum() / vals.sum() * 100
    x = np.arange(1, len(vals) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    ax1.bar(np.arange(len(IPIP_DOMAINS)), max_abs, color=C["blue"], alpha=0.78, label="Best observed congruence")
    ax1.axhline(0.85, color=C["orange"], ls=":", lw=1, label="Good recovery")
    ax1.axhline(0.95, color=C["red"], ls="--", lw=1, label="Excellent recovery")
    for i, val in enumerate(max_abs):
        ax1.text(i, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax1.set_xticks(np.arange(len(IPIP_DOMAINS)))
    ax1.set_xticklabels([d[:5] for d in IPIP_DOMAINS], rotation=0)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Tucker congruence")
    ax1.set_title("(A) Big Five recovery is weak")
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", frameon=False, fontsize=6.7)

    ax2.plot(x, cum, color=C["blue"], marker="o", lw=1.7, ms=3.5)
    ax2.axhline(80, color=C["red"], ls=":", lw=1)
    ax2.axvline(3, color=C["orange"], ls="--", lw=1)
    ax2.scatter([3], [cum[2]], color=C["red"], zorder=3)
    ax2.text(3.25, cum[2], f"{cum[2]:.1f}% at 3 factors", fontsize=7.2, va="center", color=C["red"], fontweight="bold")
    ax2.set_xlabel("Number of factors")
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_title("(B) Collapsed factor structure")
    ax2.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17])
    ax2.set_ylim(35, 102)
    ax2.grid(True)
    save(fig, "fig7_factor_collapse.png")


def fig_model_dashboard() -> None:
    pir = pd.read_csv(ANALYSIS / "pir_sdr_crossvalidation.csv").set_index("model")
    sdr = pd.read_csv(ANALYSIS / "sdr_by_model.csv").set_index("model")
    inv = pd.read_csv(ANALYSIS / "persona_invariance.csv").groupby("model")["pearson_r"].mean()
    df = pd.DataFrame(
        {
            "PIR": pir["mean_pir"],
            "SDR": sdr["sdr_composite"],
            "Persona r": inv,
        }
    ).dropna()
    order_pir = df["PIR"].sort_values().index
    order_sdr = df["SDR"].sort_values().index
    order_inv = df["Persona r"].sort_values().index

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    for ax, metric, order, color, title in [
        (axs[0, 0], "PIR", order_pir, C["orange"], "(A) Inconsistency ranking"),
        (axs[0, 1], "SDR", order_sdr, C["green"], "(B) Social desirability ranking"),
        (axs[1, 0], "Persona r", order_inv, C["blue"], "(C) Persona invariance"),
    ]:
        y = np.arange(len(order))
        vals = df.loc[order, metric]
        ax.barh(y, vals, color=color, alpha=0.82)
        ax.set_yticks(y)
        ax.set_yticklabels([short_model(m, 12) for m in order], fontsize=5.7)
        ax.set_title(title)
        ax.grid(axis="x")
        ax.tick_params(axis="x", labelsize=7)
    axs[0, 0].set_xlabel("Mean PIR")
    axs[0, 1].set_xlabel("SDR composite")
    axs[1, 0].set_xlabel("Mean r with Default")

    axs[1, 1].scatter(df["PIR"], df["Persona r"], s=42, c=[FAMILY_COLORS[family(m)] for m in df.index], edgecolor="white", linewidth=0.6)
    for idx in df["PIR"].nlargest(2).index.tolist() + df["Persona r"].nlargest(1).index.tolist() + df["Persona r"].nsmallest(1).index.tolist():
        axs[1, 1].annotate(short_model(idx, 10), (df.loc[idx, "PIR"], df.loc[idx, "Persona r"]), xytext=(4, 3), textcoords="offset points", fontsize=6)
    axs[1, 1].set_xlabel("Mean PIR")
    axs[1, 1].set_ylabel("Mean persona invariance")
    axs[1, 1].set_title("(D) PIR vs persona invariance")
    axs[1, 1].grid(True)
    save(fig, "fig8_model_dashboard.png")


def fig_default_heatmap_and_radar() -> None:
    mat = default_ipip()
    fam_order = sorted(mat.index, key=lambda m: (family(m), short_model(m)))
    mat = mat.loc[fam_order]
    z = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0)

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(IPIP_DOMAINS)))
    ax.set_xticklabels(["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"], rotation=18, ha="right")
    ax.set_yticks(np.arange(len(mat)))
    ax.set_yticklabels([short_model(m, 14) for m in mat.index], fontsize=7.4)
    for tick, model in zip(ax.get_yticklabels(), mat.index):
        tick.set_color(FAMILY_COLORS[family(model)])
        tick.set_fontweight("bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            color = "white" if abs(z.iloc[i, j]) > 1.15 else C["dark"]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.2, color=color)
    ax.set_title("Default persona: IPIP-NEO-120 domain scores")
    cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("z-score across models; cells show raw 1-5 scores")
    cb.ax.tick_params(labelsize=7)
    # Family separators.
    prev = None
    for i, model in enumerate(mat.index):
        fam = family(model)
        if prev is not None and fam != prev:
            ax.axhline(i - 0.5, color=C["dark"], lw=0.7)
        prev = fam
    save(fig, "fig_heatmap_default.png")

    angles = np.linspace(0, 2 * np.pi, len(IPIP_DOMAINS), endpoint=False).tolist()
    closed_angles = angles + angles[:1]
    fig = plt.figure(figsize=(6.2, 5.15), constrained_layout=True)
    ax = fig.add_subplot(111, polar=True)
    for _, row in mat.iterrows():
        vals = row.to_numpy().tolist() + [row.iloc[0]]
        ax.plot(closed_angles, vals, color="#BBBBBB", alpha=0.35, lw=0.8)
    for fam_name, group_idx in mat.groupby(mat.index.map(family)).groups.items():
        fam_mean = mat.loc[list(group_idx)].mean(axis=0)
        vals = fam_mean.to_numpy().tolist() + [fam_mean.iloc[0]]
        ax.plot(closed_angles, vals, color=FAMILY_COLORS[fam_name], lw=2.0, label=fam_name)
        ax.fill(closed_angles, vals, color=FAMILY_COLORS[fam_name], alpha=0.045)
    ax.set_xticks(angles)
    ax.set_xticklabels(IPIP_DOMAINS, fontsize=8.5)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7, color=C["gray"])
    ax.grid(color="#CFCFCF", lw=0.7)
    ax.set_title("Default persona: IPIP-NEO-120 family profiles", pad=18, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=True, fontsize=7.2)
    save(fig, "fig_radar_default.png")


def cross_cutting_data() -> tuple[list[str], list[str], list[str], np.ndarray, dict[str, list[str]]]:
    _, models, personas, domains, scores = load_json_results()
    scales = ["IPIP-NEO-120", "SD3", "ZKPQ-50-CC", "EPQR-A"]
    scale_domains = {s: [d for d in domains if d.startswith(s)] for s in scales}
    return models, personas, domains, scores, scale_domains


def fig_model_clustering(models: list[str], personas: list[str], domains: list[str], scores: np.ndarray) -> None:
    default_idx = personas.index("Default")
    default_profiles = scores[:, default_idx, :]
    default_z = (default_profiles - default_profiles.mean(axis=0, keepdims=True)) / (default_profiles.std(axis=0, keepdims=True) + 1e-9)
    z_link = linkage(default_z, method="ward")

    fig, ax = plt.subplots(figsize=(6.8, 4.9), constrained_layout=True)
    dendrogram(z_link, labels=[short_model(m, 14) for m in models], orientation="right", leaf_font_size=7.2, ax=ax, color_threshold=0)
    for lbl in ax.get_yticklabels():
        text = lbl.get_text()
        for model in models:
            if short_model(model, 14) == text:
                lbl.set_color(FAMILY_COLORS[family(model)])
                lbl.set_fontweight("bold")
                break
    ax.set_title("Model clustering by psychometric profile")
    ax.set_xlabel("Ward linkage distance")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS[f], markersize=6, label=f)
        for f in sorted({family(m) for m in models})
    ]
    ax.legend(handles=handles, loc="lower right", ncol=2, title="Family", frameon=True, fontsize=6.5, title_fontsize=7)
    save(fig, "fig7_model_clustering.png")


def ordered_domain_labels(domains: list[str], scale_domains: dict[str, list[str]]) -> tuple[list[str], list[int], list[str]]:
    scale_order = ["IPIP-NEO-120", "SD3", "ZKPQ-50-CC", "EPQR-A"]
    ordered_domains: list[str] = []
    for scale in scale_order:
        ordered_domains.extend(scale_domains[scale])
    order_idx = [domains.index(d) for d in ordered_domains]
    labels = [short_domain(d).replace(" ", "\n", 1) for d in ordered_domains]
    return ordered_domains, order_idx, labels


def fig_mbti_effects(models: list[str], personas: list[str], domains: list[str], scores: np.ndarray, scale_domains: dict[str, list[str]]) -> None:
    axes = {
        "E/I": ("E", [p for p in personas if p.startswith("E")], "I", [p for p in personas if p.startswith("I")]),
        "S/N": ("S", [p for p in personas if len(p) == 4 and p[1] == "S"], "N", [p for p in personas if len(p) == 4 and p[1] == "N"]),
        "T/F": ("T", [p for p in personas if len(p) == 4 and p[2] == "T"], "F", [p for p in personas if len(p) == 4 and p[2] == "F"]),
        "J/P": ("J", [p for p in personas if len(p) == 4 and p[3] == "J"], "P", [p for p in personas if len(p) == 4 and p[3] == "P"]),
    }
    effects = np.zeros((len(axes), len(domains)))
    axis_names = list(axes)
    for i, (_axis, (_g1, p1, _g2, p2)) in enumerate(axes.items()):
        idx1 = [personas.index(p) for p in p1 if p != "Default"]
        idx2 = [personas.index(p) for p in p2 if p != "Default"]
        for k in range(len(domains)):
            a = scores[:, idx1, k].ravel()
            b = scores[:, idx2, k].ravel()
            pooled = math.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
            effects[i, k] = (a.mean() - b.mean()) / pooled if pooled > 0 else 0

    ordered_domains, order_idx, labels = ordered_domain_labels(domains, scale_domains)
    arr = effects[:, order_idx]
    vmax = np.abs(arr).max()

    fig, ax = plt.subplots(figsize=(7.5, 3.65), constrained_layout=True)
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=6.3)
    ax.set_yticks(np.arange(len(axis_names)))
    ax.set_yticklabels(axis_names, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            color = "white" if abs(val) > vmax * 0.52 else C["dark"]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=5.6, color=color)
    pos = 0
    for scale in ["IPIP-NEO-120", "SD3", "ZKPQ-50-CC", "EPQR-A"]:
        n = len(scale_domains[scale])
        if pos > 0:
            ax.axvline(pos - 0.5, color=C["dark"], lw=0.7)
        pos += n
    ax.set_title("MBTI dimension effects on domain scores (Cohen's d)", pad=16)
    cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("Cohen's d")
    cb.ax.tick_params(labelsize=7)
    save(fig, "fig8_mbti_dimension_effects.png")


def fig_persona_sensitivity(models: list[str], personas: list[str], domains: list[str], scores: np.ndarray, scale_domains: dict[str, list[str]]) -> None:
    sensitivity = scores.std(axis=1).mean(axis=0)
    ordered_domains, order_idx, labels = ordered_domain_labels(domains, scale_domains)
    ordered_vals = sensitivity[order_idx]
    scale_order = ["IPIP-NEO-120", "SD3", "ZKPQ-50-CC", "EPQR-A"]
    colors = [SCALE_COLORS[d.split("::")[0]] for d in ordered_domains]

    ranges = {"IPIP-NEO-120": 4, "SD3": 4, "ZKPQ-50-CC": 1, "EPQR-A": 1}
    norm = {}
    for scale in scale_order:
        idx = [domains.index(d) for d in scale_domains[scale]]
        norm[scale] = float(np.mean(sensitivity[idx] / ranges[scale]))
    format_vals = [
        np.mean([norm["IPIP-NEO-120"], norm["SD3"]]),
        np.mean([norm["ZKPQ-50-CC"], norm["EPQR-A"]]),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.35), gridspec_kw={"width_ratios": [1.55, 0.85]}, constrained_layout=True)
    y = np.arange(len(ordered_domains))
    ax1.barh(y, ordered_vals, color=colors, edgecolor="white", linewidth=0.4)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=6.1)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean SD across personas")
    ax1.set_title("(A) Persona sensitivity by domain")
    ax1.grid(axis="x")
    for scale in scale_order:
        handles = [mpatches.Patch(color=SCALE_COLORS[s], label=s.split("-")[0]) for s in scale_order]
    ax1.legend(handles=handles, loc="lower right", frameon=False, ncol=2, fontsize=6)

    ax2.bar([0, 1], format_vals, color=[SCALE_COLORS["IPIP-NEO-120"], SCALE_COLORS["ZKPQ-50-CC"]], width=0.55)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Likert\n(IPIP+SD3)", "Binary\n(ZKPQ+EPQR)"])
    ax2.set_ylabel("Normalized sensitivity")
    ax2.set_title("(B) Scale format")
    ax2.grid(axis="y")
    for i, val in enumerate(format_vals):
        ax2.text(i, val + 0.008, f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    save(fig, "fig9_persona_sensitivity.png")


def fig_within_family(models: list[str], personas: list[str], domains: list[str], scores: np.ndarray) -> None:
    default_idx = personas.index("Default")
    default_profiles = scores[:, default_idx, :]
    default_z = (default_profiles - default_profiles.mean(axis=0, keepdims=True)) / (default_profiles.std(axis=0, keepdims=True) + 1e-9)
    vendor_models: dict[str, list[int]] = defaultdict(list)
    for i, model in enumerate(models):
        vendor_models[family(model)].append(i)
    multi = [v for v in sorted(vendor_models) if len(vendor_models[v]) > 1]
    within_vals = [float(pdist(default_z[vendor_models[v]], "euclidean").mean()) for v in multi]
    cross_vals = []
    for i, a in enumerate(multi):
        for b in multi[i + 1 :]:
            cross_vals.append(float(cdist(default_z[vendor_models[a]], default_z[vendor_models[b]], "euclidean").mean()))
    cross_mean = float(np.mean(cross_vals))

    centered = default_z - default_z.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    var_ratio = (s**2) / np.sum(s**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.95), gridspec_kw={"width_ratios": [1.05, 1.0]}, constrained_layout=True)
    labels = [f"{v}\n({len(vendor_models[v])} models)" for v in multi] + ["Cross-family\n(avg)"]
    vals = within_vals + [cross_mean]
    cols = [FAMILY_COLORS[v] for v in multi] + [C["gray"]]
    y = np.arange(len(vals))
    ax1.barh(y, vals, color=cols, alpha=0.86)
    ax1.axvline(cross_mean, color=C["gray"], ls="--", lw=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean Euclidean distance")
    ax1.set_title("(A) Within vs cross-family")
    ax1.grid(axis="x")
    for yi, val in enumerate(vals):
        ax1.text(val + 0.08, yi, f"{val:.2f}", va="center", fontsize=7)

    for i, model in enumerate(models):
        fam = family(model)
        ax2.scatter(coords[i, 0], coords[i, 1], s=42, color=FAMILY_COLORS[fam], edgecolor="white", linewidth=0.6, zorder=3)
        ax2.annotate(short_model(model, 9), (coords[i, 0], coords[i, 1]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=5.6, color=FAMILY_COLORS[fam])
    for fam in multi:
        pts = coords[vendor_models[fam]]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax2.plot(pts[simplex, 0], pts[simplex, 1], color=FAMILY_COLORS[fam], alpha=0.28, lw=1)
            except Exception:
                pass
        elif len(pts) == 2:
            ax2.plot(pts[:, 0], pts[:, 1], color=FAMILY_COLORS[fam], alpha=0.3, lw=1, ls="--")
    ax2.axhline(0, color="#EEEEEE", lw=0.8)
    ax2.axvline(0, color="#EEEEEE", lw=0.8)
    ax2.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}% var.)")
    ax2.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}% var.)")
    ax2.set_title("(B) Psychometric space")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS[f], markersize=5, label=f)
        for f in sorted({family(m) for m in models})
    ]
    ax2.legend(handles=handles, loc="lower left", ncol=2, frameon=True, fontsize=5.6)
    save(fig, "fig10_within_family_divergence.png")


def sync_outputs() -> None:
    workspace_figures = [
        "fig1_scree_loadings.png",
        "fig2_pir_sdr.png",
        "fig3_variance_decomposition.png",
        "fig4_measurement_invariance.png",
        "fig5_response_styles.png",
        "fig6_convergent_validity.png",
        "fig7_factor_collapse.png",
        "fig8_model_dashboard.png",
    ]
    draft_figures = [
        "fig1_factor_structure.png",
        "fig1_scree_loadings.png",
        "fig2_cronbach_comparison.png",
        "fig2_pir_sdr.png",
        "fig3_acquiescence_mechanism.png",
        "fig3_variance_decomposition.png",
        "fig4_measurement_invariance.png",
        "fig4_variance_decomposition.png",
        "fig5_convergent_validity.png",
        "fig5_response_styles.png",
        "fig6_convergent_validity.png",
        "fig6_invariance_robustness.png",
        "fig7_factor_collapse.png",
        "fig8_model_dashboard.png",
    ]
    final_figures = [
        "fig3_acquiescence_mechanism.png",
        "fig4_variance_decomposition.png",
        "fig5_convergent_validity.png",
        "fig6_invariance_robustness.png",
        "fig7_model_clustering.png",
        "fig8_mbti_dimension_effects.png",
        "fig9_persona_sensitivity.png",
        "fig10_within_family_divergence.png",
        "fig_heatmap_default.png",
        "fig_radar_default.png",
    ]
    for name in workspace_figures:
        for target_dir in [WORKSPACE / "figures", WORKSPACE / "inputs" / "figures"]:
            shutil.copy2(FIGURES / name, target_dir / name)
    for name in draft_figures:
        shutil.copy2(FIGURES / name, WORKSPACE / "drafts" / name)
    for name in final_figures:
        shutil.copy2(FIGURES / name, WORKSPACE / "final" / name)
    print("synchronized duplicate figure directories")


def main() -> None:
    fig_factor_structure()
    fig_cronbach()
    fig_pir_sdr()
    fig_acquiescence()
    fig_variance()
    fig_measurement_invariance()
    fig_response_styles()
    fig_convergent_validity()
    fig_invariance_robustness()
    fig_factor_collapse()
    fig_model_dashboard()
    fig_default_heatmap_and_radar()

    models, personas, domains, scores, scale_domains = cross_cutting_data()
    fig_model_clustering(models, personas, domains, scores)
    fig_mbti_effects(models, personas, domains, scores, scale_domains)
    fig_persona_sensitivity(models, personas, domains, scores, scale_domains)
    fig_within_family(models, personas, domains, scores)
    sync_outputs()


if __name__ == "__main__":
    main()
