#!/usr/bin/env python3
"""
Generate corrected-data figures for the rewritten paper.

Produces:
  - figures/fig_corr_corrected.pdf       Inter-dimension correlation matrix (BFI 5x5)
  - figures/fig_radar_corrected.pdf      Radar plot of family-level Big Five profiles
  - figures/fig_mbti_pfi.pdf             Per-model Persona Fidelity Index bar chart
  - figures/fig_mbti_covfid.pdf          Per-model Covariance Fidelity bar chart
  - figures/fig_e_axis_inversion.pdf     Scatter of LLM vs human inter-dim correlations
  - figures/fig_response_styles_corrected.pdf  Response habit indicators (corrected)

All figures pull from results/corrected_analysis/*.csv produced by analyze_corrected.py
or directly from results/vendor_exp/corrected/.
"""

import json
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_DIR = Path("figures")
PAPER_FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = Path("results/corrected_analysis")

BFI_LABELS = ["E", "A", "C", "N", "O"]
HUMAN_BFI2 = {
    ("E", "A"): 0.18, ("E", "C"): 0.11, ("E", "N"): -0.34, ("E", "O"): 0.22,
    ("A", "C"): 0.29, ("A", "N"): -0.25, ("A", "O"): 0.19,
    ("C", "N"): -0.30, ("C", "O"): 0.10,
    ("N", "O"): -0.08,
}


def save(fig, name):
    fig.savefig(FIG_DIR / name)
    fig.savefig(PAPER_FIG_DIR / name)
    plt.close(fig)
    print(f"  -> {name}")


def fig_corr_corrected():
    R = pd.read_csv(ANALYSIS_DIR / "corr_matrix_corrected.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(R.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(5))
    ax.set_xticklabels(R.columns)
    ax.set_yticks(range(5))
    ax.set_yticklabels(R.index)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{R.iloc[i, j]:+.2f}",
                    ha="center", va="center",
                    color="white" if abs(R.iloc[i, j]) > 0.5 else "black",
                    fontsize=7.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("LLM Big Five inter-dimension r (n=43, corrected)")
    save(fig, "fig_corr_corrected.pdf")


def fig_radar_corrected():
    means = pd.read_csv(ANALYSIS_DIR / "model_means_corrected.csv", index_col=0)
    means.columns = [c.replace("bfi.", "").capitalize()[:6] for c in means.columns]
    angles = np.linspace(0, 2 * np.pi, len(means.columns), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(3.4, 3.4), subplot_kw={"projection": "polar"})
    cmap = plt.cm.tab20(np.linspace(0, 1, len(means)))
    for idx, (m, row) in enumerate(means.iterrows()):
        vals = list(row.values) + [row.values[0]]
        ax.plot(angles, vals, color=cmap[idx], linewidth=0.6, alpha=0.55)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(means.columns)
    ax.set_ylim(1, 5)
    ax.set_yticks([2, 3, 4])
    ax.set_yticklabels(["2", "3", "4"], color="grey")
    ax.set_title(f"Big Five profiles (n = {len(means)} models, corrected)")
    save(fig, "fig_radar_corrected.pdf")


def fig_mbti_pfi():
    p = ANALYSIS_DIR / "mbti_pfi.csv"
    if not p.exists() or p.stat().st_size == 0:
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    df = df.sort_values("PFI", ascending=False)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.barh(df["model_id"], df["PFI"], color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Persona Fidelity Index")
    ax.set_xlim(-1, 1)
    ax.set_title("PFI per model (16 MBTI personas)")
    save(fig, "fig_mbti_pfi.pdf")


def fig_mbti_covfid():
    p = ANALYSIS_DIR / "mbti_covfid.csv"
    if not p.exists() or p.stat().st_size == 0:
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    df = df.sort_values("CovFid", ascending=False)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.barh(df["model_id"], df["CovFid"], color="darkgreen", edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Covariance Fidelity")
    ax.set_xlim(-1, 1)
    ax.set_title("Persona-induced covariance vs human BFI-2")
    save(fig, "fig_mbti_covfid.pdf")


def fig_response_styles_corrected():
    """Three-panel figure: midpoint preference, extreme answering, and corrected
    agreement bias for the 15-model paper cohort under corrected scoring."""
    PAPER15 = {
        "Qwen/Qwen3.5-397B-A17B": "Qwen", "Pro/deepseek-ai/DeepSeek-V3.2": "DeepSeek",
        "Pro/zai-org/GLM-5": "GLM", "Pro/moonshotai/Kimi-K2.5": "Kimi",
        "baidu/ERNIE-4.5-300B-A47B": "ERNIE", "tencent/Hunyuan-A13B-Instruct": "Hunyuan",
        "ByteDance-Seed/Seed-OSS-36B-Instruct": "Seed", "internlm/internlm2_5-7b-chat": "InternLM",
        "inclusionAI/Ring-flash-2.0": "Ring", "stepfun-ai/Step-3.5-Flash": "Step",
        "ascend-tribe/pangu-pro-moe": "Pangu", "Kwaipilot/KAT-Dev": "KAT",
        "Pro/MiniMaxAI/MiniMax-M2.5": "MiniMax", "gpt-5": "GPT", "claude-opus-4-5-20251101": "Claude",
    }
    REVERSE = {"extraversion": [1, 3, 5, 7], "agreeableness": [3, 4, 6, 7],
               "conscientiousness": [1, 3, 5, 7], "neuroticism": [0], "openness": [1, 3, 5]}
    fam_data = {}
    for f in glob.glob("results/vendor_exp/corrected/*.json"):
        if "checkpoint" in f:
            continue
        try:
            recs = json.loads(open(f).read())
        except Exception:
            continue
        for r in recs:
            if r.get("study") != 1:
                continue
            if r.get("thinking_mode", "chat") != "chat":
                continue
            if r.get("prompt_variant") and r.get("prompt_variant") != "default":
                continue
            fam = PAPER15.get(r["model_id"])
            if not fam:
                continue
            items = r.get("items", {})
            all_raw, raw_pos, raw_neg = [], [], []
            for trait, revs in REVERSE.items():
                key = f"bfi_{trait}"
                if key not in items:
                    continue
                for i, v in enumerate(items[key]):
                    if i in revs:
                        all_raw.append(6 - v)
                        raw_neg.append(6 - v)
                    else:
                        all_raw.append(v)
                        raw_pos.append(v)
            if not all_raw or not raw_pos or not raw_neg:
                continue
            fam_data.setdefault(fam, []).append({
                "midpoint": sum(1 for x in all_raw if x == 3) / len(all_raw),
                "extreme":  sum(1 for x in all_raw if x in (1, 5)) / len(all_raw),
                "agree_bias": np.mean(raw_pos) - np.mean(raw_neg),
            })

    fams = sorted(fam_data, key=lambda f: -np.mean([d["midpoint"] for d in fam_data[f]]))
    mid = [np.mean([d["midpoint"] for d in fam_data[f]]) * 100 for f in fams]
    ext = [np.mean([d["extreme"] for d in fam_data[f]]) * 100 for f in fams]
    bias = [np.mean([d["agree_bias"] for d in fam_data[f]]) for f in fams]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), sharey=False)
    axes[0].barh(fams, mid, color="steelblue", edgecolor="black", linewidth=0.4)
    axes[0].set_xlabel("Midpoint preference (%)")
    axes[0].set_title("(a) Picks the middle (3)")
    axes[1].barh(fams, ext, color="orange", edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Extreme answering (%)")
    axes[1].set_title("(b) Picks 1 or 5")
    axes[1].set_yticklabels([])
    axes[2].barh(fams, bias, color="green", edgecolor="black", linewidth=0.4)
    axes[2].axvline(0, color="black", linewidth=0.5)
    axes[2].set_xlabel("Agreement bias (positive minus negative items)")
    axes[2].set_title("(c) Acquiescence")
    axes[2].set_yticklabels([])
    fig.tight_layout()
    save(fig, "fig_response_styles_corrected.pdf")


def main():
    print("Generating corrected figures ...")
    if not (ANALYSIS_DIR / "corr_matrix_corrected.csv").exists():
        print("Run analyze_corrected.py first.")
        return
    fig_corr_corrected()
    fig_radar_corrected()
    fig_mbti_pfi()
    fig_mbti_covfid()
    fig_e_axis_inversion()
    fig_response_styles_corrected()
    print("Done.")


def fig_e_axis_inversion():
    R = pd.read_csv(ANALYSIS_DIR / "corr_matrix_corrected.csv", index_col=0)
    pairs = list(HUMAN_BFI2.keys())
    human = np.array([HUMAN_BFI2[p] for p in pairs])
    llm = np.array([R.loc[a, b] for a, b in pairs])
    colors = ["crimson" if "E" in p else "steelblue" for p in pairs]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)
    ax.plot([-0.5, 0.8], [-0.5, 0.8], "--", color="grey", linewidth=0.5, label="agreement")
    for (a, b), x, y, c in zip(pairs, human, llm, colors):
        ax.scatter(x, y, c=c, s=40, edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(f"{a}-{b}", (x, y), xytext=(3, 3),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("Human BFI-2 r (Soto and John 2017)")
    ax.set_ylabel("LLM r (n = 43, corrected)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.6, 0.8)
    ax.set_title("LLM vs human inter-dimension correlations")
    legend_h = [plt.scatter([], [], c="crimson", s=30, label="involves E"),
                plt.scatter([], [], c="steelblue", s=30, label="other")]
    ax.legend(handles=legend_h, loc="upper left", frameon=False)
    save(fig, "fig_e_axis_inversion.pdf")


if __name__ == "__main__":
    main()
