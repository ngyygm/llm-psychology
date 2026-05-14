"""
Generate publication-quality figures for the psychometric validation analysis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis_output")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_all_results():
    results = {}
    for fpath in sorted(RESULTS_DIR.glob("exp_mbti_*.json")):
        model_name = fpath.stem.replace("exp_mbti_", "")
        with open(fpath) as f:
            results[model_name] = json.load(f)
    return results


def extract_item_responses(model_data, persona="Default"):
    rows = []
    for r in model_data["results_by_persona"][persona]["responses"]:
        rows.append({
            "item_id": r["item_id"], "scale": r["scale"], "domain": r["domain"],
            "facet": r["facet"], "keyed": r["keyed"],
            "parsed_value": r["parsed_value"], "scored_value": r["scored_value"],
            "response_format": r["response_format"],
        })
    return pd.DataFrame(rows)


# ── Figure 1: Scree Plot + Factor Loading Heatmap ──
def fig1_scree_and_loadings():
    eig_df = pd.read_csv(OUTPUT_DIR / "efa_eigenvalues.csv")
    loadings_df = pd.read_csv(OUTPUT_DIR / "efa_domain_loadings.csv", index_col=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    eigs = eig_df["eigenvalue"].values
    x = range(1, len(eigs) + 1)
    ax1.plot(x, eigs, 'bo-', markersize=8, linewidth=2, label='Observed')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Kaiser (λ=1)')
    ax1.fill_between(range(1, 4), [eigs[0]]*3, alpha=0.15, color='blue')
    ax1.annotate(f'{eigs[0]:.2f}', (1, eigs[0]), textcoords="offset points",
                 xytext=(10, 5), fontsize=9, fontweight='bold')
    ax1.annotate(f'{eigs[1]:.2f}', (2, eigs[1]), textcoords="offset points",
                 xytext=(10, 5), fontsize=9, fontweight='bold')
    ax1.annotate(f'{eigs[2]:.2f}', (3, eigs[2]), textcoords="offset points",
                 xytext=(10, 5), fontsize=9, fontweight='bold')
    ax1.set_xlabel('Factor Number')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('(A) Scree Plot: Domain-Level Factor Analysis')
    ax1.legend()
    ax1.set_xticks(range(1, len(eigs) + 1))

    # Factor loading heatmap
    loadings_abs = loadings_df.abs()
    im = ax2.imshow(loadings_df.values, cmap='RdBu_r', aspect='auto',
                     vmin=-1, vmax=1)
    ax2.set_xticks(range(loadings_df.shape[1]))
    ax2.set_xticklabels(loadings_df.columns)
    ax2.set_yticks(range(loadings_df.shape[0]))
    ax2.set_yticklabels([n.replace('_', '\n', 1) if len(n) > 15 else n
                          for n in loadings_df.index], fontsize=7)
    ax2.set_title('(B) Factor Loadings (3-Factor Solution)')
    plt.colorbar(im, ax=ax2, label='Loading', shrink=0.8)

    # Annotate cells
    for i in range(loadings_df.shape[0]):
        for j in range(loadings_df.shape[1]):
            val = loadings_df.iloc[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=6, color=color)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_scree_loadings.png")
    plt.close()
    print(f"Saved fig1_scree_loadings.png")


# ── Figure 2: PIR by Domain + PIR×SDR Scatter ──
def fig2_pir_sdr():
    pir_df = pd.read_csv(OUTPUT_DIR / "pir_by_model_domain.csv")
    sdr_df = pd.read_csv(OUTPUT_DIR / "sdr_by_model.csv")
    pir_sdr = pd.read_csv(OUTPUT_DIR / "pir_sdr_crossvalidation.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PIR by domain (boxplot)
    pir_df["label"] = pir_df["scale"].str[:4] + ":" + pir_df["domain"].str[:12]
    domains = pir_df.groupby("label")["pir"].mean().sort_values(ascending=False).index

    data_by_domain = [pir_df[pir_df["label"] == d]["pir"].values for d in domains]
    bp = ax1.boxplot(data_by_domain, labels=range(len(domains)), patch_artist=True,
                      vert=True, showfliers=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(domains)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_xticklabels(domains, rotation=45, ha='right', fontsize=7)
    ax1.set_ylabel('Pairwise Inconsistency Rate')
    ax1.set_title('(A) PIR by Domain (higher = more inconsistent)')
    ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='0.5 threshold')
    ax1.legend()

    # PIR × SDR scatter
    ax2.scatter(pir_sdr["sdr_composite"], pir_sdr["mean_pir"],
                s=80, c='steelblue', edgecolors='navy', alpha=0.7, zorder=3)
    for _, row in pir_sdr.iterrows():
        model_short = row["model"].replace("Gemini-3.", "G3.").replace("Gemini_3.", "G3.") \
            .replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-") \
            .replace("DeepSeek-", "DS-").replace("Claude-", "C-") \
            .replace("MiniMax-", "MM-")[:12]
        ax2.annotate(model_short, (row["sdr_composite"], row["mean_pir"]),
                     fontsize=5.5, alpha=0.7)

    r, p = stats.spearmanr(pir_sdr["sdr_composite"], pir_sdr["mean_pir"])
    ax2.set_xlabel('SDR Composite (Social Desirability)')
    ax2.set_ylabel('Mean PIR (Inconsistency)')
    ax2.set_title(f'(B) PIR × SDR: r = {r:.3f}, p = {p:.4f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_pir_sdr.png")
    plt.close()
    print(f"Saved fig2_pir_sdr.png")


# ── Figure 3: Variance Decomposition Stacked Bar ──
def fig3_variance_decomposition():
    var_df = pd.read_csv(OUTPUT_DIR / "variance_decomposition.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (analysis, group) in enumerate(var_df.groupby("analysis")):
        components = group["component"].values
        percentages = group["percentage"].values
        bottom = 0
        colors = {'model': '#e41a1c', 'domain': '#377eb8', 'persona': '#4daf4a',
                  'item': '#ff7f00', 'residual': '#999999'}
        for comp, pct in zip(components, percentages):
            ax.barh(i, pct, left=bottom, color=colors.get(comp, 'gray'),
                    edgecolor='white', linewidth=0.5, label=comp if i == 0 else "")
            if pct > 3:
                ax.text(bottom + pct/2, i, f'{pct:.1f}%', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')
            bottom += pct

    ax.set_yticks(range(2))
    ax.set_yticklabels(var_df["analysis"].unique())
    ax.set_xlabel('Variance (%)')
    ax.set_title('Variance Decomposition: What Drives LLM Personality Responses?')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right',
              title='Component')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_variance_decomposition.png")
    plt.close()
    print(f"Saved fig3_variance_decomposition.png")


# ── Figure 4: Measurement Invariance Heatmap ──
def fig4_invariance():
    inv_df = pd.read_csv(OUTPUT_DIR / "persona_invariance.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap: model × persona
    pivot = inv_df.pivot_table(index="model", columns="persona", values="pearson_r")
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    im = ax1.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=0.8, aspect='auto')
    ax1.set_xticks(range(pivot.shape[1]))
    ax1.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=7)
    ax1.set_yticks(range(pivot.shape[0]))
    short_names = [n.replace("Gemini-3.", "G3.").replace("Gemini_3.", "G3.") \
                   .replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-") \
                   .replace("DeepSeek-", "DS-").replace("Claude-", "C-") \
                   .replace("MiniMax-", "MM-")[:15]
                  for n in pivot.index]
    ax1.set_yticklabels(short_names, fontsize=7)
    ax1.set_title('(A) Profile Correlation: Default vs MBTI Persona')
    plt.colorbar(im, ax=ax1, label='Pearson r', shrink=0.8)

    # Persona-level mean r
    persona_means = inv_df.groupby("persona")["pearson_r"].mean().sort_values()
    colors = ['red' if r < 0.3 else 'orange' if r < 0.5 else 'green' for r in persona_means]
    ax2.barh(range(len(persona_means)), persona_means.values, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(persona_means)))
    ax2.set_yticklabels(persona_means.index, fontsize=8)
    ax2.set_xlabel('Mean Pearson r with Default Profile')
    ax2.set_title('(B) Mean Profile Correlation by Persona')
    ax2.axvline(x=0.5, color='red', linestyle=':', alpha=0.5, label='0.5 threshold')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_measurement_invariance.png")
    plt.close()
    print(f"Saved fig4_measurement_invariance.png")


# ── Figure 5: Response Style Radar + Model Comparison ──
def fig5_response_styles():
    rs_df = pd.read_csv(OUTPUT_DIR / "response_styles.csv")
    default_rs = rs_df[rs_df["condition"] == "Default"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Acquiescence vs Extreme Response scatter
    ax1.scatter(default_rs["acquiescence"], default_rs["extreme_response"],
                s=100, c=default_rs["midpoint_response"], cmap='coolwarm',
                edgecolors='black', alpha=0.8, zorder=3)
    for _, row in default_rs.iterrows():
        model_short = row["model"].replace("Gemini-3.", "G3.").replace("Gemini_3.", "G3.") \
            .replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-") \
            .replace("DeepSeek-", "DS-").replace("Claude-", "C-") \
            .replace("MiniMax-", "MM-")[:10]
        ax1.annotate(model_short, (row["acquiescence"], row["extreme_response"]),
                     fontsize=5, alpha=0.7)
    ax1.set_xlabel('Acquiescence Bias')
    ax1.set_ylabel('Extreme Response Rate')
    ax1.set_title('(A) Response Style Map (color = midpoint rate)')
    ax1.grid(True, alpha=0.3)

    # PIR × SDR with model labels
    sdr_df = pd.read_csv(OUTPUT_DIR / "sdr_by_model.csv")
    pir_df = pd.read_csv(OUTPUT_DIR / "pir_by_model_domain.csv")
    pir_model = pir_df.groupby("model")["pir"].mean().reset_index()
    merged = pir_model.merge(sdr_df, on="model")

    # Color by family
    model_families = {}
    for m in merged["model"]:
        if "GPT" in m: model_families[m] = "OpenAI"
        elif "Claude" in m: model_families[m] = "Anthropic"
        elif "Gemini" in m: model_families[m] = "Google"
        elif "Qwen" in m: model_families[m] = "Alibaba"
        elif "DeepSeek" in m: model_families[m] = "DeepSeek"
        elif "Kimi" in m: model_families[m] = "Moonshot"
        else: model_families[m] = "Other"

    family_colors = {"OpenAI": "green", "Anthropic": "purple", "Google": "blue",
                     "Alibaba": "orange", "DeepSeek": "red", "Moonshot": "brown",
                     "Other": "gray"}

    for _, row in merged.iterrows():
        family = model_families.get(row["model"], "Other")
        ax2.scatter(row["pir"], row["lie_scale"],
                    c=family_colors[family], s=80, edgecolors='black',
                    alpha=0.7, zorder=3, label=family if family not in ax2.get_legend_handles_labels()[1] else "")
        model_short = row["model"].replace("Gemini-3.", "G3.").replace("Gemini_3.", "G3.") \
            .replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-") \
            .replace("DeepSeek-", "DS-").replace("Claude-", "C-") \
            .replace("MiniMax-", "MM-")[:10]
        ax2.annotate(model_short, (row["pir"], row["lie_scale"]),
                     fontsize=5, alpha=0.7)

    ax2.set_xlabel('Mean PIR (Inconsistency)')
    ax2.set_ylabel('Lie Scale Score')
    ax2.set_title('(B) Inconsistency vs Lie Scale (by developer)')
    ax2.grid(True, alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), fontsize=7, loc='upper right')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_response_styles.png")
    plt.close()
    print(f"Saved fig5_response_styles.png")


# ── Figure 6: Convergent Validity Forest Plot ──
def fig6_convergent_validity():
    conv_df = pd.read_csv(OUTPUT_DIR / "convergent_validity_enhanced.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = range(len(conv_df))
    colors = ['green' if match else 'red' for match in conv_df["sign_match"]]

    ax.barh(y_pos, conv_df["r_pearson"], color=colors, alpha=0.6, height=0.6,
            edgecolor='black', linewidth=0.5)

    for i, (_, row) in enumerate(conv_df.iterrows()):
        marker = '*' if row["p_pearson"] < 0.01 else '.' if row["p_pearson"] < 0.05 else ''
        ax.text(row["r_pearson"] + 0.02 * np.sign(row["r_pearson"]), i,
                f'r={row["r_pearson"]:.2f}{marker}', va='center', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(conv_df["pair"], fontsize=9)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Pearson r')
    ax.set_title('Cross-Scale Convergent Validity (green = sign match, red = mismatch)')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_convergent_validity.png")
    plt.close()
    print(f"Saved fig6_convergent_validity.png")


# ── Figure 7: Big Five Factor Collapse Summary ──
def fig7_factor_collapse():
    loadings_df = pd.read_csv(OUTPUT_DIR / "efa_domain_loadings.csv", index_col=0)
    eig_df = pd.read_csv(OUTPUT_DIR / "efa_eigenvalues.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Expected vs Observed factor structure comparison
    domains_order = ["IPIP_Neuroticism", "IPIP_Extraversion", "IPIP_Openness",
                     "IPIP_Agreeableness", "IPIP_Conscientiousness"]
    domain_short = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]

    # Which factor each domain loads on
    ipip_loadings = loadings_df.loc[domains_order]
    assigned_factors = [int(np.argmax(np.abs(ipip_loadings.iloc[i].values))) for i in range(5)]
    max_loadings = [np.max(np.abs(ipip_loadings.iloc[i].values)) for i in range(5)]

    # Expected: each domain on its own factor
    expected_factors = list(range(5))
    observed_factors = assigned_factors

    x = np.arange(5)
    width = 0.35
    bars1 = ax1.bar(x - width/2, expected_factors, width, label='Expected (5 factors)',
                     color='steelblue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, observed_factors, width, label='Observed (3 factors)',
                     color='coral', alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(domain_short, rotation=30, ha='right')
    ax1.set_ylabel('Assigned Factor (0-indexed)')
    ax1.set_title('(A) Big Five Factor Assignment: Expected vs Observed')
    ax1.legend()
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels(['F1', 'F2', 'F3', 'F4', 'F5'])

    # Add annotations for collapse
    for i in range(5):
        if observed_factors[i] != expected_factors[i]:
            ax1.annotate('COLLAPSED', (x[i] + width/2, observed_factors[i]),
                         textcoords="offset points", xytext=(0, 10),
                         fontsize=6, color='red', ha='center', fontweight='bold')

    # Variance explained: cumulative
    eigs = eig_df["eigenvalue"].values
    cum_var = np.cumsum(eigs) / np.sum(eigs)

    ax2.plot(range(1, len(eigs) + 1), cum_var * 100, 'bo-', markersize=8, linewidth=2)
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')

    # Mark the 3-factor solution
    ax2.plot(3, cum_var[2] * 100, 'ro', markersize=15, markerfacecolor='none',
             markeredgewidth=3, label=f'3 factors = {cum_var[2]*100:.1f}%')

    ax2.set_xlabel('Number of Factors')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('(B) Cumulative Variance Explained')
    ax2.legend()
    ax2.set_xticks(range(1, len(eigs) + 1))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_factor_collapse.png")
    plt.close()
    print(f"Saved fig7_factor_collapse.png")


# ── Figure 8: Model-Level Summary Dashboard ──
def fig8_model_dashboard():
    pir_df = pd.read_csv(OUTPUT_DIR / "pir_by_model_domain.csv")
    sdr_df = pd.read_csv(OUTPUT_DIR / "sdr_by_model.csv")
    inv_df = pd.read_csv(OUTPUT_DIR / "persona_invariance.csv")

    # Aggregate metrics per model
    model_metrics = pir_df.groupby("model").agg(
        mean_pir=("pir", "mean"),
    ).reset_index()

    model_metrics = model_metrics.merge(
        sdr_df[["model", "sdr_composite", "lie_scale", "extreme_response", "midpoint_response"]],
        on="model"
    )

    model_inv = inv_df.groupby("model").agg(
        mean_invariance_r=("pearson_r", "mean"),
    ).reset_index()
    model_metrics = model_metrics.merge(model_inv, on="model", how="left")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) PIR ranking
    ax = axes[0, 0]
    sorted_df = model_metrics.sort_values("mean_pir")
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_df)))
    ax.barh(range(len(sorted_df)), sorted_df["mean_pir"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(sorted_df)))
    short_names = [n.replace("Gemini-3.", "G3.").replace("Gemini_3.", "G3.") \
                   .replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-") \
                   .replace("DeepSeek-", "DS-").replace("Claude-", "C-") \
                   .replace("MiniMax-", "MM-")[:15]
                  for n in sorted_df["model"]]
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel('Mean PIR (Inconsistency Rate)')
    ax.set_title('(A) Inconsistency Ranking')
    ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.5)

    # (B) SDR ranking
    ax = axes[0, 1]
    sorted_df = model_metrics.sort_values("sdr_composite")
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_df)))
    ax.barh(range(len(sorted_df)), sorted_df["sdr_composite"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel('SDR Composite (Social Desirability)')
    ax.set_title('(B) Social Desirability Ranking')

    # (C) Persona Invariance
    ax = axes[1, 0]
    sorted_inv = model_metrics.dropna(subset=["mean_invariance_r"]).sort_values("mean_invariance_r")
    if len(sorted_inv) > 0:
        colors = ['green' if r > 0.5 else 'orange' if r > 0.3 else 'red'
                   for r in sorted_inv["mean_invariance_r"]]
        ax.barh(range(len(sorted_inv)), sorted_inv["mean_invariance_r"], color=colors, alpha=0.8)
        short_inv = [n.replace("Gemini-3.", "G3.").replace("Gemini_3.", "G3.") \
                     .replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-") \
                     .replace("DeepSeek-", "DS-").replace("Claude-", "C-") \
                     .replace("MiniMax-", "MM-")[:15]
                    for n in sorted_inv["model"]]
        ax.set_yticks(range(len(sorted_inv)))
        ax.set_yticklabels(short_inv, fontsize=7)
        ax.set_xlabel('Mean r(Default, MBTI)')
        ax.set_title('(C) Persona Invariance')
        ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.5)

    # (D) Summary: PIR vs Invariance
    ax = axes[1, 1]
    valid = model_metrics.dropna(subset=["mean_invariance_r", "mean_pir"])
    if len(valid) > 2:
        ax.scatter(valid["mean_pir"], valid["mean_invariance_r"],
                    s=100, c='steelblue', edgecolors='navy', alpha=0.7)
        for _, row in valid.iterrows():
            ms = row["model"].replace("Gemini-3.", "G3.").replace("Gemini_3.", "G3.") \
                .replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-") \
                .replace("DeepSeek-", "DS-").replace("Claude-", "C-") \
                .replace("MiniMax-", "MM-")[:10]
            ax.annotate(ms, (row["mean_pir"], row["mean_invariance_r"]),
                         fontsize=6, alpha=0.7)
        r, p = stats.spearmanr(valid["mean_pir"], valid["mean_invariance_r"])
        ax.set_xlabel('Mean PIR (Inconsistency)')
        ax.set_ylabel('Mean Invariance r')
        ax.set_title(f'(D) Inconsistency vs Persona Invariance (r={r:.2f})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_model_dashboard.png")
    plt.close()
    print(f"Saved fig8_model_dashboard.png")


def main():
    print("Generating publication-quality figures...")
    fig1_scree_and_loadings()
    fig2_pir_sdr()
    fig3_variance_decomposition()
    fig4_invariance()
    fig5_response_styles()
    fig6_convergent_validity()
    fig7_factor_collapse()
    fig8_model_dashboard()
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
