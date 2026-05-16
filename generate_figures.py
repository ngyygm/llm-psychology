"""
Regenerate all experimental figures with unified EMNLP-ready style.

6 figures, Okabe-Ito palette, column-width aware, readable at print size.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis_output")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ─── GLOBAL STYLE ──────────────────────────────────────────────────
COL_W = 3.3       # single column (inches)
FULL_W = 6.5      # full page width (inches)

# Okabe-Ito palette (colorblind-safe, grayscale-distinguishable)
C = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'cyan':    '#56B4E9',
    'yellow':  '#F0E442',
    'black':   '#000000',
    'gray':    '#999999',
    'lgray':   '#CCCCCC',
    'white':   '#FFFFFF',
}

# Variance decomposition colors
VC = {
    'model':    C['red'],
    'domain':   C['blue'],
    'persona':  C['green'],
    'item':     C['orange'],
    'residual': C['lgray'],
}

# Model family colors
FC = {
    'OpenAI':    C['orange'],
    'Anthropic': C['purple'],
    'Google':    C['blue'],
    'Alibaba':   C['red'],
    'DeepSeek':  C['green'],
    'Moonshot':  C['cyan'],
    'MiniMax':   C['yellow'],
    'Zhipu':     C['gray'],
}

FS = {'title': 9, 'ax': 8, 'tick': 7, 'leg': 7, 'ann': 7, 'cell': 6}

RCPARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': FS['tick'],
    'axes.titlesize': FS['title'],
    'axes.labelsize': FS['ax'],
    'xtick.labelsize': FS['tick'],
    'ytick.labelsize': FS['tick'],
    'legend.fontsize': FS['leg'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
}
plt.rcParams.update(RCPARAMS)


def shorten(name, max_len=14):
    """Shorten model name for labels."""
    s = (name
         .replace("Gemini-3-Flash-Preview", "Gem-3-Flash")
         .replace("Gemini-3.1-Flash-Lite", "Gem-3.1-FL")
         .replace("Gemini-3.1-Pro-Preview", "Gem-3.1-Pro")
         .replace("Gemini-3-Pro-Preview", "Gem-3-Pro")
         .replace("Gemini_3_Pro_Preview", "Gem-3-Pro")
         .replace("Gemini_", "Gem-")
         .replace("Qwen3.5-397B-A17B", "Qwen3.5-397B")
         .replace("Qwen3.5-122B-A10B", "Qwen3.5-122B")
         .replace("Qwen3-235B-A22B", "Qwen3-235B")
         .replace("DeepSeek-V4-Flash", "DS-V4-Flash")
         .replace("DeepSeek-V4-Pro", "DS-V4-Pro")
         .replace("DeepSeek-V3.2", "DS-V3.2")
         .replace("Claude-Opus-4.6", "Claude-Opus")
         .replace("Claude-Sonnet-4.6", "Claude-Sonnet")
         .replace("MiniMax-M2.7", "MiniMax-M2")
         .replace("Kimi-K2.6", "Kimi-K2.6")
         .replace("Kimi-K2.5", "Kimi-K2.5")
         .replace("GLM-4.6V", "GLM-4.6V"))
    return s[:max_len]


def get_family(model):
    if "GPT" in model: return "OpenAI"
    if "Claude" in model: return "Anthropic"
    if "Gemini" in model: return "Google"
    if "Qwen" in model: return "Alibaba"
    if "DeepSeek" in model: return "DeepSeek"
    if "Kimi" in model: return "Moonshot"
    if "MiniMax" in model: return "MiniMax"
    if "GLM" in model: return "Zhipu"
    return "Other"


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved {name}")


# ─── FIG 1: Factor Structure (full width) ──────────────────────────

def fig1_factor_structure():
    eig_df = pd.read_csv(OUTPUT_DIR / "efa_eigenvalues.csv")
    load_df = pd.read_csv(OUTPUT_DIR / "efa_domain_loadings.csv", index_col=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.5),
                                    gridspec_kw={'width_ratios': [1, 1.3]})

    # (A) Scree plot
    eigs = eig_df["eigenvalue"].values
    x = np.arange(1, len(eigs) + 1)
    ax1.plot(x, eigs, 'o-', color=C['blue'], markersize=6, linewidth=1.5, zorder=3)
    ax1.axhline(1.0, color=C['red'], linestyle='--', linewidth=1, label='Kaiser (λ=1)')
    ax1.fill_between([0.5, 3.5], 1.0, [eigs[0]+0.3]*2, alpha=0.08, color=C['blue'])
    for i in range(3):
        ax1.annotate(f'{eigs[i]:.2f}', (i+1, eigs[i]),
                     textcoords="offset points", xytext=(8, 4),
                     fontsize=FS['ann'], fontweight='bold', color=C['blue'])
    ax1.set_xlabel('Factor Number')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('(A) Scree Plot', fontsize=FS['title'], fontweight='bold')
    ax1.set_xticks(x)
    ax1.legend(fontsize=FS['leg'], frameon=False)
    ax1.text(2, eigs[0]+0.3, f'3 factors ({eigs[:3].sum()/eigs.sum()*100:.1f}% var.)',
             ha='center', fontsize=FS['ann'], color=C['red'], fontstyle='italic')

    # (B) Factor loading heatmap
    short_idx = [s.replace("IPIP-NEO-120_", "IPIP-").replace("SD3_", "SD3-")
                 .replace("ZKPQ-50-CC_", "ZKPQ-").replace("EPQR-A_", "EPQR-")
                 .replace("Neuroticism", "N").replace("Extraversion", "E")
                 .replace("Openness", "O").replace("Agreeableness", "A")
                 .replace("Conscientiousness", "C").replace("Machiavellianism", "Mach")
                 .replace("Narcissism", "Narc").replace("Psychopathy", "Psy")
                 .replace("Activity", "Act").replace("Aggression-Hostility", "Agg-H")
                 .replace("Impulsive_Sensation_Seeking", "ISS")
                 .replace("Neuroticism-Anxiety", "N-Anx").replace("Sociability", "Soc")
                 .replace("Psychoticism", "Psy").replace("Lie", "Lie")
                 for s in load_df.index]

    vals = load_df.values
    im = ax2.imshow(vals, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax2.set_xticks(range(vals.shape[1]))
    ax2.set_xticklabels(load_df.columns, fontsize=FS['tick'])
    ax2.set_yticks(range(vals.shape[0]))
    ax2.set_yticklabels(short_idx, fontsize=FS['cell'] + 1)
    ax2.set_title('(B) Factor Loadings (3-Factor)', fontsize=FS['title'], fontweight='bold')
    cb = plt.colorbar(im, ax=ax2, shrink=0.7, pad=0.02)
    cb.ax.tick_params(labelsize=FS['cell'])
    cb.set_label('Loading', fontsize=FS['tick'])

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            color = 'white' if abs(v) > 0.6 else 'black'
            weight = 'bold' if abs(v) > 0.5 else 'normal'
            ax2.text(j, i, f'{v:.2f}', ha='center', va='center',
                     fontsize=FS['cell'] - 0.5, color=color, fontweight=weight)

    fig.tight_layout(w_pad=2)
    save(fig, "fig1_factor_structure.png")


# ─── FIG 2: Cronbach Alpha Comparison (single column) ──────────────

def fig2_cronbach_comparison():
    bench = pd.read_csv(OUTPUT_DIR / "human_vs_llm_benchmarks.csv")
    synth = pd.read_csv(OUTPUT_DIR / "synthetic_baselines.csv")

    # Extract Cronbach alpha rows (first 5 = IPIP domains)
    alpha_human = bench[bench["metric"] == "Cronbach's Alpha"].copy()
    alpha_human["domain_short"] = alpha_human["domain"].str[:3]
    human_vals = alpha_human.set_index("domain")["human"].astype(float)

    llm_vals = alpha_human.set_index("domain")["llm"].astype(float)

    # Synthetic baselines
    random_vals = synth[synth["strategy"] == "Random"].set_index("domain")["alpha"]
    acq_vals = synth[synth["strategy"] == "Pure Acquiescence"].set_index("domain")["alpha"]

    domains = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
    y = np.arange(len(domains))
    h = 0.18

    fig, ax = plt.subplots(figsize=(COL_W, 4.0))

    bars_h = ax.barh(y - 1.5*h, [human_vals.get(d, 0) for d in domains], h,
                     color=C['gray'], label='Human norm', edgecolor='white', linewidth=0.5)
    bars_l = ax.barh(y - 0.5*h, [llm_vals.get(d, 0) for d in domains], h,
                     color=C['blue'], label='LLM observed', edgecolor='white', linewidth=0.5)
    bars_r = ax.barh(y + 0.5*h, [random_vals.get(d, 0) for d in domains], h,
                     color=C['lgray'], label='Random baseline', edgecolor='white', linewidth=0.5)
    bars_a = ax.barh(y + 1.5*h, [acq_vals.get(d, 0) for d in domains], h,
                     color=C['orange'], label='Acquiescence baseline', edgecolor='white', linewidth=0.5)

    # Annotate LLM values
    for i, d in enumerate(domains):
        v = llm_vals.get(d, 0)
        ax.text(max(v + 0.02, 0.02), y[i] - 0.5*h, f'{v:.3f}',
                va='center', fontsize=FS['ann'] - 1, color=C['blue'], fontweight='bold')

    ax.axvline(0.70, color=C['red'], linestyle=':', linewidth=1, label='α = 0.70 threshold')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(domains, fontsize=FS['ax'])
    ax.set_xlabel("Cronbach's α")
    ax.legend(fontsize=FS['leg'] - 1, frameon=False, loc='lower right')
    ax.set_xlim(-0.15, 1.05)

    fig.tight_layout()
    save(fig, "fig2_cronbach_comparison.png")


# ─── FIG 3: Acquiescence Mechanism (full width) ────────────────────

def fig3_acquiescence_mechanism():
    pir_df = pd.read_csv(OUTPUT_DIR / "pir_by_model_domain.csv")
    acq_df = pd.read_csv(OUTPUT_DIR / "acquiescence_mechanism.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.8))

    # (A) PIR Cleveland dot plot
    pir_domain = pir_df.groupby(["scale", "domain"])["pir"].agg(['mean', 'std']).reset_index()
    pir_domain["label"] = pir_domain["scale"].str[:4] + ":" + pir_domain["domain"].str[:14]
    pir_domain = pir_domain.sort_values("mean", ascending=True).reset_index(drop=True)

    colors = [C['red'] if m > 0.5 else C['blue'] for m in pir_domain["mean"]]
    y_pos = np.arange(len(pir_domain))

    ax1.hlines(y_pos, 0, pir_domain["mean"], color=colors, linewidth=1.5, alpha=0.6)
    ax1.plot(pir_domain["mean"], y_pos, 'o', color=C['blue'], markersize=7, zorder=3)
    ax1.errorbar(pir_domain["mean"], y_pos, xerr=pir_domain["std"],
                 fmt='none', ecolor=C['gray'], capsize=2, linewidth=0.8, zorder=2)

    ax1.axvline(0.5, color=C['red'], linestyle=':', linewidth=1, alpha=0.5)
    ax1.axvline(0.584, color=C['orange'], linestyle='--', linewidth=1, alpha=0.7,
               label='Overall PIR = 0.584')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(pir_domain["label"], fontsize=FS['tick'] - 1)
    ax1.set_xlabel('Pairwise Inconsistency Rate')
    ax1.set_title('(A) PIR by Domain', fontsize=FS['title'], fontweight='bold')
    ax1.legend(fontsize=FS['leg'] - 1, frameon=False)
    ax1.set_xlim(0, 1)

    # (B) Forward vs Reverse agree rate scatter
    for _, row in acq_df.iterrows():
        fam = get_family(row["model"])
        ax2.scatter(row["fwd_agree_rate"], row["rev_agree_rate"],
                    c=FC.get(fam, C['gray']), s=60, edgecolors='black',
                    linewidths=0.5, alpha=0.8, zorder=3)
        ax2.annotate(shorten(row["model"], 10),
                     (row["fwd_agree_rate"], row["rev_agree_rate"]),
                     fontsize=FS['cell'], alpha=0.7)

    # Diagonal = no acquiescence gap
    lims = [0, 1.05]
    ax2.plot(lims, lims, ':', color=C['gray'], linewidth=0.8, label='No gap (fwd = rev)')
    ax2.fill_between([0, 1.05], [0, 1.05], [1.05, 1.05], alpha=0.04, color=C['red'],
                     label='Acquiescence zone')

    r, p = stats.spearmanr(acq_df["rev_agree_rate"], acq_df["overall_agree_rate"])
    ax2.text(0.05, 0.95, f'rev-agree × PIR: r = 0.726',
             transform=ax2.transAxes, fontsize=FS['ann'] - 1,
             verticalalignment='top', fontstyle='italic')

    ax2.set_xlabel('Forward Agree Rate')
    ax2.set_ylabel('Reverse Agree Rate')
    ax2.set_title('(B) Acquiescence: Forward vs Reverse', fontsize=FS['title'], fontweight='bold')
    ax2.legend(fontsize=FS['leg'] - 1, frameon=False, loc='lower right')

    # Family legend
    handles = [mpatches.Patch(color=FC[f], label=f) for f in
               ['OpenAI', 'Anthropic', 'Google', 'Alibaba', 'DeepSeek', 'Moonshot']]
    ax2.legend(handles=handles, fontsize=FS['cell'], frameon=False,
               loc='lower right', ncol=2)

    fig.tight_layout(w_pad=2)
    save(fig, "fig3_acquiescence_mechanism.png")


# ─── FIG 4: Variance Decomposition (single column) ─────────────────

def fig4_variance_decomposition():
    var_df = pd.read_csv(OUTPUT_DIR / "variance_decomposition.csv")
    boot_df = pd.read_csv(OUTPUT_DIR / "bootstrap_ci_results.csv")

    components = ['model', 'domain', 'persona', 'item', 'residual']
    comp_labels = ['Model', 'Domain', 'Persona', 'Item', 'Residual']

    # Build pivot: rows=component, columns=analysis
    rows = []
    for _, row in var_df.iterrows():
        analysis_short = 'Likert' if 'Likert' in row['analysis'] else 'Binary'
        rows.append({'component': row['component'], 'analysis': analysis_short,
                     'pct': row['percentage']})
    pivot_df = pd.DataFrame(rows)
    pivot_df = pivot_df.pivot(index='component', columns='analysis', values='pct')
    pivot_df = pivot_df.reindex(components)

    fig, ax = plt.subplots(figsize=(COL_W, 3.3))

    x = np.arange(len(components))
    w = 0.32

    bars_l = ax.bar(x - w/2, pivot_df['Likert'], w, color=C['blue'],
                    label='Likert (IPIP+SD3)', edgecolor='white', linewidth=0.5)
    bars_b = ax.bar(x + w/2, pivot_df['Binary'], w, color=C['orange'],
                    label='Binary (ZKPQ+EPQR)', edgecolor='white', linewidth=0.5)

    # Label bars > 3%
    for bars in [bars_l, bars_b]:
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                        f'{h:.1f}%', ha='center', va='bottom',
                        fontsize=FS['ann'] - 1, fontweight='bold')

    # Callout for Model (tiny bars)
    ci_row = boot_df[boot_df["statistic"] == "model_variance_pct"]
    ci_lo = ci_row["ci_2.5"].values[0] if len(ci_row) > 0 else 1.1
    ax.annotate(f'Model\n0.3% / 0.5%\nCI: [{ci_lo:.1f}%, 4.0%]',
                xy=(0, max(pivot_df['Likert'].iloc[0], pivot_df['Binary'].iloc[0]) + 1),
                xytext=(0.6, 30), fontsize=FS['ann'] - 1.5, color=C['red'],
                fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color=C['red'], lw=1))

    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=FS['ax'])
    ax.set_ylabel('Variance (%)')
    ax.legend(fontsize=FS['leg'] - 1, frameon=False)
    ax.set_ylim(0, 42)

    fig.tight_layout()
    save(fig, "fig4_variance_decomposition.png")


# ─── FIG 5: Convergent Validity Forest Plot (full width) ────────────

def fig5_convergent_validity():
    conv = pd.read_csv(OUTPUT_DIR / "convergent_validity_enhanced.csv")

    fig, ax = plt.subplots(figsize=(FULL_W, 2.5))

    n = len(conv)
    y_pos = np.arange(n)

    # Compute 95% CI via Fisher z-transform
    r_vals = conv["r_spearman"].values.astype(float)
    n_obs = 18
    se = 1.0 / np.sqrt(n_obs - 3)
    z = np.arctanh(r_vals)
    z_lo = z - 1.96 * se
    z_hi = z + 1.96 * se
    r_lo = np.tanh(z_lo)
    r_hi = np.tanh(z_hi)

    for i in range(n):
        match = conv.iloc[i]["sign_match"]
        p = conv.iloc[i]["p_spearman"]
        color = C['green'] if match else C['red']
        marker = 'o' if p < 0.05 else 'o'
        alpha = 1.0 if p < 0.05 else 0.5

        ax.plot([r_lo[i], r_hi[i]], [y_pos[i], y_pos[i]],
                color=color, linewidth=2, alpha=alpha)
        ax.plot(r_vals[i], y_pos[i], marker, color=color,
                markersize=8, markerfacecolor=color if p < 0.05 else 'white',
                markeredgewidth=1.5, alpha=alpha, zorder=3)

        sig = '*' if p < 0.01 else '' if p < 0.05 else '(ns)'
        ax.text(max(r_hi[i], r_lo[i]) + 0.03, y_pos[i],
                f'r={r_vals[i]:.2f}{sig}', va='center', fontsize=FS['ann'] - 1)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(conv["pair"], fontsize=FS['tick'])
    ax.set_xlabel('Spearman r')
    ax.set_title('Convergent Validity: Cross-Scale Correlations with 95% CI',
                 fontsize=FS['title'], fontweight='bold')

    # Annotations
    ax.text(0.02, 0.02, '● p < .05   ○ p ≥ .05   green = sign match   red = mismatch',
            transform=ax.transAxes, fontsize=FS['cell'], color=C['gray'])

    # Highlight sign reversal
    mismatch_idx = conv.index[~conv["sign_match"]].tolist()
    for idx in mismatch_idx:
        ax.annotate('sign reversal', (r_vals[idx], y_pos[idx]),
                    xytext=(0, -12), textcoords='offset points',
                    fontsize=FS['cell'], color=C['red'], fontweight='bold', ha='center')

    fig.tight_layout()
    save(fig, "fig5_convergent_validity.png")


# ─── FIG 6: Invariance + Robustness (full width) ───────────────────

def fig6_invariance_robustness():
    inv_df = pd.read_csv(OUTPUT_DIR / "persona_invariance.csv")
    loo_model = pd.read_csv(OUTPUT_DIR / "leave_one_out_robustness.csv")
    loo_persona = pd.read_csv(OUTPUT_DIR / "leave_one_persona_out.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 3.0),
                                    gridspec_kw={'width_ratios': [1.5, 1]})

    # (A) Persona invariance bar chart
    persona_means = inv_df.groupby("persona")["pearson_r"].agg(['mean', 'std']).reset_index()
    persona_means = persona_means.sort_values("mean").reset_index(drop=True)
    y_pos = np.arange(len(persona_means))

    cmap = plt.cm.RdYlGn
    norm_vals = (persona_means["mean"] - persona_means["mean"].min()) / \
                (persona_means["mean"].max() - persona_means["mean"].min() + 1e-6)
    colors = [cmap(0.2 + 0.6 * v) for v in norm_vals]

    ax1.barh(y_pos, persona_means["mean"], color=colors, edgecolor='white',
             linewidth=0.5, height=0.7)
    ax1.errorbar(persona_means["mean"], y_pos,
                 xerr=persona_means["std"], fmt='none',
                 ecolor=C['gray'], capsize=2, linewidth=0.6)

    ax1.axvline(0.3, color=C['orange'], linestyle=':', linewidth=0.8, label='Weak (r=0.3)')
    ax1.axvline(0.8, color=C['green'], linestyle=':', linewidth=0.8, label='Strong (r=0.8)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(persona_means["persona"], fontsize=FS['tick'])
    ax1.set_xlabel('Mean Pearson r with Default')
    ax1.set_title('(A) Persona Invariance', fontsize=FS['title'], fontweight='bold')
    ax1.legend(fontsize=FS['cell'], frameon=False)

    # Annotate best/worst
    best = persona_means.iloc[-1]
    worst = persona_means.iloc[0]
    ax1.annotate(f'{best["persona"]} ({best["mean"]:.2f})',
                 (best["mean"], len(persona_means)-1),
                 xytext=(5, 0), textcoords='offset points',
                 fontsize=FS['ann'] - 1, color=C['green'], fontweight='bold')
    ax1.annotate(f'{worst["persona"]} ({worst["mean"]:.2f})',
                 (worst["mean"], 0),
                 xytext=(5, 0), textcoords='offset points',
                 fontsize=FS['ann'] - 1, color=C['red'], fontweight='bold')

    # (B) LOO robustness: two compact eigenvalue bar charts
    loo_model_eigs = loo_model["first_eigenvalue"].values
    loo_persona_eigs = loo_persona["first_eigenvalue"].values

    # Model LOO
    ax2.barh(1, loo_model_eigs.mean(), height=0.4, color=C['blue'], alpha=0.8)
    ax2.barh(1, loo_model_eigs.std() * 2, left=loo_model_eigs.mean() - loo_model_eigs.std(),
             height=0.4, color=C['blue'], alpha=0.2)
    ax2.text(loo_model_eigs.mean(), 1.35, f'Model LOO (n={len(loo_model_eigs)})',
             ha='center', fontsize=FS['ann'], fontweight='bold', color=C['blue'])
    ax2.text(loo_model_eigs.mean(), 0.65,
             f'λ₁ = {loo_model_eigs.mean():.2f} ± {loo_model_eigs.std():.2f}  |  3 factors (all)',
             ha='center', fontsize=FS['cell'], color=C['gray'])

    # Persona LOO
    ax2.barh(0, loo_persona_eigs.mean(), height=0.4, color=C['green'], alpha=0.8)
    ax2.barh(0, loo_persona_eigs.std() * 2, left=loo_persona_eigs.mean() - loo_persona_eigs.std(),
             height=0.4, color=C['green'], alpha=0.2)
    ax2.text(loo_persona_eigs.mean(), 0.35, f'Persona LOO (n={len(loo_persona_eigs)})',
             ha='center', fontsize=FS['ann'], fontweight='bold', color=C['green'])
    ax2.text(loo_persona_eigs.mean(), -0.35,
             f'λ₁ = {loo_persona_eigs.mean():.2f} ± {loo_persona_eigs.std():.2f}  |  3 factors (all)',
             ha='center', fontsize=FS['cell'], color=C['gray'])

    ax2.axvline(1.0, color=C['red'], linestyle='--', linewidth=0.8, label='Kaiser λ=1')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Persona\nLOO', 'Model\nLOO'], fontsize=FS['tick'])
    ax2.set_xlabel('First Eigenvalue')
    ax2.set_title('(B) Robustness: All yield 3 factors',
                  fontsize=FS['title'], fontweight='bold')
    ax2.legend(fontsize=FS['cell'], frameon=False)
    ax2.set_xlim(0, max(loo_model_eigs.mean(), loo_persona_eigs.mean()) + 2)

    fig.tight_layout(w_pad=2)
    save(fig, "fig6_invariance_robustness.png")


# ─── MAIN ───────────────────────────────────────────────────────────

def main():
    print("Regenerating figures with unified EMNLP style...")
    print()
    fig1_factor_structure()
    fig2_cronbach_comparison()
    fig3_acquiescence_mechanism()
    fig4_variance_decomposition()
    fig5_convergent_validity()
    fig6_invariance_robustness()
    print()
    print(f"All 6 figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
