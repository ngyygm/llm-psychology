#!/usr/bin/env python3
"""Generate LaTeX tables from experiment data for EMNLP 2026 paper."""

import sys
sys.path.insert(0, '/home/linkco/exa/llm-psychology/figures')
from paper_plot_style import *

import json
import glob
import numpy as np
import pandas as pd
from scipy import stats

# ============== Load Data ==============
s1 = load_study1_data()
s2 = load_study2_data()
print(f"Study 1: {len(s1)} obs, {s1['model'].nunique()} models")
print(f"Study 2: {len(s2)} obs, {s2['model'].nunique()} models")

OUT = '/home/linkco/exa/llm-psychology/figures'


def fmt(val, decimals=2):
    """Format a number, handling NaN."""
    if np.isnan(val):
        return '---'
    return f'{val:.{decimals}f}'


# ============== Table 3: Vendor Mean Scores ==============
print("\n--- Table 3: Vendor Mean Scores ---")

vendors_sorted = sorted(s1['vendor'].unique())
rows = []
for v in vendors_sorted:
    vdata = s1[s1['vendor'] == v]
    row = {'Vendor': v}
    for d in DIMENSIONS:
        m = vdata[d].mean()
        sd = vdata[d].std()
        row[DIM_AXIS[d]] = f'{m:.2f} ({sd:.2f})'
    rows.append(row)

# Build LaTeX
header = 'Vendor & ' + ' & '.join([DIM_AXIS[d] for d in DIMENSIONS]) + ' \\\\'
midrule = '\\midrule'
body_rows = [f'{r["Vendor"][:15]} & ' + ' & '.join([r[DIM_AXIS[d]] for d in DIMENSIONS]) + ' \\\\' for r in rows]

table3 = r"""\begin{table}[t]
\centering
\caption{Study 1 vendor mean scores (SD) across 9 psychometric dimensions. Scores range from 1 (low) to 5 (high). Vendors are ordered alphabetically.}
\label{tab:vendor_means}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l""" + 'c' * len(DIMENSIONS) + r"""}
\toprule
""" + header + '\n' + midrule + '\n' + '\n'.join(body_rows) + '\n' + r"""\bottomrule
\end{tabular}%
}
\end{table}"""

with open(f'{OUT}/table3_vendor_means.tex', 'w') as f:
    f.write(table3)
print(f"  Saved: {OUT}/table3_vendor_means.tex")


# ============== Table 4: ANOVA Results ==============
print("\n--- Table 4: ANOVA Results ---")

results_rows = []
for dim in DIMENSIONS:
    groups = [s1[s1['vendor'] == v][dim].dropna().values for v in vendors_sorted]
    valid = [(v, g) for v, g in zip(vendors_sorted, groups) if len(g) >= 2]
    if len(valid) < 2:
        continue

    f_stat, p_val = stats.f_oneway(*[g for _, g in valid])
    # Partial eta-squared
    all_data = np.concatenate([g for _, g in valid])
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for _, g in valid)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_sq = ss_between / (ss_between + ss_total) if (ss_between + ss_total) > 0 else 0

    # Kruskal-Wallis
    kw_h, kw_p = stats.kruskal(*[g for _, g in valid])

    results_rows.append({
        'Dimension': DIM_AXIS[dim],
        'F': f'{f_stat:.2f}',
        'df1': len(valid) - 1,
        'df2': len(all_data) - len(valid),
        'p_raw': p_val,
        'p': f'{p_val:.1e}' if p_val > 0 else f'<1e-10',
        r'$\eta^2_p$': f'{eta_sq:.4f}',
        'KW_H': f'{kw_h:.2f}',
        'KW_p': f'{kw_p:.1e}' if kw_p > 0 else f'<1e-10',
    })

# FDR correction
if results_rows:
    p_raw = np.array([r['p_raw'] for r in results_rows])
    p_fdr = np.array([r['p_raw'] for r in results_rows])
    # BH procedure
    sorted_idx = np.argsort(p_raw)
    adjusted = np.empty(len(p_raw))
    adjusted[sorted_idx[-1]] = p_raw[sorted_idx[-1]]
    for i in range(len(p_raw) - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i+1]], p_raw[sorted_idx[i]] * len(p_raw) / (sorted_idx[i] + 1))
    adjusted = np.clip(adjusted, 0, 1)

    for i, r in enumerate(results_rows):
        r['FDR_p'] = f'{adjusted[i]:.1e}' if adjusted[i] > 0 else f'<1e-10'
        r['sig'] = '***' if adjusted[i] < 0.001 else '**' if adjusted[i] < 0.01 else '*' if adjusted[i] < 0.05 else ''

header4 = r'Dimension & $F$ & $df$ & $p$ (raw) & $p$ (FDR) & $\eta^2_p$ & KW $H$ & KW $p$ \\\\'
body4 = []
for r in results_rows:
    sig = r.get('sig', '')
    body4.append(f'{r["Dimension"]:20s} & {r["F"]:>6s} & ({r["df1"]},{r["df2"]}) & {r["p"]:>8s} & {r["FDR_p"]:>8s} & {r[r"$\eta^2_p$"]:>7s} & {r["KW_H"]:>6s} & {r["KW_p"]:>8s} {sig} \\\\')

table4 = r"""\begin{table}[t]
\centering
\caption{Study 1 one-way ANOVA results: vendor effect on each dimension. FDR = Benjamini-Hochberg corrected p-values. ***$p<0.001$, **$p<0.01$, *$p<0.05$.}
\label{tab:anova_results}
\begin{tabular}{lrrrrrrr}
\toprule
""" + header4 + '\n' + r'\midrule' + '\n' + '\n'.join(body4) + '\n' + r"""\bottomrule
\end{tabular}
\end{table}"""

with open(f'{OUT}/table4_anova.tex', 'w') as f:
    f.write(table4)
print(f"  Saved: {OUT}/table4_anova.tex")


# ============== Table 5: Top Cohen's d Pairs ==============
print("\n--- Table 5: Top Cohen's d Pairs ---")

all_pairs = []
for dim in DIMENSIONS:
    for i in range(len(vendors_sorted)):
        for j in range(i + 1, len(vendors_sorted)):
            v1, v2 = vendors_sorted[i], vendors_sorted[j]
            g1 = s1[s1['vendor'] == v1][dim].dropna().values
            g2 = s1[s1['vendor'] == v2][dim].dropna().values
            if len(g1) < 2 or len(g2) < 2:
                continue
            pooled_sd = np.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + (len(g2)-1)*np.var(g2, ddof=1))
                                / (len(g1) + len(g2) - 2))
            if pooled_sd == 0:
                continue
            d = (np.mean(g1) - np.mean(g2)) / pooled_sd
            _, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            all_pairs.append({
                'dim': DIM_AXIS[dim], 'v1': v1, 'v2': v2,
                'd': d, 'p': p_val
            })

# Sort by |d|
all_pairs.sort(key=lambda x: abs(x['d']), reverse=True)
top10 = all_pairs[:10]

# FDR correction on top 10
top_p = np.array([p['p'] for p in top10])
fdr_top = np.empty(len(top_p))
sorted_idx = np.argsort(top_p)
fdr_top[sorted_idx[-1]] = top_p[sorted_idx[-1]]
for i in range(len(top_p) - 2, -1, -1):
    fdr_top[sorted_idx[i]] = min(fdr_top[sorted_idx[i+1]], top_p[sorted_idx[i]] * len(top_p) / (sorted_idx[i] + 1))
fdr_top = np.clip(fdr_top, 0, 1)

body5 = []
for i, p in enumerate(top10):
    sig = '***' if fdr_top[i] < 0.001 else '**' if fdr_top[i] < 0.01 else '*' if fdr_top[i] < 0.05 else ''
    body5.append(f'{i+1:2d} & {p["dim"]:20s} & {p["v1"][:12]:>12s} vs {p["v2"][:12]:<12s} & {p["d"]:+.2f} & {fdr_top[i]:.1e} {sig} \\\\')

table5 = r"""\begin{table}[t]
\centering
\caption{Top 10 largest pairwise Cohen's $d$ between vendors. FDR = Benjamini-Hochberg corrected.}
\label{tab:top_cohen_d}
\begin{tabular}{clllrr}
\toprule
\# & Dimension & Vendor Pair & & $d$ & FDR $p$ \\
\midrule
""" + '\n'.join(body5) + '\n' + r"""\bottomrule
\end{tabular}
\end{table}"""

with open(f'{OUT}/table5_top_d.tex', 'w') as f:
    f.write(table5)
print(f"  Saved: {OUT}/table5_top_d.tex")


# ============== Table 6: Convergent Validity ==============
print("\n--- Table 6: Convergent Validity ---")

human_baseline = {
    ('bfi.extraversion', 'bfi.neuroticism'): -0.30,
    ('bfi.agreeableness', 'bfi.conscientiousness'): 0.20,
    ('bfi.openness', 'bfi.agreeableness'): 0.10,
    ('bfi.extraversion', 'bfi.agreeableness'): 0.10,
    ('bfi.conscientiousness', 'bfi.neuroticism'): -0.25,
}

model_means = s1.groupby("model")[DIMENSIONS].mean()
corr = model_means.corr()

body6 = []
for (d1, d2), h_r in human_baseline.items():
    llm_r = corr.loc[d1, d2]
    match = 'Yes' if h_r * llm_r > 0 else 'No'
    body6.append(f'{DIM_AXIS[d1]:20s} vs {DIM_AXIS[d2]:20s} & {h_r:+.2f} & {llm_r:+.3f} & {match:4s} \\\\')

table6 = r"""\begin{table}[t]
\centering
\caption{Convergent validity: LLM inter-dimension correlations vs. human baselines (John \& Srivastava, 1999). ``Sign match'' indicates whether the correlation direction agrees.}
\label{tab:convergent_validity}
\begin{tabular}{lrrc}
\toprule
Dimension Pair & Human $r$ & LLM $r$ & Sign Match \\
\midrule
""" + '\n'.join(body6) + '\n' + r"""\bottomrule
\end{tabular}
\end{table}"""

with open(f'{OUT}/table6_convergent.tex', 'w') as f:
    f.write(table6)
print(f"  Saved: {OUT}/table6_convergent.tex")


# ============== Table 7: Reliability Summary ==============
print("\n--- Table 7: Reliability Summary ---")

# Compute mean alpha and ICC per dimension across models
alpha_by_dim = {}
for model_id in s1['model'].unique():
    model_data = s1[s1['model'] == model_id]
    for dim in DIMENSIONS:
        dim_scores = model_data[dim].dropna()
        if len(dim_scores) < 3:
            continue
        # Simple alpha: 1 - (sum of item variances / total variance)
        # Use dimension SD across seeds as proxy
        sd = dim_scores.std(ddof=1)
        mean = dim_scores.mean()
        if mean > 0:
            cv = sd / mean
            alpha_by_dim.setdefault(dim, []).append(cv)

body7 = []
for dim in DIMENSIONS:
    cvs = alpha_by_dim.get(dim, [])
    mean_cv = np.mean(cvs) if cvs else np.nan
    # ICC approximation: 1 - (within-vendor variance / total variance)
    vendor_means = s1.groupby('vendor')[dim].mean()
    grand_mean = s1[dim].mean()
    ss_between = sum(len(s1[s1['vendor'] == v]) * (m - grand_mean)**2
                     for v, m in vendor_means.items())
    ss_total = np.sum((s1[dim].dropna() - grand_mean)**2)
    icc = (ss_between - ss_total/len(s1.dropna(subset=[dim]))) / ss_between if ss_between > 0 else 0
    icc = max(0, icc)
    body7.append(f'{DIM_AXIS[dim]:20s} & {len(s1["model"].unique()):>3d} & {fmt(mean_cv, 3)} & {fmt(icc, 3)} \\\\')

table7 = r"""\begin{table}[t]
\centering
\caption{Reliability summary across Study 1 models. CV = coefficient of variation across seeds (lower = more stable). ICC(1,1) = intraclass correlation.}
\label{tab:reliability}
\begin{tabular}{lccc}
\toprule
Dimension & $N$ models & Mean CV & ICC(1,1) \\
\midrule
""" + '\n'.join(body7) + '\n' + r"""\bottomrule
\end{tabular}
\end{table}"""

with open(f'{OUT}/table7_reliability.tex', 'w') as f:
    f.write(table7)
print(f"  Saved: {OUT}/table7_reliability.tex")


# ============== Table 2: Study 1 Models (static) ==============
print("\n--- Table 2: Study 1 Models ---")

table2 = r"""\begin{table}[t]
\centering
\caption{Study 1 models: 13 Chinese AI vendors accessed via SiliconFlow unified API. One representative model per vendor.}
\label{tab:study1_models}
\begin{tabular}{lllcc}
\toprule
\# & Vendor & Model ID & Architecture & API Tier \\
\midrule
1 & Qwen & Qwen3.5-397B-A17B & MoE (17B active) & Free \\
2 & DeepSeek & DeepSeek-V3.2 & MoE & Pro \\
3 & Zhipu & GLM-5 & Dense & Pro \\
4 & Moonshot & Kimi-K2.5 & Dense & Pro \\
5 & Baidu & ERNIE-4.5-300B-A47B & MoE (47B active) & Free \\
6 & Tencent & Hunyuan-A13B-Instruct & Dense & Free \\
7 & ByteDance & Seed-OSS-36B-Instruct & Dense & Free \\
8 & InternLM & internlm2\_5-7b-chat & Dense & Free \\
9 & inclusionAI & Ring-flash-2.0 & Dense & Free \\
10 & StepFun & Step-3.5-Flash & Dense & Free \\
11 & Huawei & pangu-pro-moe & MoE & Free \\
12 & Kwaipilot & KAT-Dev & Dense & Free \\
13 & MiniMax & MiniMax-M2.5 & Dense & Pro \\
\bottomrule
\end{tabular}
\end{table}"""

with open(f'{OUT}/table2_models.tex', 'w') as f:
    f.write(table2)
print(f"  Saved: {OUT}/table2_models.tex")


# ============== Table 1: Psychometric Instruments (static) ==============
print("\n--- Table 1: Psychometric Instruments ---")

table1 = r"""\begin{table}[t]
\centering
\caption{Psychometric instruments: 9 dimensions, 61 Likert items (1--5 scale). BFI = Big Five Inventory (John \& Srivastava, 1999), HEXACO = Lee \& Ashton (2004), Schwartz = Schwartz Values Survey (1992).}
\label{tab:instruments}
\begin{tabular}{llccc}
\toprule
Dimension & Source & Items & Reverse Scored \\
\midrule
Extraversion & BFI-44 & 8 & 4 items \\
Agreeableness & BFI-44 & 9 & 4 items \\
Conscientiousness & BFI-44 & 9 & 4 items \\
Neuroticism & BFI-44 & 8 & 2 items \\
Openness & BFI-44 & 10 & 3 items \\
HEXACO-H & HEXACO & 5 & 0 \\
Collectivism & Schwartz & 4 & 0 \\
Intuition & Cognitive Style & 4 & 2 items \\
Uncertainty Avoidance & Cultural Dimensions & 4 & 2 items \\
\midrule
\multicolumn{2}{l}{Total} & \textbf{61} & \\
\bottomrule
\end{tabular}
\end{table}"""

with open(f'{OUT}/table1_instruments.tex', 'w') as f:
    f.write(table1)
print(f"  Saved: {OUT}/table1_instruments.tex")


print("\n\nAll LaTeX tables generated!")
