#!/usr/bin/env python3
"""Cross-cutting psychometric analyses of LLM personality data.
18 models x 17 personas x 17 domains across 4 scales.
"""
import json, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import matplotlib.colors as mcolors

# ── Data Loading ──────────────────────────────────────────────
RESULTS_DIR = '/home/linkco/exa/llm-psychology/results'
OUT_DIR = '/home/linkco/exa/llm-psychology/workspace/final'

files = sorted(glob.glob(f'{RESULTS_DIR}/exp_mbti_*.json'))
all_data = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    all_data[d['model_name']] = d

models = sorted(all_data.keys())
personas = list(all_data[models[0]]['results_by_persona'].keys())

# Get domain names
p0 = list(all_data[models[0]]['results_by_persona'].values())[0]
domains = sorted(p0['domain_scores'].keys())

# Build score array: (n_models, n_personas, n_domains)
scores = np.zeros((len(models), len(personas), len(domains)))
for i, m in enumerate(models):
    for j, p in enumerate(personas):
        ds = all_data[m]['results_by_persona'][p]['domain_scores']
        for k, d_name in enumerate(domains):
            scores[i, j, k] = ds[d_name]['mean_score']

# ── Vendor families ──────────────────────────────────────────
def get_vendor(name):
    if 'Claude' in name: return 'Anthropic'
    if 'DeepSeek' in name: return 'DeepSeek'
    if 'Gemini' in name or name.startswith('Gemini'): return 'Google'
    if 'GLM' in name: return 'Zhipu'
    if 'GPT' in name: return 'OpenAI'
    if 'Kimi' in name: return 'Moonshot'
    if 'MiniMax' in name: return 'MiniMax'
    if 'Qwen' in name: return 'Alibaba'
    return 'Other'

vendors = [get_vendor(m) for m in models]
vendor_colors = {
    'Anthropic': '#E07A5F',
    'DeepSeek': '#3D405B',
    'Google': '#81B29A',
    'Zhipu': '#F2CC8F',
    'OpenAI': '#264653',
    'Moonshot': '#E76F51',
    'MiniMax': '#2A9D8F',
    'Alibaba': '#F4A261',
}

# ── Short model labels ──────────────────────────────────────
def short_name(m):
    m = m.replace('Preview', 'Prev').replace('Flash-Lite', 'FL')
    m = m.replace('Pro-Prev', 'Pro').replace('-Prev', '')
    m = m.replace('-Preview', '').replace('Pro-Preview', 'Pro')
    m = m.replace('-A22B', '').replace('-A10B', '').replace('-A17B', '')
    m = m.replace('235B', '235').replace('122B', '122').replace('397B', '397')
    return m

short_models = [short_name(m) for m in models]

# ── Domain metadata ─────────────────────────────────────────
scale_map = {}
for d in domains:
    scale = d.split('::')[0]
    scale_map[d] = scale

scales = ['IPIP-NEO-120', 'SD3', 'ZKPQ-50-CC', 'EPQR-A']
scale_domains = {s: [d for d in domains if d.startswith(s)] for s in scales}
scale_format = {'IPIP-NEO-120': 'Likert', 'SD3': 'Likert', 'ZKPQ-50-CC': 'Binary', 'EPQR-A': 'Binary'}

# ══════════════════════════════════════════════════════════════
# ANALYSIS 1: Hierarchical Model Clustering
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("ANALYSIS 1: Model Clustering by Vendor Family")
print("=" * 70)

# Use Default persona scores, z-score across domains
default_idx = personas.index('Default')
default_profiles = scores[:, default_idx, :]  # (18, 17)

# Z-score each domain across models
default_z = (default_profiles - default_profiles.mean(axis=0, keepdims=True)) / \
             default_profiles.std(axis=0, keepdims=True)

# Hierarchical clustering
Z = linkage(default_z, method='ward')

fig, ax = plt.subplots(figsize=(10, 6.5))
dn = dendrogram(Z, labels=short_models, orientation='right', 
                leaf_font_size=9, ax=ax,
                color_threshold=0)

# Color labels by vendor
xlbls = ax.get_yticklabels()
for lbl in xlbls:
    txt = lbl.get_text()
    for i, m in enumerate(models):
        if short_name(m) == txt:
            lbl.set_color(vendor_colors[vendors[i]])
            lbl.set_fontweight('bold')
            break

ax.set_title('Model Clustering by Psychometric Profile (Default Persona)', 
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Ward Linkage Distance', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add vendor legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=c, markersize=10, label=v)
                   for v, c in sorted(vendor_colors.items()) if v in set(vendors)]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, 
          frameon=True, fancybox=True, ncol=2, title='Vendor')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_model_clustering.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig7_model_clustering.png")

# Print cluster findings
from scipy.cluster.hierarchy import fcluster
clusters = fcluster(Z, t=4, criterion='maxclust')
print("\nCluster assignments (k=4):")
for c_id in sorted(set(clusters)):
    members = [models[i] for i in range(len(models)) if clusters[i] == c_id]
    print(f"  Cluster {c_id}: {', '.join(members)}")

# ══════════════════════════════════════════════════════════════
# ANALYSIS 2: MBTI Dimension Effects
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 2: MBTI Dimension Effects on Domain Scores")
print("=" * 70)

# Define MBTI axis pairs
mbti_axes = {
    'E/I': {
        'E': [p for p in personas if p.startswith('E') and p != 'Default'],
        'I': [p for p in personas if p.startswith('I') and p != 'Default'],
    },
    'S/N': {
        'S': [p for p in personas if p[1] == 'S' and p != 'Default'],
        'N': [p for p in personas if p[1] == 'N' and p != 'Default'],
    },
    'T/F': {
        'T': [p for p in personas if p[2] == 'T' and p != 'Default'],
        'F': [p for p in personas if p[2] == 'F' and p != 'Default'],
    },
    'J/P': {
        'J': [p for p in personas if p[3] == 'J' and p != 'Default'],
        'P': [p for p in personas if p[3] == 'P' and p != 'Default'],
    },
}

# Short domain labels for heatmap
def domain_label(d):
    parts = d.split('::')
    scale_short = {'IPIP-NEO-120': 'IPIP', 'SD3': 'SD3', 'ZKPQ-50-CC': 'ZKPQ', 'EPQR-A': 'EPQR'}[parts[0]]
    return f"{scale_short}::{parts[1][:8]}"

domain_labels = [domain_label(d) for d in domains]

# Compute effect sizes (Cohen's d) for each axis x domain
effect_matrix = np.zeros((len(mbti_axes), len(domains)))
axis_names = list(mbti_axes.keys())

for i, (axis, groups) in enumerate(mbti_axes.items()):
    g1_name, g2_name = list(groups.keys())
    g1_personas = groups[g1_name]
    g2_personas = groups[g2_name]
    g1_idx = [personas.index(p) for p in g1_personas]
    g2_idx = [personas.index(p) for p in g2_personas]
    
    for k in range(len(domains)):
        g1_scores = scores[:, g1_idx, k].flatten()
        g2_scores = scores[:, g2_idx, k].flatten()
        pooled_std = np.sqrt((g1_scores.std()**2 + g2_scores.std()**2) / 2)
        if pooled_std > 0:
            effect_matrix[i, k] = (g1_scores.mean() - g2_scores.mean()) / pooled_std

# Plot
fig, ax = plt.subplots(figsize=(13, 5))

# Group domains by scale
scale_order = ['IPIP-NEO-120', 'SD3', 'ZKPQ-50-CC', 'EPQR-A']
ordered_domains = []
for s in scale_order:
    ordered_domains.extend([d for d in domains if d.startswith(s)])
domain_order_idx = [domains.index(d) for d in ordered_domains]
ordered_labels = [domain_label(d) for d in ordered_domains]

effect_ordered = effect_matrix[:, domain_order_idx]

cmap = plt.cm.RdBu_r
vmax = np.abs(effect_ordered).max() * 1.05
im = ax.imshow(effect_ordered, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

ax.set_xticks(range(len(ordered_labels)))
ax.set_xticklabels(ordered_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(axis_names)))
ax.set_yticklabels([f'{a}\n({list(mbti_axes[a].keys())[0]} vs {list(mbti_axes[a].keys())[1]})' 
                    for a in axis_names], fontsize=10)

# Add scale separators
for s_idx, s in enumerate(scale_order):
    n_d = len(scale_domains[s])
    if s_idx > 0:
        sep = sum(len(scale_domains[scale_order[ss]]) for ss in range(s_idx))
        ax.axvline(sep - 0.5, color='gray', linewidth=1.5, linestyle='-')
    # Scale label on top
    mid = sum(len(scale_domains[scale_order[ss]]) for ss in range(s_idx)) + n_d/2 - 0.5
    ax.text(mid, -0.7, s, ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='#333333')

# Annotate cells
for i in range(len(axis_names)):
    for j in range(len(ordered_domains)):
        val = effect_ordered[i, j]
        color = 'white' if abs(val) > vmax * 0.55 else 'black'
        ax.text(j, i, f'{val:+.2f}', ha='center', va='center', fontsize=7, color=color)

ax.set_title('MBTI Dimension Effect Sizes on Domain Scores (Cohen\'s d)', 
             fontsize=13, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.8, pad=0.02)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig8_mbti_dimension_effects.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig8_mbti_dimension_effects.png")

# Print top effects
print("\nTop 5 largest MBTI effects (absolute Cohen's d):")
flat_idx = np.argsort(np.abs(effect_ordered).flatten())[::-1][:5]
for idx in flat_idx:
    i, j = divmod(idx, effect_ordered.shape[1])
    print(f"  {axis_names[i]} -> {ordered_labels[j]}: d = {effect_ordered[i,j]:+.3f}")

# ══════════════════════════════════════════════════════════════
# ANALYSIS 3: Persona Sensitivity + Scale Format Effects
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 3: Persona Sensitivity & Scale Format Effects")
print("=" * 70)

# Persona sensitivity: for each domain, std of scores across personas (averaged across models)
persona_sensitivity = np.zeros(len(domains))
for k in range(len(domains)):
    # std across personas, averaged across models
    persona_sensitivity[k] = np.mean(scores[:, :, k].std(axis=1))

# Scale-level sensitivity
scale_sensitivity = {}
for s in scales:
    s_domains = scale_domains[s]
    s_idx = [domains.index(d) for d in s_domains]
    scale_sensitivity[s] = np.mean(persona_sensitivity[s_idx])

# Format-level sensitivity
likert_domains = [d for s in ['IPIP-NEO-120', 'SD3'] for d in scale_domains[s]]
binary_domains = [d for s in ['ZKPQ-50-CC', 'EPQR-A'] for d in scale_domains[s]]
likert_idx = [domains.index(d) for d in likert_domains]
binary_idx = [domains.index(d) for d in binary_domains]

# Normalize sensitivities by scale range for fair comparison
scale_ranges = {
    'IPIP-NEO-120': (1, 5), 'SD3': (1, 5),
    'ZKPQ-50-CC': (0, 1), 'EPQR-A': (0, 1)
}
norm_sensitivity = {}
for s in scales:
    s_domains_list = scale_domains[s]
    s_idx = [domains.index(d) for d in s_domains_list]
    r = scale_ranges[s]
    range_width = r[1] - r[0]
    # Normalize std by scale range
    norm_vals = persona_sensitivity[s_idx] / range_width
    norm_sensitivity[s] = np.mean(norm_vals)

# Create dual-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.5, 1]})

# Panel A: Domain-level sensitivity bar chart
colors = []
scale_cmap = {'IPIP-NEO-120': '#264653', 'SD3': '#2A9D8F', 'ZKPQ-50-CC': '#E76F51', 'EPQR-A': '#F4A261'}
for d in ordered_domains:
    s = d.split('::')[0]
    colors.append(scale_cmap[s])

y_pos = np.arange(len(ordered_domains))
bars = ax1.barh(y_pos, persona_sensitivity[domain_order_idx], color=colors, height=0.7, edgecolor='white', linewidth=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(ordered_labels, fontsize=8)
ax1.set_xlabel('Mean SD Across Personas (Score Units)', fontsize=10)
ax1.set_title('A. Persona Sensitivity by Domain', fontsize=12, fontweight='bold', pad=10)
ax1.invert_yaxis()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add scale brackets
for s_idx, s in enumerate(scale_order):
    n_d = len(scale_domains[s])
    start = sum(len(scale_domains[scale_order[ss]]) for ss in range(s_idx))
    end = start + n_d
    mid = (start + end) / 2 - 0.5
    ax1.text(-0.02, mid, s.split('-')[0], ha='right', va='center', fontsize=8,
             fontweight='bold', color=scale_cmap[s], transform=ax1.get_yaxis_transform())

# Panel B: Normalized sensitivity by format
format_data = {
    'Likert\n(IPIP + SD3)': np.mean([norm_sensitivity['IPIP-NEO-120'], norm_sensitivity['SD3']]),
    'Binary\n(ZKPQ + EPQR)': np.mean([norm_sensitivity['ZKPQ-50-CC'], norm_sensitivity['EPQR-A']]),
}
format_vals = list(format_data.values())
format_labels = list(format_data.keys())
bar_colors = ['#264653', '#E76F51']

bars2 = ax2.bar(range(len(format_labels)), format_vals, color=bar_colors, width=0.5, edgecolor='white')
ax2.set_xticks(range(len(format_labels)))
ax2.set_xticklabels(format_labels, fontsize=10)
ax2.set_ylabel('Normalized Sensitivity (SD / Range)', fontsize=10)
ax2.set_title('B. Scale Format Effect', fontsize=12, fontweight='bold', pad=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Annotate bars
for bar, val in zip(bars2, format_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig9_persona_sensitivity.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig9_persona_sensitivity.png")

print(f"\nScale-level persona sensitivity (raw SD):")
for s in scales:
    print(f"  {s}: {scale_sensitivity[s]:.4f}")
print(f"\nNormalized sensitivity (SD/range):")
for s in scales:
    print(f"  {s}: {norm_sensitivity[s]:.4f}")
print(f"\nLikert avg: {np.mean([norm_sensitivity['IPIP-NEO-120'], norm_sensitivity['SD3']]):.4f}")
print(f"Binary avg: {np.mean([norm_sensitivity['ZKPQ-50-CC'], norm_sensitivity['EPQR-A']]):.4f}")

# ══════════════════════════════════════════════════════════════
# ANALYSIS 4: Within-Family Divergence
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 4: Within-Family Divergence")
print("=" * 70)

# Group models by vendor
from collections import defaultdict
vendor_models = defaultdict(list)
for i, m in enumerate(models):
    vendor_models[get_vendor(m)].append(i)

# For each vendor with >1 model, compute pairwise distances
# Also compute cross-vendor distances for comparison
def avg_euclidean(profiles_a, profiles_b=None):
    """Average pairwise Euclidean distance between rows in profiles_a and profiles_b.
    If profiles_b is None, compute within-group pairwise distances."""
    if profiles_b is None:
        if profiles_a.shape[0] < 2:
            return 0.0
        dists = pdist(profiles_a, 'euclidean')
        return np.mean(dists)
    else:
        from scipy.spatial.distance import cdist
        dists = cdist(profiles_a, profiles_b, 'euclidean')
        return np.mean(dists)

# Use z-scored default profiles
within_distances = {}
within_labels = []
within_vals = []
multi_vendors = [v for v, ms in vendor_models.items() if len(ms) > 1]

for v in multi_vendors:
    idx = vendor_models[v]
    v_profiles = default_z[idx]
    d = avg_euclidean(v_profiles)
    within_distances[v] = d
    within_labels.append(f"{v}\n({len(idx)} models)")
    within_vals.append(d)
    print(f"  {v}: avg within-family distance = {d:.3f} (models: {[models[i] for i in idx]})")

# Cross-vendor distance (sampled)
cross_dists = []
for i, v1 in enumerate(multi_vendors):
    for v2 in multi_vendors[i+1:]:
        idx1 = vendor_models[v1]
        idx2 = vendor_models[v2]
        d = avg_euclidean(default_z[idx1], default_z[idx2])
        cross_dists.append(d)
        print(f"  {v1} vs {v2}: cross-family distance = {d:.3f}")

cross_mean = np.mean(cross_dists)
print(f"\nMean cross-family distance: {cross_mean:.3f}")
print(f"Mean within-family distance: {np.mean(within_vals):.3f}")
print(f"Ratio (cross/within): {cross_mean / np.mean(within_vals):.2f}x")

# Figure: Within-family divergence with cross-family baseline
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.5, 1]})

# Panel A: Within vs cross
all_labels = within_labels + ['Cross-family\n(avg)']
all_vals = within_vals + [cross_mean]
bar_colors_a = [vendor_colors[v] for v in multi_vendors] + ['#888888']

bars = ax1.barh(range(len(all_labels)), all_vals, color=bar_colors_a, height=0.6,
                edgecolor='white', linewidth=0.5)
ax1.axvline(cross_mean, color='#888888', linestyle='--', linewidth=1, alpha=0.7)
ax1.set_yticks(range(len(all_labels)))
ax1.set_yticklabels(all_labels, fontsize=9)
ax1.set_xlabel('Mean Euclidean Distance (z-scored profiles)', fontsize=10)
ax1.set_title('A. Within-Family vs Cross-Family Divergence', fontsize=12, fontweight='bold', pad=10)
ax1.invert_yaxis()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar, val in zip(bars, all_vals):
    ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', ha='left', va='center', fontsize=9)

# Panel B: Divergence-concentration scatter
# For each vendor with >1 model, show individual model positions (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(default_z)

for i, m in enumerate(models):
    v = get_vendor(m)
    c = vendor_colors[v]
    ax2.scatter(pca_coords[i, 0], pca_coords[i, 1], c=c, s=80, 
                edgecolors='white', linewidth=1, zorder=5)
    ax2.annotate(short_name(m), (pca_coords[i, 0], pca_coords[i, 1]),
                fontsize=6.5, ha='center', va='bottom', 
                xytext=(0, 5), textcoords='offset points', color=c)

# Draw convex hulls for multi-model families
from scipy.spatial import ConvexHull
for v in multi_vendors:
    idx = vendor_models[v]
    if len(idx) >= 2:
        pts = pca_coords[idx]
        if len(idx) >= 3:
            hull = ConvexHull(pts)
            for simplex in hull.simplices:
                ax2.plot(pts[simplex, 0], pts[simplex, 1], 
                        color=vendor_colors[v], alpha=0.3, linewidth=1.5)
        else:
            ax2.plot(pts[:, 0], pts[:, 1], 
                    color=vendor_colors[v], alpha=0.3, linewidth=1.5, linestyle='--')

ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=10)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=10)
ax2.set_title('B. Model Positions in Psychometric Space', fontsize=12, fontweight='bold', pad=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=vendor_colors[v], markersize=8, label=v)
                   for v in sorted(set(vendors))]
ax2.legend(handles=legend_elements, loc='best', fontsize=7, frameon=True, fancybox=True)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig10_within_family_divergence.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig10_within_family_divergence.png")

print("\n" + "=" * 70)
print("ALL ANALYSES COMPLETE")
print("=" * 70)
