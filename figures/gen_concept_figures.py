"""Generate concept figure and pipeline figure for the paper."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# Figure 1: Core Concept — same item, different responses
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.2))

# Simulated response distributions for 3 models
np.random.seed(42)
models = ["Model A\n(ERNIE)", "Model B\n(Qwen)", "Model C\n(Hunyuan)"]
means = [1.0, 2.8, 4.2]
stds = [0.3, 0.8, 0.6]
colors = ['#4A9EDA', '#E8744A', '#74AA63']
item = '"I would buy stolen goods\nif I were sure I would\nnot get caught."'

for ax, mean, std, color, model in zip(axes, means, stds, colors, models):
    # Likert scale 1-5
    x = np.array([1, 2, 3, 4, 5])
    # Simulate distribution centered on mean
    probs = np.exp(-0.5 * ((x - mean) / std) ** 2)
    probs /= probs.sum()

    bars = ax.bar(x, probs, color=color, alpha=0.7, edgecolor='white', width=0.7)
    ax.set_xlim(0.3, 5.7)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1\nStrongly\nDisagree", "2\nDisagree", "3\nNeutral",
                          "4\nAgree", "5\nStrongly\nAgree"], fontsize=5.5)
    ax.set_ylabel("Probability", fontsize=7)
    ax.set_title(model, fontsize=8, fontweight='bold', color=color)
    ax.tick_params(axis='y', labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(f'HEXACO-H Item: {item}', fontsize=7, y=1.02, style='italic')
fig.text(0.5, -0.05, 'Same item → Different response distributions → Model-specific response style signatures',
         ha='center', fontsize=7, style='italic', color='#555')
plt.tight_layout()
plt.savefig('intro_figure_prompt.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', pad_inches=0.1)
plt.close()
print("Saved intro_figure_prompt.pdf")


# ============================================================
# Figure 2: Experimental Pipeline
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 2.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

# Pipeline stages
stages = [
    (1.0, 2.0, "15 Model\nFamilies", "#4A9EDA"),
    (3.0, 2.0, "61 Likert Items\n9 Dimensions", "#E8744A"),
    (5.0, 2.0, "12 Seeds Each\n10,980 Calls", "#74AA63"),
    (7.0, 2.0, "Statistical\nAnalysis", "#9B59B6"),
    (9.0, 2.0, "Results &\nFindings", "#E74C3C"),
]

for x, y, text, color in stages:
    rect = mpatches.FancyBboxPatch((x - 0.7, y - 0.6), 1.4, 1.2,
                                     boxstyle="round,pad=0.1",
                                     facecolor=color, alpha=0.15,
                                     edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=6.5, fontweight='bold', color=color)

# Arrows between stages
for i in range(len(stages) - 1):
    x1 = stages[i][0] + 0.7
    x2 = stages[i + 1][0] - 0.7
    y = 2.0
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))

# Sub-labels below
sublabels = [
    (1.0, 0.8, "SiliconFlow + YiHe\nUnified API", "#666"),
    (3.0, 0.8, "BFI-44 + HEXACO-H\nSchwartz + Cognitive", "#666"),
    (5.0, 0.8, "Temperature=0.7\nExponential Backoff", "#666"),
    (7.0, 0.8, "ANOVA · OLR · Cohen's d\nFDR · ICC · PCA", "#666"),
    (9.0, 0.8, "15 families, 9 dims\nConvergent validity", "#666"),
]

for x, y, text, color in sublabels:
    ax.text(x, y, text, ha='center', va='center', fontsize=5, color=color, style='italic')

# Analysis modules detail at bottom
modules = ["ANOVA", "OLR", "Cohen's d", "FDR", "ICC", "PCA", "Convergent\nValidity", "Acquiescence\nCorrection"]
mod_y = 0.0
mod_x_start = 2.0
mod_spacing = 0.85
for i, mod in enumerate(modules):
    x = mod_x_start + i * mod_spacing
    rect = mpatches.FancyBboxPatch((x - 0.35, mod_y - 0.25), 0.7, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#f0f0f0", edgecolor="#ccc", linewidth=0.5)
    ax.add_patch(rect)
    ax.text(x, mod_y, mod, ha='center', va='center', fontsize=4.5, color="#555")

ax.text(5.0, -0.45, "Analysis Modules", ha='center', fontsize=6, color="#888", style='italic')

plt.tight_layout()
plt.savefig('framework_figure_prompt.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', pad_inches=0.1)
plt.close()
print("Saved framework_figure_prompt.pdf")
