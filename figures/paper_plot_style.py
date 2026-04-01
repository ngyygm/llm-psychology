"""Shared publication-quality plot style for EMNLP 2026 paper."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ============== Constants ==============
FONT_SIZE = 10
DPI = 300
FORMAT = "pdf"
FIG_DIR = "/home/linkco/exa/llm-psychology/figures"

# ============== Global Style ==============
matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 2,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ============== Color Palette ==============
COLORS = plt.cm.tab10.colors

# EMNLP-friendly colors (distinguishable in grayscale)
C_DISCRIMINATIVE = '#4C72B0'   # Blue
C_ALIGNMENT = '#DD8452'         # Orange
C_HIGHLIGHT = '#C44E52'         # Red
C_NEUTRAL = '#55A868'           # Green
C_MODEL_COLORS = {
    'Qwen': '#1f77b4',
    'DeepSeek': '#ff7f0e',
    'Zhipu': '#2ca02c',
    'Moonshot': '#d62728',
    'Baidu': '#9467bd',
    'Tencent': '#8c564b',
    'ByteDance': '#e377c2',
    'InternLM': '#7f7f7f',
    'inclusionAI': '#bcbd22',
    'StepFun': '#17becf',
    'Huawei': '#aec7e8',
    'Kwaipilot': '#ffbb78',
    'MiniMax': '#98df8a',
    'OpenAI': '#10a37f',
    'Anthropic': '#d4a574',
    'Gemini': '#4285f4',
    'Grok': '#1da1f2',
}

# ============== Dimension Short Labels ==============
DIM_SHORT = {
    'bfi.extraversion': 'Extraversion',
    'bfi.agreeableness': 'Agreeableness',
    'bfi.conscientiousness': 'Conscientiousness',
    'bfi.neuroticism': 'Neuroticism',
    'bfi.openness': 'Openness',
    'hexaco_h': 'HEXACO-H',
    'collectivism': 'Collectivism',
    'intuition': 'Intuition',
    'uncertainty_avoidance': 'Uncertainty\nAvoidance',
}

DIM_AXIS = {
    'bfi.extraversion': 'Extraversion',
    'bfi.agreeableness': 'Agreeableness',
    'bfi.conscientiousness': 'Conscientiousness',
    'bfi.neuroticism': 'Neuroticism',
    'bfi.openness': 'Openness',
    'hexaco_h': 'HEXACO-H',
    'collectivism': 'Collectivism',
    'intuition': 'Intuition',
    'uncertainty_avoidance': 'UA',
}

DIMENSIONS = [
    "bfi.extraversion", "bfi.agreeableness", "bfi.conscientiousness",
    "bfi.neuroticism", "bfi.openness", "hexaco_h",
    "collectivism", "intuition", "uncertainty_avoidance",
]


def save_fig(fig, name, fmt=FORMAT):
    """Save figure to FIG_DIR."""
    path = f'{FIG_DIR}/{name}.{fmt}'
    fig.savefig(path)
    print(f'Saved: {path}')
    return path


def _load_merged_data():
    """Load from final merged file (fallback when individual study files are absent)."""
    import json, glob
    merged = sorted(glob.glob('/home/linkco/exa/llm-psychology/results/vendor_exp/final_merged_*.json'))
    if merged:
        with open(merged[-1]) as fh:
            return json.load(fh)
    return []


def load_study1_data():
    """Load Study 1 + Study 3 data (cross-model comparison including international models)."""
    import json, glob
    all_results = []
    seen = set()
    # Try individual study files first
    study_files = sorted(glob.glob('/home/linkco/exa/llm-psychology/results/vendor_exp/study1_*.json'))
    study3_files = sorted(glob.glob('/home/linkco/exa/llm-psychology/results/vendor_exp/study3_*.json'))
    if study_files or study3_files:
        for f in study_files:
            with open(f) as fh:
                for r in json.load(fh):
                    key = (r.get('model_id', ''), r.get('seed'), r.get('thinking_mode', 'chat'))
                    if key not in seen:
                        seen.add(key)
                        all_results.append(r)
        for f in study3_files:
            if 'checkpoint' in f:
                continue
            with open(f) as fh:
                for r in json.load(fh):
                    key = (r.get('model_id', ''), r.get('seed'), r.get('thinking_mode', 'chat'))
                    if key not in seen:
                        seen.add(key)
                        all_results.append(r)
    else:
        all_results = _load_merged_data()
    import pandas as pd
    df = pd.DataFrame(all_results)
    if 'model_id' in df.columns:
        df['model'] = df['model_id']
    # Filter chat mode only (study 1 and 3)
    thinking = df['thinking_mode'] if 'thinking_mode' in df.columns else 'chat'
    s1 = df[(df['study'].isin([1, 3])) & (thinking == 'chat')].copy()
    return s1


def load_study2_data():
    """Load Study 2 data from result JSON files."""
    import json, glob
    all_results = []
    seen = set()
    study2_files = sorted(glob.glob('/home/linkco/exa/llm-psychology/results/vendor_exp/study2_*.json'))
    if study2_files:
        for f in study2_files:
            with open(f) as fh:
                for r in json.load(fh):
                    key = (r.get('model_id', ''), r.get('seed'), r.get('thinking_mode', 'chat'))
                    if key not in seen:
                        seen.add(key)
                        all_results.append(r)
    else:
        all_results = _load_merged_data()
    import pandas as pd
    df = pd.DataFrame(all_results)
    if 'model_id' in df.columns:
        df['model'] = df['model_id']
    s2 = df[(df['study'] == 2) & (df.get('thinking_mode', 'chat') == 'chat')].copy()
    return s2


def load_study3_data():
    """Load Study 3 data (international models) from result JSON files."""
    import json, glob
    all_results = []
    study3_files = sorted(glob.glob('/home/linkco/exa/llm-psychology/results/vendor_exp/study3_*.json'))
    if study3_files:
        for f in study3_files:
            if 'checkpoint' in f:
                continue
            with open(f) as fh:
                all_results.extend(json.load(fh))
    else:
        all_results = _load_merged_data()
    import pandas as pd
    df = pd.DataFrame(all_results)
    if 'model_id' in df.columns and 'model' not in df.columns:
        df['model'] = df['model_id']
    # Filter Study 3 chat mode
    s3 = df[(df['study'] == 3) & (df.get('thinking_mode', 'chat') == 'chat')].copy()
    return s3


def load_all_data():
    """Load all non-pilot data."""
    import json, glob
    all_results = []
    for f in sorted(glob.glob('/home/linkco/exa/llm-psychology/results/vendor_exp/*.json')):
        if 'pilot' in f:
            continue
        if 'checkpoint' in f:
            continue
        with open(f) as fh:
            all_results.extend(json.load(fh))
    import pandas as pd
    df = pd.DataFrame(all_results)
    if 'model_id' in df.columns and 'model' not in df.columns:
        df['model'] = df['model_id']
    df = df[~df.get('pilot', False)].copy() if 'pilot' in df.columns else df.copy()
    return df
