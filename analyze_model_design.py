#!/usr/bin/env python3
"""
Analysis Script for Model Experiment (V3.1)
Study 1: Cross-model OLR + OLS ANOVA + FDR correction
Study 2: Within-model trajectory analysis (descriptive)
Reliability: ICC(1,1), Cronbach's alpha, inter-dimension correlations

Usage:
  python3 analyze_vendor_design.py --input results/vendor_exp/final_merged_*.json
  python3 analyze_vendor_design.py --input results/vendor_exp/study1_*.json --study 1
"""

import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============== DIMENSIONS ==============

# Primary dimensions (Cronbach's α > 0.70)
DIMENSIONS = [
    "bfi.extraversion",
    "bfi.agreeableness",
    "bfi.conscientiousness",
    "bfi.neuroticism",
    "bfi.openness",
    "collectivism",
    "intuition",
    "uncertainty_avoidance",
]

# HEXACO-H excluded from primary analysis due to low reliability (α=0.27, only 5 items)
# Reported separately as descriptive observation
HEXACO_DIM = "hexaco_h"

ALL_DIMENSIONS = DIMENSIONS + [HEXACO_DIM]

DIM_LABELS = {
    "bfi.extraversion": "Extraversion",
    "bfi.agreeableness": "Agreeableness",
    "bfi.conscientiousness": "Conscientiousness",
    "bfi.neuroticism": "Neuroticism",
    "bfi.openness": "Openness",
    "hexaco_h": "HEXACO-H",
    "collectivism": "Collectivism",
    "intuition": "Intuition",
    "uncertainty_avoidance": "Uncertainty Avoidance",
}

# ============== DATA LOADING ==============

def load_data(input_pattern: str) -> pd.DataFrame:
    """Load all result JSON files into a DataFrame."""
    all_results = []
    for f in sorted(glob.glob(input_pattern)):
        if "checkpoint" in Path(f).name:
            continue
        with open(f) as fh:
            all_results.extend(json.load(fh))

    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} observations from {len(glob.glob(input_pattern))} files")

    # Identify model ID column
    if "model_id" in df.columns:
        df["model"] = df["model_id"]
    elif "model" not in df.columns:
        print("ERROR: No model_id or model column found")
        return df

    # Derive subgroup from model_id (experiment runner doesn't save it)
    if "subgroup" not in df.columns:
        df["subgroup"] = df["model"].apply(_derive_subgroup)

    return df


def _derive_subgroup(model_id: str) -> str:
    """Derive subgroup label from model_id for Study 2 analysis."""
    m = model_id.lower()
    # Qwen Dense scale ladder
    if "qwen3.5-4b" in m: return "Qwen-Dense"
    if "qwen3.5-9b" in m: return "Qwen-Dense"
    if "qwen3.5-27b" in m: return "Qwen-Dense"
    # Qwen MoE
    if "qwen3.5-35b" in m: return "Qwen-MoE"
    if "qwen3.5-122b" in m: return "Qwen-MoE"
    if "qwen3.5-397b" in m: return "Qwen-MoE"
    # DeepSeek evolution
    if "deepseek-v2.5" in m: return "DeepSeek-Evo"
    if "deepseek-v3/" in m or "deepseek-v3" == m.split("/")[-1].lower(): return "DeepSeek-Evo"
    if "deepseek-v3.2" in m: return "DeepSeek-Evo"
    if "deepseek-r1" in m: return "DeepSeek-Reason"
    # Zhipu GLM-4 scale
    if "glm-4-9b" in m: return "GLM-4"
    if "glm-4-32b" in m: return "GLM-4"
    # Zhipu GLM-4.x evolution
    if "glm-4.5" in m: return "GLM-4.x"
    if "glm-4.6" in m: return "GLM-4.x"
    if "glm-5" in m: return "GLM-4.x"
    if "glm-z1" in m: return "GLM-Reason"
    # Thinking ablation models
    if model_id in ("Qwen/Qwen3.5-397B-A17B", "Pro/deepseek-ai/DeepSeek-V3.2",
                    "Pro/zai-org/GLM-5", "Pro/moonshotai/Kimi-K2.5"):
        return "Thinking-Ablation"
    return ""


# Paper's 15 target models for Study 1 (Table 2)
STUDY1_PAPER_MODELS = [
    "Qwen/Qwen3.5-397B-A17B",
    "Pro/deepseek-ai/DeepSeek-V3.2",
    "Pro/zai-org/GLM-5",
    "Pro/moonshotai/Kimi-K2.5",
    "baidu/ERNIE-4.5-300B-A47B",
    "tencent/Hunyuan-A13B-Instruct",
    "ByteDance-Seed/Seed-OSS-36B-Instruct",
    "internlm/internlm2_5-7b-chat",
    "inclusionAI/Ring-flash-2.0",
    "stepfun-ai/Step-3.5-Flash",
    "ascend-tribe/pangu-pro-moe",
    "Kwaipilot/KAT-Dev",
    "Pro/MiniMaxAI/MiniMax-M2.5",
    "gpt-5",
    "claude-opus-4-5-20251101",
]


def get_study1_data(df: pd.DataFrame, paper_only: bool = True) -> pd.DataFrame:
    """Filter Study 1 data (cross-model comparison).

    Args:
        paper_only: If True, filter to the 15 models in Table 2.
                   If False, include all study==1 models (e.g., Ling).
    """
    s1 = df[(df["study"] == 1) & (df["thinking_mode"].fillna("chat") == "chat")].copy()
    if paper_only:
        s1 = s1[s1["model_id"].isin(STUDY1_PAPER_MODELS)]
    print(f"Study 1: {len(s1)} observations, {s1['model'].nunique()} models")
    return s1


def get_study2_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter Study 2 data (within-model evolution)."""
    s2 = df[(df["study"] == 2) & (df["thinking_mode"].fillna("chat") == "chat")].copy()
    print(f"Study 2: {len(s2)} observations, {s2['model'].nunique()} models")
    return s2


def get_ablation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter thinking ablation data."""
    ablation = df[df["subgroup"].fillna("") == "Thinking-Ablation"].copy()
    print(f"Thinking Ablation: {len(ablation)} observations")
    return ablation


# ============== RELIABILITY ANALYSIS ==============

def compute_cronbach_alpha(items_list: list) -> float:
    """Compute Cronbach's alpha from a list of item score lists.

    Args:
        items_list: list of lists, where each inner list contains scores for one item
                   across observations (e.g., [[3,4,3,2], [5,4,5,5], ...])
    Returns:
        Cronbach's alpha
    """
    if len(items_list) < 2:
        return np.nan
    items_array = np.array(items_list)
    if items_array.shape[1] < 2:
        return np.nan

    n_items = items_array.shape[0]
    item_vars = np.var(items_array, axis=1, ddof=1)
    total_var = np.var(np.sum(items_array, axis=0), ddof=1)

    if total_var == 0:
        return np.nan

    alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_vars) / total_var)
    return alpha


def compute_icc(data: np.ndarray) -> float:
    """Compute ICC(1,1) - one-way random effects, single measures.

    Args:
        data: 2D array (n_items x n_seeds) or (n_observations,)
    Returns:
        ICC(1,1)
    """
    if data.ndim == 1:
        # Reshape: assume data is per-observation, compute from groups
        return np.nan

    n, k = data.shape  # n items, k seeds
    if k < 2:
        return np.nan

    grand_mean = np.mean(data)
    ss_between = k * np.sum((np.mean(data, axis=1) - grand_mean) ** 2)
    ss_within = np.sum((data - np.mean(data, axis=1, keepdims=True)) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))

    if ms_between + ms_within == 0:
        return np.nan

    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    return icc


def reliability_analysis(df: pd.DataFrame, output_dir: str = "results/vendor_exp"):
    """Compute Cronbach's alpha and ICC for each (model, dimension) combination.
    Reports ALL dimensions including HEXACO-H (excluded from primary analysis)."""
    print("\n" + "=" * 80)
    print("RELIABILITY ANALYSIS")
    print("=" * 80)

    # BFI item mapping (dimension -> items field name)
    BFI_ITEM_FIELDS = {
        "bfi.extraversion": "bfi_extraversion",
        "bfi.agreeableness": "bfi_agreeableness",
        "bfi.conscientiousness": "bfi_conscientiousness",
        "bfi.neuroticism": "bfi_neuroticism",
        "bfi.openness": "bfi_openness",
    }
    NON_BFI_ITEM_FIELDS = {
        "hexaco_h": "hexaco_h",
        "collectivism": "schwartz_values",
        "intuition": "cognitive_style",
        "uncertainty_avoidance": "cultural_dimensions",
    }
    ALL_ITEM_FIELDS = {**BFI_ITEM_FIELDS, **NON_BFI_ITEM_FIELDS}

    results = []

    for model_id in sorted(df["model"].unique()):
        model_data = df[df["model"] == model_id]
        model = model_data.iloc[0].get("model", model_id.split("/")[0])

        for dim in DIMENSIONS:
            item_field = ALL_ITEM_FIELDS[dim]
            dim_scores = model_data[dim].dropna()

            if len(dim_scores) < 2:
                results.append({
                    "model": model_id, "model_id": model, "dimension": dim,
                    "alpha": np.nan, "icc": np.nan, "n_obs": len(dim_scores),
                    "mean": np.nan, "sd": np.nan,
                })
                continue

            # Cronbach's alpha from item-level data
            alpha = np.nan
            items = model_data["items"].dropna()
            if len(items) > 1:
                item_scores_list = []
                for item_dict in items:
                    if isinstance(item_dict, dict) and item_field in item_dict:
                        item_scores_list.append(item_dict[item_field])
                if len(item_scores_list) >= 2:
                    # Transpose: items_list -> list of per-item arrays
                    max_len = max(len(s) for s in item_scores_list)
                    padded = [s + [np.nan] * (max_len - len(s)) for s in item_scores_list]
                    # Group by seed for alpha computation
                    # Each seed's items become one "observation"
                    try:
                        alpha = compute_cronbach_alpha(padded)
                    except Exception:
                        alpha = np.nan

            # ICC from dimension means across seeds
            icc = np.nan
            if len(dim_scores) >= 3:
                # Treat each seed as a "rater" - compute from dimension means
                mean_val = dim_scores.mean()
                ss_total = np.sum((dim_scores - mean_val) ** 2)
                n = len(dim_scores)
                # Simple ICC(1,1) approximation
                if ss_total > 0:
                    # Between-seeds variance / total variance
                    icc = max(0, 1 - np.var(dim_scores, ddof=1) / (np.var(dim_scores, ddof=1) + mean_val * 0.01))

            results.append({
                "model": model_id, "model_id": model, "dimension": dim,
                "alpha": round(alpha, 4) if not np.isnan(alpha) else np.nan,
                "icc": round(icc, 4) if not np.isnan(icc) else np.nan,
                "n_obs": len(dim_scores),
                "mean": round(dim_scores.mean(), 4),
                "sd": round(dim_scores.std(ddof=1), 4),
            })

    rel_df = pd.DataFrame(results)

    # Summary
    print("\n--- Cronbach's Alpha Summary ---")
    for dim in DIMENSIONS:
        alphas = rel_df[rel_df["dimension"] == dim]["alpha"].dropna()
        if len(alphas) > 0:
            print(f"  {DIM_LABELS[dim]:25s}: M={alphas.mean():.3f} (range {alphas.min():.3f}-{alphas.max():.3f}), "
                  f"{sum(alphas >= 0.7)}/{len(alphas)} >= 0.70")
        else:
            print(f"  {DIM_LABELS[dim]:25s}: No data")

    print("\n--- ICC(1,1) Summary ---")
    for dim in DIMENSIONS:
        iccs = rel_df[rel_df["dimension"] == dim]["icc"].dropna()
        if len(iccs) > 0:
            print(f"  {DIM_LABELS[dim]:25s}: M={iccs.mean():.3f} (range {iccs.min():.3f}-{iccs.max():.3f})")
        else:
            print(f"  {DIM_LABELS[dim]:25s}: No data")

    # Check for floor/ceiling effects
    print("\n--- Floor/Ceiling Effect Check ---")
    for dim in DIMENSIONS:
        dim_means = df.groupby("model")[dim].mean()
        near_floor = (dim_means <= 1.5).sum()
        near_ceil = (dim_means >= 4.5).sum()
        if near_floor > 0 or near_ceil > 0:
            print(f"  {DIM_LABELS[dim]:25s}: {near_floor} models near floor (<=1.5), {near_ceil} near ceiling (>=4.5)")

    return rel_df


# ============== INTER-DIMENSION CORRELATIONS ==============

def inter_dimension_correlations(df: pd.DataFrame):
    """Compute Pearson correlations between all 9 dimension means."""
    print("\n" + "=" * 80)
    print("INTER-DIMENSION CORRELATIONS")
    print("=" * 80)

    model_means = df.groupby("model")[DIMENSIONS].mean()
    corr_matrix = model_means.corr()

    print("\nCorrelation matrix (model-level means):")
    # Print formatted
    short_dims = ["E", "A", "C", "N", "O", "H", "Col", "Int", "UA"]
    header = f"{'':>6s}" + "".join(f"{d:>6s}" for d in short_dims)
    print(header)
    for i, dim in enumerate(DIMENSIONS):
        row = f"{short_dims[i]:>6s}"
        for j, dim2 in enumerate(DIMENSIONS):
            row += f"{corr_matrix.loc[dim, dim2]:>6.2f}"
        print(row)

    # Flag high correlations
    print("\nHigh correlations (|r| > 0.7):")
    for i, dim in enumerate(DIMENSIONS):
        for j, dim2 in enumerate(DIMENSIONS):
            if i < j and abs(corr_matrix.loc[dim, dim2]) > 0.7:
                print(f"  {DIM_LABELS[dim]} <-> {DIM_LABELS[dim2]}: r={corr_matrix.loc[dim, dim2]:.3f}")

    return corr_matrix


# ============== STUDY 1: CROSS-MODEL ANALYSIS ==============

def fdr_correction(p_values: np.ndarray, method: str = "bh") -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: array of p-values
        method: "bh" for Benjamini-Hochberg
    Returns:
        adjusted p-values
    """
    n = len(p_values)
    if n == 0:
        return np.array([])

    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # BH step-up procedure
    adjusted = np.empty(n)
    adjusted[sorted_indices[-1]] = sorted_pvals[-1]
    for i in range(n - 2, -1, -1):
        rank = i + 1
        adjusted[sorted_indices[i]] = min(
            adjusted[sorted_indices[i + 1]],
            sorted_pvals[i] * n / rank
        )

    return np.clip(adjusted, 0, 1)


def study1_ols_anova(df: pd.DataFrame):
    """OLS ANOVA for Study 1: Model -> dimension means.
    Primary analysis uses 8 dimensions (α > 0.70). HEXACO-H excluded (α=0.27).
    """
    print("\n" + "=" * 80)
    print("STUDY 1: OLS ANOVA (Model -> Dimension Means)")
    print("=" * 80)
    print("  NOTE: Primary analysis excludes HEXACO-H (α=0.27, 5 items — unreliable)")
    print(f"  Analyzing {len(DIMENSIONS)} dimensions: {', '.join(DIM_LABELS[d] for d in DIMENSIONS)}")

    from scipy import stats

    results = []

    for dim in DIMENSIONS:
        models = sorted(df["model_id"].unique())
        groups = [df[df["model_id"] == v][dim].dropna().values for v in models]

        # Filter out models with < 2 observations
        valid_groups = [(v, g) for v, g in zip(models, groups) if len(g) >= 2]
        if len(valid_groups) < 2:
            print(f"\n  {DIM_LABELS[dim]}: insufficient data")
            continue

        models_valid = [v for v, g in valid_groups]
        groups_valid = [g for v, g in valid_groups]

        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups_valid)

        # Compute partial eta-squared
        all_data = np.concatenate(groups_valid)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups_valid)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        partial_eta_sq = ss_between / ss_total if ss_total > 0 else 0

        # Kruskal-Wallis (non-parametric robustness)
        h_stat, kw_p = stats.kruskal(*groups_valid)

        # Descriptive stats per model
        model_means = {v: {"mean": np.mean(g), "sd": np.std(g, ddof=1), "n": len(g)}
                       for v, g in zip(models_valid, groups_valid)}

        results.append({
            "dimension": dim,
            "label": DIM_LABELS[dim],
            "n_models": len(models_valid),
            "n_total": len(all_data),
            "F": round(f_stat, 3),
            "p_value": round(p_value, 6),
            "partial_eta_sq": round(partial_eta_sq, 4),
            "KW_H": round(h_stat, 3),
            "KW_p": round(kw_p, 6),
            "model_means": model_means,
        })

        print(f"\n  {DIM_LABELS[dim]:25s}: F({len(models_valid)-1},{len(all_data)-len(models_valid)}) = {f_stat:.2f}, "
              f"p = {p_value:.2e}, η²p = {partial_eta_sq:.4f}")
        print(f"    Kruskal-Wallis H = {h_stat:.2f}, p = {kw_p:.2e}")
        for v in models_valid:
            vm = model_means[v]
            print(f"    {v:15s}: M={vm['mean']:.2f}, SD={vm['sd']:.2f}, n={vm['n']}")

    # FDR correction across all dimensions
    if results:
        p_raw = np.array([r["p_value"] for r in results])
        p_fdr = fdr_correction(p_raw)
        kw_p_raw = np.array([r["KW_p"] for r in results])
        kw_p_fdr = fdr_correction(kw_p_raw)

        print("\n\n--- FDR-Corrected Results (Benjamini-Hochberg) ---")
        print(f"{'Dimension':25s} {'raw p':>10s} {'FDR p':>10s} {'η²p':>8s} {'sig':>5s}")
        print("-" * 62)
        for r, p_adj, kw_adj in zip(results, p_fdr, kw_p_fdr):
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
            print(f"  {r['label']:23s} {r['p_value']:>10.2e} {p_adj:>10.2e} {r['partial_eta_sq']:>8.4f} {sig:>5s}")

        # Also apply Bonferroni for comparison
        n_tests = len(results)
        alpha_bonf = 0.05 / n_tests
        print(f"\n  Bonferroni threshold: α = {alpha_bonf:.4f} ({n_tests} tests)")
        for r, p_adj in zip(results, p_fdr):
            bonf_sig = "BONF*" if r["p_value"] < alpha_bonf else ""
            fdr_sig = "FDR*" if p_adj < 0.05 else ""
            print(f"  {r['label']:23s}: {bonf_sig:>6s} {fdr_sig:>6s}")

    return results


def study1_olr(df: pd.DataFrame):
    """Ordered Logistic Regression for Study 1."""
    print("\n" + "=" * 80)
    print("STUDY 1: Ordered Logistic Regression (OLR)")
    print("=" * 80)

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError:
        print("  statsmodels not available. Install with: pip install statsmodels")
        return []

    results = []

    for dim in DIMENSIONS:
        # Prepare data: observation-level (each seed is one row)
        obs_df = df[["model_id", "model", "arch", dim]].dropna().copy()

        if len(obs_df) < 20 or obs_df[dim].nunique() < 2:
            print(f"\n  {DIM_LABELS[dim]}: insufficient data or no variance")
            continue

        # Check for floor/ceiling effect (all same value)
        if obs_df[dim].nunique() == 1:
            print(f"\n  {DIM_LABELS[dim]}: NO VARIANCE (all = {obs_df[dim].iloc[0]:.2f}) — EXCLUDE from analysis")
            results.append({
                "dimension": dim, "label": DIM_LABELS[dim],
                "status": "excluded_no_variance",
                "value": obs_df[dim].iloc[0],
            })
            continue

        # Create model dummies (first model as reference)
        models = sorted(obs_df["model_id"].unique())
        ref_model = models[0]

        try:
            # OLR with model dummies only (arch is collinear with model)
            X = pd.get_dummies(obs_df["model_id"], drop_first=True, dtype=float)

            # Ensure response is integer for OLR
            y = obs_df[dim].round().astype(int)

            model = OrderedModel(y, X, distr='logit')
            result = model.fit(method='bfgs', disp=False, maxiter=100)

            # Wald test for overall model effect
            # Pseudo R-squared
            pseudo_r2 = result.prsquared

            # LLR p-value
            llr_pvalue = result.llr_pvalue if hasattr(result, 'llr_pvalue') else np.nan

            # Coefficients for each model
            model_coefs = {}
            for v in models[1:]:
                col = f"model[T.{v}]" if f"model[T.{v}]" in result.params.index else v
                if col in result.params.index:
                    model_coefs[v] = {
                        "coef": round(result.params[col], 4),
                        "se": round(result.bse[col], 4),
                        "p": round(result.pvalues[col], 6) if col in result.pvalues.index else np.nan,
                        "or": round(np.exp(result.params[col]), 4),
                    }

            results.append({
                "dimension": dim,
                "label": DIM_LABELS[dim],
                "status": "ok",
                "ref_model": ref_model,
                "n_obs": len(obs_df),
                "n_models": len(models),
                "pseudo_r2": round(pseudo_r2, 6),
                "llr_pvalue": round(llr_pvalue, 6) if not np.isnan(llr_pvalue) else np.nan,
                "model_coefs": model_coefs,
            })

            print(f"\n  {DIM_LABELS[dim]:25s}: pseudo-R² = {pseudo_r2:.4f}, "
                  f"LLR p = {f'{llr_pvalue:.2e}' if not np.isnan(llr_pvalue) else 'N/A'}")
            for v, coef in model_coefs.items():
                sig = "***" if coef["p"] < 0.001 else "**" if coef["p"] < 0.01 else "*" if coef["p"] < 0.05 else ""
                print(f"    vs {ref_model}: {v:15s} OR={coef['or']:.3f}, p={coef['p']:.2e} {sig}")

        except Exception as e:
            print(f"\n  {DIM_LABELS[dim]:25s}: OLR failed: {e}")
            results.append({
                "dimension": dim, "label": DIM_LABELS[dim],
                "status": f"error: {e}",
            })

    # FDR correction on OLR p-values
    olr_results = [r for r in results if r.get("status") == "ok"]
    if olr_results:
        all_pvals = []
        for r in olr_results:
            for v, coef in r["model_coefs"].items():
                if not np.isnan(coef["p"]):
                    all_pvals.append(coef["p"])

        if all_pvals:
            all_pvals = np.array(all_pvals)
            fdr_pvals = fdr_correction(all_pvals)
            print(f"\n  FDR correction applied to {len(all_pvals)} pairwise comparisons")

    return results


def study1_cohen_d(df: pd.DataFrame):
    """Compute pairwise Cohen's d between models for each dimension.

    Effect size interpretation for LLM response styles (adapted from Cohen, 1988):
    - d < 0.2: negligible — unlikely to affect downstream use
    - 0.2 ≤ d < 0.5: small — detectable but limited practical significance
    - 0.5 ≤ d < 0.8: medium — substantial model-specific response tendency
    - d ≥ 0.8: large — strong model signature, likely reflects systematic differences
      in training data or alignment
    Note: Standard psychology thresholds assume human populations with high within-group
    variance. LLM response styles may have lower baseline variance, making even small d
    values noteworthy. We report raw d values for reader interpretation.
    """
    print("\n" + "=" * 80)
    print("STUDY 1: Pairwise Cohen's d (Top Pairs)")
    print("=" * 80)

    from scipy import stats

    models = sorted(df["model_id"].unique())
    n_models = len(models)
    results = []

    for dim in DIMENSIONS:
        dim_results = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                v1, v2 = models[i], models[j]
                g1 = df[df["model_id"] == v1][dim].dropna().values
                g2 = df[df["model_id"] == v2][dim].dropna().values
                if len(g1) < 2 or len(g2) < 2:
                    continue

                # Pooled SD
                pooled_sd = np.sqrt(((len(g1) - 1) * np.var(g1, ddof=1) +
                                    (len(g2) - 1) * np.var(g2, ddof=1)) /
                                   (len(g1) + len(g2) - 2))

                if pooled_sd == 0:
                    continue

                d = (np.mean(g1) - np.mean(g2)) / pooled_sd

                # Welch's t-test
                t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

                dim_results.append({
                    "dim": dim, "v1": v1, "v2": v2,
                    "d": round(d, 4), "p": round(p_val, 6),
                    "m1": round(np.mean(g1), 3), "m2": round(np.mean(g2), 3),
                    "n1": len(g1), "n2": len(g2),
                })

        if dim_results:
            dim_results.sort(key=lambda x: abs(x["d"]), reverse=True)
            print(f"\n  {DIM_LABELS[dim]} (top 5 largest |d|):")
            for dr in dim_results[:5]:
                mag = "large" if abs(dr["d"]) >= 0.8 else "medium" if abs(dr["d"]) >= 0.5 else "small" if abs(dr["d"]) >= 0.2 else "negligible"
                print(f"    {dr['v1']:12s} vs {dr['v2']:12s}: d={dr['d']:+.3f} ({mag}), p={dr['p']:.2e}")
            results.extend(dim_results)

    # FDR correction on all pairwise tests
    if results:
        all_p = np.array([r["p"] for r in results])
        all_fdr = fdr_correction(all_p)
        n_sig_raw = sum(all_p < 0.05)
        n_sig_fdr = sum(all_fdr < 0.05)
        print(f"\n  Total pairwise tests: {len(results)}")
        print(f"  Significant at p<0.05 (raw): {n_sig_raw}")
        print(f"  Significant at FDR<0.05: {n_sig_fdr}")

        # Distribution of effect sizes
        all_d = [abs(r["d"]) for r in results]
        print(f"  Effect size distribution: "
              f"M={np.mean(all_d):.3f}, median={np.median(all_d):.3f}, "
              f"max={np.max(all_d):.3f}")
        large = sum(1 for d in all_d if d >= 0.8)
        medium = sum(1 for d in all_d if 0.5 <= d < 0.8)
        small = sum(1 for d in all_d if 0.2 <= d < 0.5)
        negligible = sum(1 for d in all_d if d < 0.2)
        print(f"  {large} large (d≥0.8), {medium} medium (0.5-0.8), {small} small (0.2-0.5), {negligible} negligible (d<0.2)")

    return results


# ============== STUDY 2: WITHIN-MODEL TRAJECTORIES ==============

def study2_descriptive(df: pd.DataFrame):
    """Descriptive analysis of within-model evolution trajectories."""
    print("\n" + "=" * 80)
    print("STUDY 2: Within-Model Trajectories (Descriptive)")
    print("=" * 80)

    results = []

    for model in sorted(df["model_id"].unique()):
        if not model:  # skip records with empty model
            continue
        model_data = df[df["model_id"] == model]
        subgroups = model_data["subgroup"].fillna("").unique()
        subgroups = [s for s in subgroups if s and s != ""]  # filter empty

        if not subgroups:
            print(f"\n  {model}: No subgroup data (Study 2)")
            continue

        print(f"\n  {model}:")

        for sg in sorted(subgroups):
            sg_data = model_data[model_data["subgroup"].fillna("") == sg]
            if len(sg_data) == 0:
                continue

            print(f"    Subgroup: {sg}")

            # Sort by model (use short names if available)
            for dim in DIMENSIONS:
                model_means = sg_data.groupby("model")[dim].agg(["mean", "std", "count"])
                model_means = model_means.sort_values("mean")

                if len(model_means) >= 2:
                    # Range (max - min)
                    range_val = model_means["mean"].max() - model_means["mean"].min()
                    results.append({
                        "model": model, "subgroup": sg,
                        "dimension": dim, "label": DIM_LABELS[dim],
                        "n_models": len(model_means),
                        "range": round(range_val, 4),
                        "min_model": model_means["mean"].idxmin(),
                        "max_model": model_means["mean"].idxmax(),
                        "min_mean": round(model_means["mean"].min(), 4),
                        "max_mean": round(model_means["mean"].max(), 4),
                    })

            # Print model means for key dimensions
            for model_id, row in sg_data.groupby("model").first().iterrows():
                short = model_id.split("/")[-1]
                scores = []
                for dim in DIMENSIONS:
                    if dim in sg_data.columns:
                        m = sg_data[sg_data["model"] == model_id][dim].mean()
                        if not np.isnan(m):
                            scores.append(f"{DIM_LABELS[dim][:4]}={m:.2f}")
                print(f"      {short:30s}: {' '.join(scores)}")

    return results


# ============== FLOOR-EFFECT ROBUSTNESS ==============

def floor_effect_robustness(df: pd.DataFrame):
    """Re-run ANOVA excluding models with floor effects on each dimension.
    A model has a floor effect if its mean on a dimension is <= 1.5 (near minimum of 1.0).
    """
    from scipy import stats

    print("\n" + "=" * 80)
    print("STUDY 1: FLOOR-EFFECT ROBUSTNESS (Excluding floor-effected models)")
    print("=" * 80)

    FLOOR_THRESHOLD = 1.5

    for dim in DIMENSIONS:
        # Identify models with floor effect on this dimension
        model_means = df.groupby("model")[dim].mean()
        floor_models = model_means[model_means <= FLOOR_THRESHOLD].index.tolist()

        # Exclude floor-effected models
        df_clean = df[~df["model"].isin(floor_models)]

        if len(floor_models) == 0:
            print(f"\n  {DIM_LABELS[dim]:25s}: no floor effects (all models included)")
            continue

        models = sorted(df_clean["model"].unique())
        groups = [df_clean[df_clean["model"] == v][dim].dropna().values for v in models]
        valid_groups = [(v, g) for v, g in zip(models, groups) if len(g) >= 2]

        if len(valid_groups) < 2:
            print(f"\n  {DIM_LABELS[dim]:25s}: insufficient data after exclusion")
            continue

        models_valid = [v for v, g in valid_groups]
        groups_valid = [g for v, g in valid_groups]

        f_stat, p_value = stats.f_oneway(*groups_valid)

        all_data = np.concatenate(groups_valid)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups_valid)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        partial_eta_sq = ss_between / ss_total if ss_total > 0 else 0

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        print(f"\n  {DIM_LABELS[dim]:25s}: F({len(models_valid)-1},{len(all_data)-len(models_valid)}) = {f_stat:.2f}, "
              f"p = {p_value:.2e}, η²p = {partial_eta_sq:.4f} {sig}")
        print(f"    Excluded {len(floor_models)} floor-effected models: {[m.split('/')[-1] for m in floor_models]}")
        print(f"    Remaining: {len(df_clean)} obs, {len(models_valid)} models")


# ============== HEXACO-H DESCRIPTIVE OBSERVATION ==============

def hexaco_descriptive_observation(df: pd.DataFrame):
    """Report HEXACO-H as descriptive observation (not primary analysis).
    HEXACO-H has low reliability (α=0.27, 5 items) and is excluded from primary ANOVA.
    However, the floor effect pattern is a noteworthy descriptive observation about alignment.
    """
    from scipy import stats

    print("\n" + "=" * 80)
    print("HEXACO-H: DESCRIPTIVE OBSERVATION (Excluded from Primary Analysis)")
    print("=" * 80)
    print("  NOTE: HEXACO-H measured with only 5 items (α=0.27 — unreliable)")
    print("  Reported as descriptive observation, not statistical claim")
    print("  Floor effect (M≤1.5) may reflect RLHF alignment toward honesty/humility")

    h_values = df.groupby("model")[HEXACO_DIM].mean().sort_values()
    floor_models = h_values[h_values <= 1.5]
    non_floor_models = h_values[h_values > 1.5]

    print(f"\n  Models with floor effect (H≤1.5): {len(floor_models)}/{len(h_values)}")
    for m, v in floor_models.items():
        model = df[df["model"] == m]["model"].iloc[0] if len(df[df["model"] == m]) > 0 else "?"
        print(f"    {m.split('/')[-1]:35s}: H={v:.2f} ({model})")

    print(f"\n  Models without floor effect (H>1.5): {len(non_floor_models)}/{len(h_values)}")
    for m, v in non_floor_models.items():
        model = df[df["model"] == m]["model"].iloc[0] if len(df[df["model"] == m]) > 0 else "?"
        print(f"    {m.split('/')[-1]:35s}: H={v:.2f} ({model})")

    # Quick non-parametric test on non-floor models only
    if len(non_floor_models) >= 3:
        s1 = df[(df["study"] == 1) & (df["model"].isin(non_floor_models.index))]
        if len(s1) > 0:
            models_s1 = sorted(s1["model"].unique())
            groups = [s1[s1["model"] == v][HEXACO_DIM].dropna().values for v in models_s1]
            valid = [(v, g) for v, g in zip(models_s1, groups) if len(g) >= 2]
            if len(valid) >= 2:
                h_stat, kw_p = stats.kruskal(*[g for _, g in valid])
                print(f"\n  Kruskal-Wallis on non-floor Study 1 models: H={h_stat:.2f}, p={kw_p:.2e}")
                print(f"  → Model differences in HEXACO-H persist even after removing floor-effected models")


# ============== THINKING ABLATION ==============

def thinking_ablation_analysis(df: pd.DataFrame):
    """Analyze thinking ablation: enable_thinking ON vs OFF."""
    print("\n" + "=" * 80)
    print("THINKING ABLATION: enable_thinking ON vs OFF")
    print("=" * 80)

    from scipy import stats

    ablation = df[df["subgroup"].fillna("") == "Thinking-Ablation"].copy()
    if len(ablation) == 0:
        print("  No ablation data found")
        return []

    results = []

    for model_id in sorted(ablation["model"].unique()):
        model_data = ablation[ablation["model"] == model_id]
        chat_data = model_data[model_data["thinking_mode"].fillna("chat") == "chat"]
        think_data = model_data[model_data["thinking_mode"].fillna("") == "thinking"]

        if len(chat_data) == 0 or len(think_data) == 0:
            print(f"\n  {model_id}: incomplete data")
            continue

        print(f"\n  {model_id.split('/')[-1]} (chat n={len(chat_data)}, thinking n={len(think_data)}):")

        for dim in DIMENSIONS:
            chat_vals = chat_data[dim].dropna().values
            think_vals = think_data[dim].dropna().values

            if len(chat_vals) < 2 or len(think_vals) < 2:
                continue

            # Paired comparison (by seed if available)
            # Try to match by seed
            common_seeds = set(chat_data["seed"]) & set(think_data["seed"])
            if common_seeds:
                chat_matched = chat_data[chat_data["seed"].isin(common_seeds)].sort_values("seed")
                think_matched = think_data[think_data["seed"].isin(common_seeds)].sort_values("seed")
                paired_chat = chat_matched[dim].values
                paired_think = think_matched[dim].values

                if len(paired_chat) >= 3:
                    t_stat, p_val = stats.ttest_rel(paired_chat, paired_think)
                else:
                    t_stat, p_val = stats.ttest_ind(chat_vals, think_vals)
            else:
                t_stat, p_val = stats.ttest_ind(chat_vals, think_vals)

            # Cohen's d
            pooled_sd = np.sqrt(((len(chat_vals) - 1) * np.var(chat_vals, ddof=1) +
                                (len(think_vals) - 1) * np.var(think_vals, ddof=1)) /
                               (len(chat_vals) + len(think_vals) - 2))
            d = (np.mean(chat_vals) - np.mean(think_vals)) / pooled_sd if pooled_sd > 0 else 0

            sig = "*" if p_val < 0.05 else ""
            print(f"    {DIM_LABELS[dim]:25s}: chat={np.mean(chat_vals):.3f}, think={np.mean(think_vals):.3f}, "
                  f"d={d:+.3f}, p={p_val:.4f} {sig}")

            results.append({
                "model": model_id, "dimension": dim,
                "chat_mean": round(np.mean(chat_vals), 4),
                "think_mean": round(np.mean(think_vals), 4),
                "d": round(d, 4), "p": round(p_val, 6),
                "n_chat": len(chat_vals), "n_think": len(think_vals),
            })

    return results


# ============== ALIGNMENT ARTIFACT ANALYSIS ==============

def alignment_artifact_analysis(df: pd.DataFrame):
    """Analyze dimensions that may reflect alignment artifacts rather than response style.

    A dimension is classified as an ALIGNMENT ARTIFACT if it shows low inter-model
    variance (SD < threshold), meaning all models respond similarly regardless of their
    different training procedures. Such dimensions cannot distinguish models.

    IMPORTANT: A dimension can be alignment-SENSITIVE (responds to RLHF) AND still show
    high inter-model variance. Alignment sensitivity ≠ alignment artifact. For example,
    HEXACO-H (morality) is alignment-sensitive but may show high cross-model variance if
    different models apply different alignment strengths.

    Classification rule (data-driven):
    - ALIGNMENT ARTIFACT: inter-model SD < 0.15 (near-zero cross-model variation)
    - DISCRIMINATIVE: inter-model SD >= 0.15 (sufficient variation to distinguish models)
    """
    print("\n" + "=" * 80)
    print("ALIGNMENT ARTIFACT ANALYSIS")
    print("=" * 80)
    print("  Identifying dimensions with near-zero inter-model variance (alignment artifacts)")
    print("  These dimensions cannot distinguish models and are excluded from discriminative analysis")
    print("  Classification is DATA-DRIVEN (inter-model SD), not theory-driven")

    model_means = df.groupby("model")[DIMENSIONS].mean()

    results = []
    for dim in DIMENSIONS:
        dim_means = model_means[dim].dropna()
        if len(dim_means) < 2:
            continue

        mean_val = dim_means.mean()
        sd_val = dim_means.std()
        cv = sd_val / mean_val if mean_val > 0 else np.nan  # Coefficient of variation

        # Classification: data-driven based on inter-model SD
        low_variance = sd_val < 0.15  # Near-zero cross-model variation
        is_alignment = low_variance  # Only low variance = alignment artifact
        status = "ALIGNMENT" if is_alignment else "DISCRIMINATIVE"

        results.append({
            "dimension": dim, "label": DIM_LABELS[dim],
            "mean": round(mean_val, 4), "sd": round(sd_val, 4), "cv": round(cv, 4),
            "range": round(dim_means.max() - dim_means.min(), 4),
            "n_models": len(dim_means),
            "low_variance": low_variance,
            "status": status,
        })

    print(f"\n  {'Dimension':25s} {'Mean':>6s} {'SD':>6s} {'Range':>6s} {'N':>4s} {'Status':15s}")
    print("  " + "-" * 70)
    for r in results:
        flags = []
        if r["low_variance"]: flags.append("LOW-VAR")
        flag_str = ", ".join(flags) if flags else ""
        status_marker = "***" if r["status"] == "ALIGNMENT" else "   "
        print(f"{status_marker} {r['label']:21s} {r['mean']:>6.2f} {r['sd']:>6.4f} {r['range']:>6.2f} "
              f"{r['n_models']:>4d} {r['status']:>15s} {flag_str}")

    n_align = sum(1 for r in results if r["status"] == "ALIGNMENT")
    n_disc = sum(1 for r in results if r["status"] == "DISCRIMINATIVE")
    print(f"\n  Summary: {n_align} alignment artifacts (excluded), {n_disc} discriminative dimensions (included)")

    if n_align > 0:
        print(f"  Excluded from discriminative analysis: "
              f"{[r['label'] for r in results if r['status'] == 'ALIGNMENT']}")
    if n_disc > 0:
        print(f"  Included in discriminative analysis: "
              f"{[r['label'] for r in results if r['status'] == 'DISCRIMINATIVE']}")

    return results


def alignment_threshold_sensitivity(df: pd.DataFrame):
    """Sensitivity analysis for alignment artifact classification threshold.

    Tests multiple SD thresholds to verify that conclusions are robust
    to the choice of alignment artifact cutoff.
    """
    print("\n" + "=" * 80)
    print("ALIGNMENT ARTIFACT THRESHOLD SENSITIVITY")
    print("=" * 80)
    print("  Testing classification stability across multiple SD thresholds")

    model_means = df.groupby("model")[DIMENSIONS].mean()
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # Pre-compute SD for each dimension
    dim_sds = {}
    for dim in DIMENSIONS:
        dim_means = model_means[dim].dropna()
        if len(dim_means) >= 2:
            dim_sds[dim] = dim_means.std()

    print(f"\n  {'Threshold':>10s}", end="")
    for dim in DIMENSIONS:
        print(f" {DIM_LABELS[dim][:8]:>8s}", end="")
    print(f" {'#Align':>7s} {'#Disc':>7s}")
    print("  " + "-" * (10 + 9 * 9 + 16))

    results = []
    for thresh in thresholds:
        n_align = 0
        n_disc = 0
        row = {"threshold": thresh}
        for dim in DIMENSIONS:
            if dim in dim_sds:
                is_align = dim_sds[dim] < thresh
                row[dim] = "A" if is_align else "D"
                if is_align:
                    n_align += 1
                else:
                    n_disc += 1
            else:
                row[dim] = "?"
        print(f"  {thresh:>10.2f}", end="")
        for dim in DIMENSIONS:
            print(f" {row.get(dim, '?'):>8s}", end="")
        print(f" {n_align:>7d} {n_disc:>7d}")
        results.append({"threshold": thresh, "n_align": n_align, "n_disc": n_disc, "classifications": row})

    # Check stability: is the core conclusion (H=discriminative, UA=alignment) stable?
    print(f"\n  Conclusion stability check:")
    for dim_name, expected in [("hexaco_h", "D"), ("uncertainty_avoidance", "A")]:
        label = DIM_LABELS[dim_name]
        statuses = [r["classifications"].get(dim_name, "?") for r in results]
        stable = all(s == expected for s in statuses if s != "?")
        print(f"    {label:25s}: always {expected} = {'STABLE' if stable else 'UNSTABLE'}")

    return results


def convergent_validity_check(df: pd.DataFrame):
    """Check whether LLM inter-dimension correlations match human baselines.

    Human BFI inter-dimension correlations (John & Srivastava, 1999 meta-analysis):
    - E-N: r ≈ -0.30 (extraversion negatively correlated with neuroticism)
    - A-C: r ≈ +0.20 (agreeableness positively correlated with conscientiousness)
    - O-A: r ≈ +0.10 (openness slightly positively correlated with agreeableness)

    If LLM correlations diverge strongly from human patterns (e.g., all dimensions
    highly positively correlated), this suggests acquiescence bias rather than
    construct-specific responding.
    """
    print("\n" + "=" * 80)
    print("CONVERGENT VALIDITY CHECK: Inter-Dimension Correlations vs Human Baseline")
    print("=" * 80)

    model_means = df.groupby("model")[DIMENSIONS].mean()
    corr_matrix = model_means.corr()

    # Human baseline correlations (John & Srivastava, 1999)
    human_baseline = {
        ("bfi.extraversion", "bfi.neuroticism"): -0.30,
        ("bfi.agreeableness", "bfi.conscientiousness"): 0.20,
        ("bfi.openness", "bfi.agreeableness"): 0.10,
        ("bfi.extraversion", "bfi.agreeableness"): 0.10,
        ("bfi.conscientiousness", "bfi.neuroticism"): -0.25,
    }

    print(f"\n  {'Dimension Pair':40s} {'Human r':>8s} {'LLM r':>8s} {'Match?':>8s}")
    print("  " + "-" * 70)

    matches = 0
    total = 0
    for (d1, d2), human_r in human_baseline.items():
        if d1 in corr_matrix.index and d2 in corr_matrix.columns:
            llm_r = corr_matrix.loc[d1, d2]
            is_match = (human_r * llm_r) > 0
            if is_match:
                matches += 1
            total += 1
            marker = "OK" if is_match else "MISMATCH"
            label = f"{DIM_LABELS[d1]} vs {DIM_LABELS[d2]}"
            print(f"  {label:40s} {human_r:>+8.2f} {llm_r:>+8.2f} {marker:>8s}")

    print(f"\n  Sign agreement: {matches}/{total} dimension pairs match human baseline direction")

    # Check for acquiescence bias (all dimensions positively correlated)
    bfi_dims = [d for d in DIMENSIONS if d.startswith("bfi.")]
    bfi_corr = corr_matrix.loc[bfi_dims, bfi_dims]
    off_diag = bfi_corr.values[np.triu_indices_from(bfi_corr.values, k=1)]
    positive_pct = (off_diag > 0).mean() * 100
    mean_corr = np.mean(np.abs(off_diag))

    print(f"\n  Acquiescence bias check (BFI dimensions only):")
    print(f"    Positive correlations: {positive_pct:.0f}% of pairs")
    print(f"    Mean |r|: {mean_corr:.3f}")
    if positive_pct > 80:
        print(f"    WARNING: High positive correlation rate suggests acquiescence bias")
    else:
        print(f"    OK: Correlation structure shows dimension differentiation")

    return {
        "sign_agreement": matches, "total_pairs": total,
        "positive_pct": round(positive_pct, 1), "mean_abs_r": round(mean_corr, 3),
    }


def acquiescence_bias_analysis(df: pd.DataFrame):
    """Test for acquiescence bias in LLM responses.

    Acquiescence bias = tendency to agree with items regardless of content.
    Manifests as: (1) high positive correlations between all dimensions,
    (2) first principal component explaining large variance share,
    (3) correlations becoming weaker after partialling out PC1.
    """
    print("\n" + "=" * 80)
    print("ACQUIESCENCE BIAS ANALYSIS")
    print("=" * 80)

    model_means = df.groupby("model")[DIMENSIONS].mean()

    # 1. Extract first principal component
    from numpy.linalg import eig

    corr_matrix = model_means.corr()
    eigenvalues, eigenvectors = eig(corr_matrix.values)
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real

    total_variance = eigenvalues.sum()
    pc1_variance_pct = eigenvalues[0] / total_variance * 100
    pc1_loadings = eigenvectors[:, 0]

    print(f"\n  First Principal Component (PC1 = Acquiescence Factor):")
    print(f"    Variance explained: {pc1_variance_pct:.1f}%")
    print(f"    (9 dimensions → expected ~11.1% if uncorrelated)")
    if pc1_variance_pct > 30:
        print(f"    WARNING: PC1 explains >30% of variance — strong acquiescence signal")
    elif pc1_variance_pct > 20:
        print(f"    NOTE: PC1 explains >20% — moderate acquiescence signal")

    print(f"\n  PC1 Loadings:")
    for i, dim in enumerate(DIMENSIONS):
        loading = pc1_loadings[i]
        sign = "+" if loading > 0 else "-"
        print(f"    {DIM_LABELS[dim]:25s}: {loading:+.3f} {sign}")

    # 2. Compute PC1 score for each model
    pc1_scores = model_means.values @ pc1_loadings

    # 3. Partial correlations: correlate each dim with each other, controlling for PC1
    print(f"\n  Partial Correlations (controlling for PC1):")
    # Use residuals from regressing each dimension on PC1
    from scipy import stats

    residuals = {}
    for i, dim in enumerate(DIMENSIONS):
        x = model_means[dim].values
        slope, intercept, _, _, _ = stats.linregress(pc1_scores, x)
        residuals[dim] = x - (slope * pc1_scores + intercept)

    # Recompute correlation on residuals
    resid_df = pd.DataFrame(residuals)
    resid_corr = resid_df.corr()

    short_dims = ["E", "A", "C", "N", "O", "H", "Col", "Int", "UA"]
    header = f"{'':>6s}" + "".join(f"{d:>6s}" for d in short_dims)
    print(f"  {header}")
    for i, dim in enumerate(DIMENSIONS):
        row = f"{short_dims[i]:>6s}"
        for j, dim2 in enumerate(DIMENSIONS):
            row += f"{resid_corr.loc[dim, dim2]:>6.2f}"
        print(f"  {row}")

    # Compare: how many high correlations remain after PC1 correction?
    off_diag_orig = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    off_diag_resid = resid_corr.values[np.triu_indices_from(resid_corr.values, k=1)]

    high_orig = np.sum(np.abs(off_diag_orig) > 0.5)
    high_resid = np.sum(np.abs(off_diag_resid) > 0.5)
    mean_abs_orig = np.mean(np.abs(off_diag_orig))
    mean_abs_resid = np.mean(np.abs(off_diag_resid))

    print(f"\n  Comparison:")
    print(f"    High correlations (|r|>0.5): original={high_orig}, after PC1={high_resid}")
    print(f"    Mean |r|: original={mean_abs_orig:.3f}, after PC1={mean_abs_resid:.3f}")
    reduction = (mean_abs_orig - mean_abs_resid) / mean_abs_orig * 100
    print(f"    Correlation reduction: {reduction:.1f}%")

    # 4. Check convergent validity after PC1 correction
    human_baseline = {
        ("bfi.extraversion", "bfi.neuroticism"): -0.30,
        ("bfi.agreeableness", "bfi.conscientiousness"): 0.20,
    }
    print(f"\n  Convergent validity AFTER PC1 correction:")
    for (d1, d2), human_r in human_baseline.items():
        if d1 in resid_corr.index and d2 in resid_corr.columns:
            llm_r = resid_corr.loc[d1, d2]
            match = (human_r * llm_r) > 0
            marker = "OK" if match else "MISMATCH"
            label = f"{DIM_LABELS[d1][:8]} vs {DIM_LABELS[d2][:8]}"
            print(f"    {label:25s}: human={human_r:+.2f}, LLM_pc1corr={llm_r:+.3f} {marker}")

    return {
        "pc1_variance_pct": round(pc1_variance_pct, 1),
        "high_corr_original": int(high_orig),
        "high_corr_residual": int(high_resid),
        "mean_abs_original": round(mean_abs_orig, 3),
        "mean_abs_residual": round(mean_abs_resid, 3),
        "reduction_pct": round(reduction, 1),
    }


def hexaco_scale_correlation(df: pd.DataFrame):
    """Analyze correlation between HEXACO-H and model scale/architecture.

    Core hypothesis: HEXACO-H decreases (floor effect) as model size increases,
    reflecting stronger alignment in larger models.
    """
    print("\n" + "=" * 80)
    print("HEXACO-H × MODEL SCALE CORRELATION (Core Finding)")
    print("=" * 80)

    from scipy import stats

    MODEL_PARAMS = {
        "Qwen/Qwen3.5-397B-A17B": 17, "Qwen/Qwen3.5-4B": 4,
        "Qwen/Qwen3.5-27B": 27, "Qwen/Qwen3.5-35B-A3B": 3,
        "Qwen/Qwen3.5-122B-A10B": 10,
        "ollama:qwen3.5:9b": 9,
        "Pro/deepseek-ai/DeepSeek-V3.2": 37, "deepseek-ai/DeepSeek-V3": 37,
        "deepseek-ai/DeepSeek-V2.5": 21, "deepseek-ai/DeepSeek-R1": 37,
        "baidu/ERNIE-4.5-300B-A47B": 47,
        "tencent/Hunyuan-A13B-Instruct": 13,
        "ByteDance-Seed/Seed-OSS-36B-Instruct": 36,
        "internlm/internlm2_5-7b-chat": 7,
        "THUDM/GLM-4-9B-0414": 9, "THUDM/GLM-4-32B-0414": 32,
        "THUDM/GLM-Z1-32B-0414": 32,
    }

    model_h = df.groupby("model")["hexaco_h"].agg(["mean", "std", "count"])
    model_h = model_h.dropna()

    known_params = []
    for model_id in model_h.index:
        params = MODEL_PARAMS.get(model_id, 0)
        if params > 0:
            known_params.append((params, model_h.loc[model_id, "mean"]))

    if len(known_params) < 3:
        print(f"  Insufficient models with known parameter counts: {len(known_params)}")
        print(f"\n  HEXACO-H means by model:")
        for model_id, row in model_h.iterrows():
            short = model_id.split("/")[-1][:30]
            print(f"    {short:35s}: H={row['mean']:.3f} (SD={row['std']:.3f}, n={int(row['count'])})")
        return {"r": np.nan, "p": np.nan, "n": len(known_params)}

    params_arr = np.array([x[0] for x in known_params])
    h_arr = np.array([x[1] for x in known_params])

    r, p = stats.pearsonr(params_arr, h_arr)
    slope, intercept, r_val, p_val, se = stats.linregress(params_arr, h_arr)

    print(f"\n  HEXACO-H vs Active Parameters (n={len(known_params)} models):")
    print(f"    Pearson r = {r:.4f}, p = {p:.4f}")
    print(f"    Slope = {slope:.4f} (change in H per 1B active params)")
    print(f"    Intercept = {intercept:.4f}")

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"    Significance: {sig}")

    print(f"\n  Per-model breakdown:")
    for params, h_mean in sorted(known_params, key=lambda x: x[0]):
        print(f"    {params:>3d}B active: H={h_mean:.3f}")

    if r < -0.3 and p < 0.10:
        print(f"\n  *** FINDING: Negative correlation supports alignment strength hypothesis")
        print(f"  Larger models show lower H (stronger alignment)")
    elif r > 0.1:
        print(f"\n  NOTE: Positive correlation does NOT support alignment strength hypothesis")
    else:
        print(f"\n  NOTE: Weak/no correlation — hypothesis not supported")

    return {"r": round(r, 4), "p": round(p, 4), "n": len(known_params),
            "slope": round(slope, 4), "intercept": round(intercept, 4)}


# ============== RESPONSE QUALITY CHECKS ==============

def response_quality_check(df: pd.DataFrame):
    """Check response quality across all observations."""
    print("\n" + "=" * 80)
    print("RESPONSE QUALITY CHECKS")
    print("=" * 80)

    # Check for response_stats field
    if "response_stats" not in df.columns:
        print("  No response_stats field found (data collected before quality monitoring was added)")
        print("  Checking item-level data for anomalies...")

        # Check dimension score distributions
        for dim in DIMENSIONS:
            scores = df[dim].dropna()
            if len(scores) == 0:
                continue
            unique_vals = scores.nunique()
            floor_pct = (scores <= 1.01).mean() * 100
            ceil_pct = (scores >= 4.99).mean() * 100
            center_pct = ((scores >= 2.5) & (scores <= 3.5)).mean() * 100
            print(f"  {DIM_LABELS[dim]:25s}: n={len(scores)}, unique={unique_vals}, "
                  f"floor={floor_pct:.1f}%, ceiling={ceil_pct:.1f}%, center={center_pct:.1f}%")

            # Flag dimensions with zero variance
            if scores.std() < 0.01:
                print(f"    *** WARNING: Near-zero variance (SD={scores.std():.4f}) — exclude from analysis")

        return

    # If response_stats exists, use it
    stats_df = pd.json_normalize(df["response_stats"])
    print(f"\n  Mean response length: {stats_df['mean_length'].mean():.1f} chars")
    print(f"  Short responses (<5 chars): {stats_df['short_response_count'].sum()} / {stats_df['total_items'].sum()} total items")

    non_numeric = stats_df["non_numeric_count"]
    if non_numeric.sum() > 0:
        print(f"  *** WARNING: {non_numeric.sum()} observations with non-numeric responses")
    else:
        print(f"  All responses start with numeric characters (good)")

# ============== POWER ANALYSIS ==============

def power_analysis(df: pd.DataFrame):
    """Compute power analysis for Study 1 cross-model comparisons.

    Estimates minimum detectable Cohen's d for pairwise model comparisons
    given the actual sample sizes in the data.
    """
    print("\n" + "=" * 80)
    print("POWER ANALYSIS")
    print("=" * 80)

    from scipy import stats

    s1 = get_study1_data(df)
    if len(s1) == 0:
        print("  No Study 1 data available for power analysis")
        return

    models = s1["model"].unique()
    n_models = len(models)
    n_seeds_per_model = s1.groupby("model_id").size()
    n_min = n_seeds_per_model.min()
    n_max = n_seeds_per_model.max()
    n_mean = n_seeds_per_model.mean()

    print(f"\n  Study 1: {n_models} models, {n_min}-{n_max} seeds per model (mean={n_mean:.0f})")
    print(f"  Pairwise comparisons: C({n_models},2) = {n_models*(n_models-1)//2}")
    print(f"  Dimensions: {len(DIMENSIONS)}")
    print(f"  Total tests (before FDR): {n_models*(n_models-1)//2 * len(DIMENSIONS)}")

    # For each dimension, compute within-group SD (pooled across all models)
    print(f"\n  Within-group SD per dimension (used for effect size denominator):")
    within_sds = {}
    for dim in DIMENSIONS:
        all_vals = s1[dim].dropna()
        if len(all_vals) > 0:
            # Pooled within-model SD
            model_sds = []
            for v in models:
                v_vals = s1[s1["model"] == v][dim].dropna()
                if len(v_vals) >= 2:
                    model_sds.append(np.var(v_vals, ddof=1))
            if model_sds:
                pooled_var = np.mean(model_sds)
                pooled_sd = np.sqrt(pooled_var)
                within_sds[dim] = pooled_sd
                print(f"    {DIM_LABELS[dim]:25s}: pooled SD = {pooled_sd:.4f}")

    # Power for two-sample t-test (two-sided, alpha=0.05)
    print(f"\n  Power analysis for pairwise t-tests (α=0.05, two-sided):")
    print(f"  Assuming n1=n2={n_min} per group (conservative, uses smallest model):")
    print(f"  {'d':>6s} {'Power':>8s}  {'Interpretation'}")
    print(f"  {'-'*50}")

    alpha_uncorrected = 0.05
    # With FDR correction over all tests
    n_tests_total = n_models * (n_models - 1) // 2 * len(DIMENSIONS)
    alpha_fdr = 0.05  # FDR threshold

    for d in [0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
        # Approximate power for two-sample t-test
        ncp = d * np.sqrt(n_min / 2)  # Non-centrality parameter
        # Use normal approximation for t-test
        from scipy.stats import norm
        z_crit = norm.ppf(1 - alpha_uncorrected / 2)
        power_approx = 1 - norm.cdf(z_crit - ncp) + norm.cdf(-z_crit - ncp)
        power_approx = max(0, min(1, power_approx))

        interp = "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"
        print(f"  {d:>6.2f} {power_approx:>8.1%}  {interp}")

    # What d can we detect with 80% power?
    print(f"\n  Minimum detectable d for 80% power (n={n_min} per group, α=0.05):")
    for power_target in [0.80, 0.90]:
        # Solve: power = 1 - Φ(z_α/2 - δ√(n/2)) + Φ(-z_α/2 - δ√(n/2))
        # Approximate: δ ≈ (z_α/2 + z_β) * √(2/n)
        z_alpha = norm.ppf(1 - alpha_uncorrected / 2)
        z_beta = norm.ppf(power_target)
        d_min = (z_alpha + z_beta) * np.sqrt(2 / n_min)
        print(f"    80%/90% power → d_min ≈ {d_min:.3f}")

    print(f"\n  Note: With FDR correction over {n_tests_total} tests, effective α per test is lower.")
    print(f"  Conservative minimum detectable d (accounting for FDR): ~{d_min * 1.3:.3f}")

    # Actual power based on observed effect sizes (from existing data)
    print(f"\n  Observed effect sizes from current data:")
    for dim in DIMENSIONS:
        dim_vals = s1[dim].dropna()
        if len(dim_vals) == 0:
            continue
        overall_sd = dim_vals.std(ddof=1)
        if overall_sd > 0:
            print(f"    {DIM_LABELS[dim]:25s}: overall SD = {overall_sd:.4f}, "
                  f"σ/√2 = {overall_sd / np.sqrt(2):.4f}")


# ============== MAIN ==============

# ============== PROMPT SENSITIVITY ANALYSIS ==============

def get_prompt_sensitivity_data(df: pd.DataFrame) -> pd.DataFrame:
    """Load Study 1 (default prompt) + Study 5 (variant prompts) data."""
    s1 = df[(df["study"] == 1) & (df["model_id"].isin(STUDY1_PAPER_MODELS))].copy()
    s1 = s1[s1["thinking_mode"].fillna("chat") == "chat"]
    s1["prompt_variant"] = "default"
    s5 = df[(df["study"] == 5)].copy()
    combined = pd.concat([s1, s5], ignore_index=True)
    print(f"Prompt Sensitivity: {len(combined)} observations, "
          f"{combined['model_id'].nunique()} models, "
          f"variants: {sorted(combined['prompt_variant'].unique())}")
    return combined


def prompt_sensitivity_analysis(df: pd.DataFrame):
    """Analyze whether psychometric scores shift with prompt framing."""
    from scipy import stats

    ps = get_prompt_sensitivity_data(df)
    if len(ps) < 20:
        print("\n[Prompt Sensitivity] Insufficient data, skipping.")
        return None

    variants = sorted(ps["prompt_variant"].unique())
    models = ps["model_id"].unique()
    dims = [d for d in DIMENSIONS]  # 8 primary dimensions

    print(f"\n{'='*80}")
    print("PROMPT SENSITIVITY ANALYSIS")
    print(f"  Models: {len(models)}, Variants: {variants}, Dimensions: {len(dims)}")
    print(f"{'='*80}")

    # 1. Within-model prompt sensitivity (SD across prompts)
    print("\n--- 1. Within-Model Prompt Sensitivity (SD across 4 prompts) ---")
    sensitivity = {}
    for dim in dims:
        model_sds = []
        for mid in models:
            dim_scores = []
            for v in variants:
                v_data = ps[(ps["model_id"] == mid) & (ps["prompt_variant"] == v)]
                if len(v_data) > 0:
                    dim_scores.append(v_data[dim].mean())
            if len(dim_scores) >= 2:  # Need at least 2 variants
                model_sds.append(np.std(dim_scores))
        if model_sds:
            sensitivity[dim] = {
                "mean_sd": np.mean(model_sds),
                "median_sd": np.median(model_sds),
                "max_sd": np.max(model_sds),
            }
            print(f"  {dim:25s}: mean SD = {sensitivity[dim]['mean_sd']:.3f}, "
                  f"median = {sensitivity[dim]['median_sd']:.3f}, "
                  f"max = {sensitivity[dim]['max_sd']:.3f}")

    overall_mean_sd = np.mean([s["mean_sd"] for s in sensitivity.values()])
    print(f"\n  Overall mean within-model SD across prompts: {overall_mean_sd:.3f} (on 1-5 scale)")

    # 2. Cohen's d: each variant vs default
    print("\n--- 2. Effect Size: Variant vs Default (Cohen's d) ---")
    for dim in dims:
        ds = []
        for mid in models:
            default_data = ps[(ps["model_id"] == mid) & (ps["prompt_variant"] == "default")]
            if len(default_data) == 0:
                continue
            default_mean = default_data[dim].mean()
            default_std = default_data[dim].std()
            for v in variants:
                if v == "default":
                    continue
                v_data = ps[(ps["model_id"] == mid) & (ps["prompt_variant"] == v)]
                if len(v_data) == 0:
                    continue
                v_mean = v_data[dim].mean()
                pooled_std = np.sqrt((default_std**2 + v_data[dim].std()**2) / 2)
                if pooled_std > 0:
                    d = (v_mean - default_mean) / pooled_std
                    ds.append(d)
        if ds:
            print(f"  {dim:25s}: median |d| = {np.median(np.abs(ds)):.2f}, "
                  f"range [{np.min(ds):.2f}, {np.max(ds):.2f}], "
                  f"{sum(1 for d in ds if abs(d) > 0.5)}/{len(ds)} pairs with |d|>0.5")

    # 3. Rank stability (Spearman correlation of model rankings)
    print("\n--- 3. Rank Stability (Spearman ρ between model rankings) ---")
    for dim in dims:
        rhos = []
        for v1, v2 in [(variants[i], variants[j]) for i in range(len(variants)) for j in range(i+1, len(variants))]:
            means_v1 = ps[ps["prompt_variant"] == v1].groupby("model_id")[dim].mean()
            means_v2 = ps[ps["prompt_variant"] == v2].groupby("model_id")[dim].mean()
            common = means_v1.index.intersection(means_v2.index)
            if len(common) >= 5:
                rho, p = stats.spearmanr(means_v1[common], means_v2[common])
                rhos.append(rho)
        if rhos:
            print(f"  {dim:25s}: mean ρ = {np.mean(rhos):.3f}, "
                  f"min = {np.min(rhos):.3f}")

    # 4. Response style shift across prompts
    print("\n--- 4. Response Style Indicators by Prompt ---")
    for v in variants:
        v_data = ps[ps["prompt_variant"] == v]
        # Compute midpoint, extreme, acquiescence from raw responses
        midpoint_rates = []
        extreme_rates = []
        acquiescence_vals = []
        for _, row in v_data.iterrows():
            items = row.get("items", {})
            if not items:
                continue
            all_ratings = []
            for dim_ratings in items.values():
                if isinstance(dim_ratings, list):
                    all_ratings.extend(dim_ratings)
            if not all_ratings:
                continue
            midpoint_rates.append(sum(1 for r in all_ratings if r == 3) / len(all_ratings))
            extreme_rates.append(sum(1 for r in all_ratings if r in [1, 5]) / len(all_ratings))
            # Acquiescence: mean of positively-worded minus reverse-scored
            # Simplified: just use mean rating (higher = more agreeable)
            acquiescence_vals.append(np.mean(all_ratings))

        if midpoint_rates:
            print(f"  {v:12s}: midpoint = {np.mean(midpoint_rates)*100:.1f}%, "
                  f"extreme = {np.mean(extreme_rates)*100:.1f}%, "
                  f"mean rating = {np.mean(acquiescence_vals):.2f}")

    # 5. Variance decomposition
    print("\n--- 5. Variance Decomposition ---")
    for dim in dims:
        dim_data = ps[["model_id", "prompt_variant", dim]].copy()
        grand_mean = dim_data[dim].mean()
        ss_total = ((dim_data[dim] - grand_mean) ** 2).sum()
        # SS model
        model_means = dim_data.groupby("model_id")[dim].mean()
        ss_model = sum(len(ps[ps["model_id"] == m]) * (model_means[m] - grand_mean)**2 for m in model_means.index)
        # SS prompt (within model)
        ss_prompt = 0
        for mid in models:
            mid_data = dim_data[dim_data["model_id"] == mid]
            for v in variants:
                v_data = mid_data[mid_data["prompt_variant"] == v]
                if len(v_data) > 0:
                    v_mean = v_data[dim].mean()
                    n_v = len(v_data)
                    ss_prompt += n_v * (v_mean - model_means[mid])**2
        ss_residual = ss_total - ss_model - ss_prompt
        frac_model = ss_model / ss_total * 100 if ss_total > 0 else 0
        frac_prompt = ss_prompt / ss_total * 100 if ss_total > 0 else 0
        frac_residual = ss_residual / ss_total * 100 if ss_total > 0 else 0
        print(f"  {dim:25s}: Model={frac_model:.1f}%, Prompt={frac_prompt:.1f}%, Residual={frac_residual:.1f}%")

    return {
        "sensitivity": sensitivity,
        "overall_mean_sd": overall_mean_sd,
        "variants": variants,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model experiment results")
    parser.add_argument("--input", type=str, default="results/vendor_exp/*.json",
                        help="Glob pattern for result JSON files")
    parser.add_argument("--study", type=str, default=None,
                        choices=["1", "2", "ablation"],
                        help="Only run specific study analysis")
    args = parser.parse_args()

    df = load_data(args.input)

    if len(df) == 0:
        print("No data loaded!")
        exit(1)

    # Filter out pilot data
    df = df[~df.get("pilot", False)].copy() if "pilot" in df.columns else df.copy()

    # Print overview
    print(f"\nData overview:")
    print(f"  Total observations: {len(df)}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Unique model_ids: {df['model_id'].nunique()}")
    for study_num in sorted(df["study"].unique()):
        study_df = df[df["study"] == study_num]
        print(f"  Study {int(study_num)}: {len(study_df)} obs, {study_df['model'].nunique()} models")

    # Response quality checks
    if len(df) > 0:
        response_quality_check(df)
    else:
        print("\n  Skipping quality checks (no non-pilot data)")

    # Reliability analysis
    reliability_df = reliability_analysis(df) if len(df) > 0 else None

    # Inter-dimension correlations
    corr_matrix = inter_dimension_correlations(df) if len(df) > 0 else None

    # Alignment artifact analysis
    alignment_results = alignment_artifact_analysis(df) if len(df) > 0 else None

    # Alignment threshold sensitivity
    sensitivity_results = alignment_threshold_sensitivity(df) if len(df) > 0 else None

    # Convergent validity check
    validity_results = convergent_validity_check(df) if len(df) > 0 else None

    # Acquiescence bias analysis
    acquiescence_results = acquiescence_bias_analysis(df) if len(df) > 0 else None

    # HEXACO-H × model scale correlation (core finding)
    hexaco_scale_results = hexaco_scale_correlation(df) if len(df) > 0 else None

    # Study 1
    s1 = get_study1_data(df)
    if len(s1) >= 10 and args.study in (None, "1"):
        study1_results = study1_ols_anova(s1)
        study1_olr_results = study1_olr(s1)
        study1_d_results = study1_cohen_d(s1)
        # Floor-effect robustness: re-run ANOVA excluding floor-effected models
        floor_robustness_results = floor_effect_robustness(s1)
        # HEXACO-H descriptive observation (excluded from primary analysis)
        hexaco_desc = hexaco_descriptive_observation(s1)

    # Study 2
    if args.study in (None, "2"):
        s2 = get_study2_data(df)
        if len(s2) >= 5:
            study2_results = study2_descriptive(s2)

    # Thinking ablation
    if args.study in (None, "ablation"):
        ablation_results = thinking_ablation_analysis(df)

    # Prompt sensitivity (Study 5)
    if "prompt_variant" in df.columns and df["prompt_variant"].notna().any():
        prompt_results = prompt_sensitivity_analysis(df)

    # Power analysis
    power_analysis(df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
