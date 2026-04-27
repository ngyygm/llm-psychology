#!/usr/bin/env python3
"""
Focused analysis on the rescored data, producing exactly the numbers needed
for the rewritten paper:

  1. Overall cohort size, model counts per study
  2. Per-model Big Five means (corrected)
  3. Cross-model dimension correlation matrix (the new headline)
  4. Comparison of LLM vs human BFI-2 inter-dimension correlations
  5. ANOVA F, eta-squared, and median pairwise Cohen d (Study 1+2+6 main cohort)
  6. Reliability: ICC(1,1), Cronbach alpha
  7. Acquiescence/agreement bias indicators (PC1 variance share)
  8. Persona Fidelity Index and Covariance Fidelity if MBTI data present

Outputs a CSV summary plus an inline text report.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

CORRECTED_DIR = Path("results/vendor_exp/corrected")
MBTI_DIR = Path("results/mbti_persona/corrected")
OUT_DIR = Path("results/corrected_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BFI_LABELS = ["E", "A", "C", "N", "O"]
BFI_COLS = [f"bfi.{x.lower()}" for x in
            ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]]

# Soto & John 2017 BFI-2 correlations (community Norm Tables)
HUMAN_BFI2 = {
    ("E", "A"): 0.18, ("E", "C"): 0.11, ("E", "N"): -0.34, ("E", "O"): 0.22,
    ("A", "C"): 0.29, ("A", "N"): -0.25, ("A", "O"): 0.19,
    ("C", "N"): -0.30, ("C", "O"): 0.10,
    ("N", "O"): -0.08,
}


def load_corrected():
    rows = []
    for f in sorted(CORRECTED_DIR.glob("*.json")):
        if "checkpoint" in f.name:
            continue
        try:
            rows.extend(json.loads(f.read_text()))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    return df


def load_mbti_corrected():
    rows = []
    if not MBTI_DIR.exists():
        return pd.DataFrame()
    for f in sorted(MBTI_DIR.glob("study_mbti_*.json")):
        try:
            rows.extend(json.loads(f.read_text()))
        except Exception:
            pass
    return pd.DataFrame(rows)


def overview(df):
    print("=" * 80)
    print("OVERVIEW (corrected data)")
    print("=" * 80)
    print(f"  total records: {len(df)}")
    print(f"  unique models: {df['model_id'].nunique()}")
    if "study" in df.columns:
        for s in sorted(df["study"].dropna().unique()):
            sub = df[df["study"] == s]
            print(f"  Study {s}: {len(sub)} obs, {sub['model_id'].nunique()} models")


def main_cohort(df):
    """Studies 1, 2, 6 in chat mode under default prompt — main analysis."""
    main = df[(df["study"].isin([1, 2, 6])) &
              (df["thinking_mode"].fillna("chat") == "chat")].copy()
    if "prompt_variant" in main.columns:
        main = main[(main["prompt_variant"].fillna("") == "") |
                    (main["prompt_variant"] == "default")]
    return main


def per_model_means(df, dims=BFI_COLS):
    return df.groupby("model_id")[dims].mean()


def correlation_table(model_means_df):
    M = model_means_df.values
    R = np.corrcoef(M.T)
    cols = [c.replace("bfi.", "")[:1].upper() for c in model_means_df.columns]
    return pd.DataFrame(R, index=cols, columns=cols)


def report_corr_vs_human(R, n_models):
    print(f"\nDimension correlation matrix (across n = {n_models} models, BFI only)")
    print(R.round(3).to_string())
    print("\nLLM vs human BFI-2 (Soto & John 2017)")
    print(f"  {'Pair':<8} {'Human':>8} {'LLM':>8}  {'Direction'}")
    for (a, b), human_r in HUMAN_BFI2.items():
        if a in R.index and b in R.columns:
            llm_r = R.loc[a, b]
            sign = "FLIP" if np.sign(human_r) != np.sign(llm_r) and abs(llm_r) > 0.1 else "same"
            magn = "stronger" if abs(llm_r) > abs(human_r) + 0.1 else \
                   "weaker" if abs(llm_r) < abs(human_r) - 0.1 else "matched"
            print(f"  {a}-{b}  {human_r:>+8.2f} {llm_r:>+8.2f}  {sign:6s} ({magn})")


def cohen_d_pairwise(df, dim, n_workers=1):
    """Pairwise Cohen's d between every model pair on `dim`. Returns array of |d|.
    Uses Hedges-style pooled SD with a small floor to avoid division by zero
    when both groups happen to have zero within-model variance (which can
    happen when an LLM is deterministic on midpoint-clustered items)."""
    models = sorted(df["model_id"].unique())
    n = len(models)
    ds = []
    for i in range(n):
        for j in range(i + 1, n):
            a = df[df["model_id"] == models[i]][dim].values
            b = df[df["model_id"] == models[j]][dim].values
            if len(a) < 2 or len(b) < 2:
                continue
            sa, sb = a.std(ddof=1), b.std(ddof=1)
            pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
            if pooled < 0.01:
                pooled = 0.01
            ds.append(abs(a.mean() - b.mean()) / pooled)
    return np.array(ds)


def family_anova(df, dim):
    groups = [g[dim].values for _, g in df.groupby("model_id") if len(g) > 1]
    if len(groups) < 2:
        return None
    F, p = stats.f_oneway(*groups)
    grand_mean = df[dim].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum((x - grand_mean) ** 2 for g in groups for x in g)
    eta2 = ss_between / ss_total if ss_total > 0 else 0
    return F, p, eta2


def cohen_d_summary(df, dims=BFI_COLS):
    print("\nPairwise Cohen's d (across model pairs)")
    print(f"{'Dim':>6}  {'F':>8}  {'p':>10}  {'eta2':>6}  {'median d':>10}  {'max d':>8}")
    medians = []
    for dim in dims:
        anova = family_anova(df, dim)
        ds = cohen_d_pairwise(df, dim)
        if anova:
            F, p, e = anova
            print(f"  {dim.replace('bfi.', '')[:4]:>4}  {F:>8.2f}  {p:>10.2e}  {e:>6.3f}  "
                  f"{np.median(ds):>10.2f}  {ds.max():>8.2f}")
            medians.append(np.median(ds))
    print(f"  Median across BFI dims: {np.median(medians):.2f}")
    return medians


def pca_variance(df, dims=BFI_COLS):
    means = df.groupby("model_id")[dims].mean()
    X = (means - means.mean()) / means.std()
    pca = PCA().fit(X)
    print("\nPCA explained variance ratio (first 5)")
    for i, v in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {v*100:.1f}%")
    print(f"  PC1 loadings: {dict(zip(BFI_LABELS, pca.components_[0].round(3)))}")
    return pca


def mbti_pfi(mbti_df, default_means):
    """Persona Fidelity Index per model.
    Compares observed mean shift (persona vs default) to theoretical direction
    encoded in MBTI_TYPES[type]['big5_pred']."""
    from run_mbti_personas import MBTI_TYPES
    dim_map = {"bfi.extraversion": "E", "bfi.agreeableness": "A",
               "bfi.conscientiousness": "C", "bfi.neuroticism": "N_FFM",
               "bfi.openness": "O"}
    rows = []
    if mbti_df.empty:
        return pd.DataFrame()
    for model_id, sub in mbti_df.groupby("model_id"):
        per_persona = sub.groupby("mbti_type")[BFI_COLS].mean()
        n_pred = 0
        n_hit = 0
        per_dim_hit = defaultdict(lambda: [0, 0])
        for persona, row in per_persona.iterrows():
            base = default_means.get(model_id, {})
            for col in BFI_COLS:
                pred = MBTI_TYPES[persona]["big5_pred"].get(dim_map[col], 0)
                if pred == 0:
                    continue
                obs = row[col] - base.get(col, np.nan)
                if np.isnan(obs) or abs(obs) < 0.05:
                    continue
                n_pred += 1
                hit = 1 if np.sign(obs) == np.sign(pred) else 0
                n_hit += hit
                per_dim_hit[col][0] += 1
                per_dim_hit[col][1] += hit
        pfi = (2 * (n_hit / n_pred) - 1) if n_pred > 0 else np.nan
        rows.append({"model_id": model_id, "n_predictions": n_pred,
                     "PFI": round(pfi, 3),
                     "hit_rate": round(n_hit / n_pred, 3) if n_pred else np.nan,
                     **{f"hit_{col[-1]}": f"{v[1]}/{v[0]}"
                        for col, v in per_dim_hit.items()}})
    return pd.DataFrame(rows)


def covariance_fidelity(mbti_df):
    """1 minus Frobenius distance between persona-induced 5x5 corr matrix
    and human BFI-2 matrix, per model."""
    if mbti_df.empty:
        return pd.DataFrame()
    H = np.eye(5)
    for (a, b), r in HUMAN_BFI2.items():
        i, j = BFI_LABELS.index(a), BFI_LABELS.index(b)
        H[i, j] = r
        H[j, i] = r
    rows = []
    for model_id, sub in mbti_df.groupby("model_id"):
        means = sub.groupby("mbti_type")[BFI_COLS].mean()
        if len(means) < 5:
            continue
        R = np.corrcoef(means.values.T)
        diff = np.linalg.norm(R - H, ord="fro")
        base = np.linalg.norm(H, ord="fro")
        cf = 1 - diff / base if base > 0 else np.nan
        cn = means[[BFI_COLS[2], BFI_COLS[3]]]
        r_cn, p_cn = stats.pearsonr(cn[BFI_COLS[2]], cn[BFI_COLS[3]])
        rows.append({"model_id": model_id, "n_personas": len(means),
                     "CovFid": round(cf, 3),
                     "r_CN_persona": round(r_cn, 3),
                     "p_CN": round(p_cn, 4)})
    return pd.DataFrame(rows)


def main():
    df = load_corrected()
    overview(df)
    main_df = main_cohort(df)
    print(f"\nMain cohort (Study 1+2+6, default prompt, chat mode): "
          f"{len(main_df)} obs, {main_df['model_id'].nunique()} models")

    means = per_model_means(main_df, BFI_COLS)
    R = correlation_table(means)
    report_corr_vs_human(R, n_models=len(means))

    cohen_d_summary(main_df, BFI_COLS)
    pca = pca_variance(main_df)

    means.to_csv(OUT_DIR / "model_means_corrected.csv")
    R.to_csv(OUT_DIR / "corr_matrix_corrected.csv")
    print(f"\nSaved per-model means + correlation matrix to {OUT_DIR}/")

    mbti_df = load_mbti_corrected()
    if not mbti_df.empty and "mbti_type" in mbti_df.columns:
        print(f"\nMBTI persona records: {len(mbti_df)}, models: {mbti_df['model_id'].nunique()}")
        default = means.to_dict("index")
        pfi_df = mbti_pfi(mbti_df, default)
        cov_df = covariance_fidelity(mbti_df)
        print("\nPersona Fidelity Index (per model):")
        if not pfi_df.empty:
            print(pfi_df.to_string(index=False))
        print("\nCovariance Fidelity (per model):")
        if not cov_df.empty:
            print(cov_df.to_string(index=False))
        if not pfi_df.empty:
            pfi_df.to_csv(OUT_DIR / "mbti_pfi.csv", index=False)
        if not cov_df.empty:
            cov_df.to_csv(OUT_DIR / "mbti_covfid.csv", index=False)


if __name__ == "__main__":
    main()
