"""
Deep Psychometric Validation of LLM Personality Measurement
============================================================
Implements rigorous psychometric analyses beyond descriptive statistics:

Tier 1 (Core):
  1. EFA factor structure analysis — compare LLM to expected Big Five
  2. Pairwise Inconsistency Rate (PIR) + SDR cross-validation
  3. Crossed random-effects variance decomposition

Tier 2 (Strengthening):
  4. DIF analysis across model families
  5. Cross-scale convergent validity (enhanced)
  6. Response style quantification
  7. MBTI persona measurement invariance

References:
  - Serapio-Garcia et al. (2025) Nature Machine Intelligence
  - Suhr et al. (2025) Challenging Validity of Personality Tests for LLMs
  - Acerbi & Stubbersfield (2024) PNAS Nexus
  - Kriegmair & Wulff (2026) Crossed random-effects for LLM behavior
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

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
            "item_id": r["item_id"],
            "scale": r["scale"],
            "domain": r["domain"],
            "facet": r["facet"],
            "item_text": r["item_text"],
            "keyed": r["keyed"],
            "parsed_value": r["parsed_value"],
            "scored_value": r["scored_value"],
            "response_format": r["response_format"],
        })
    df = pd.DataFrame(rows)
    df["parsed_value"] = pd.to_numeric(df["parsed_value"], errors="coerce")
    df["scored_value"] = pd.to_numeric(df["scored_value"], errors="coerce")
    # Single model extraction: impute nulls with per-item median across same domain
    null_mask = df["parsed_value"].isna()
    if null_mask.any():
        for idx in df[null_mask].index:
            row = df.loc[idx]
            same_domain = df[(df["domain"] == row["domain"]) & (df["scale"] == row["scale"]) & df["parsed_value"].notna()]
            if len(same_domain) > 0:
                med = same_domain["parsed_value"].median()
                df.at[idx, "parsed_value"] = med
                keyed = row["keyed"]
                df.at[idx, "scored_value"] = med if keyed == "+" else (6 - med if row["response_format"] == "likert_5" else med)
    return df


def build_full_response_matrix(all_results):
    rows = []
    for model_name, model_data in all_results.items():
        for persona in model_data["results_by_persona"]:
            for r in model_data["results_by_persona"][persona]["responses"]:
                rows.append({
                    "model": model_name,
                    "persona": persona,
                    "item_id": r["item_id"],
                    "scale": r["scale"],
                    "domain": r["domain"],
                    "facet": r["facet"],
                    "keyed": r["keyed"],
                    "parsed_value": r["parsed_value"],
                    "scored_value": r["scored_value"],
                    "response_format": r["response_format"],
                })
    df = pd.DataFrame(rows)
    _impute_missing_with_median(df)
    return df


def _impute_missing_with_median(df):
    """Impute null parsed_value/scored_value with per-item median across models."""
    df["parsed_value"] = pd.to_numeric(df["parsed_value"], errors="coerce")
    df["scored_value"] = pd.to_numeric(df["scored_value"], errors="coerce")
    null_mask = df["parsed_value"].isna()
    if not null_mask.any():
        return
    n_missing = null_mask.sum()
    for idx in df[null_mask].index:
        row = df.loc[idx]
        item_id = row["item_id"]
        persona = row["persona"]
        same = df[(df["item_id"] == item_id) & (df["persona"] == persona) & df["parsed_value"].notna()]
        if len(same) == 0:
            continue
        med = same["parsed_value"].median()
        df.at[idx, "parsed_value"] = med
        keyed = row["keyed"]
        df.at[idx, "scored_value"] = med if keyed == "+" else (6 - med if row["response_format"] == "likert_5" else med)
    print(f"[Imputation] Filled {n_missing} null values with per-item median")


# ─────────────────────────────────────────────────────────────
# Analysis 1: Exploratory Factor Analysis (EFA)
# ─────────────────────────────────────────────────────────────

def _principal_factor_analysis(corr_matrix, n_factors):
    """Principal axis factoring using numpy/scipy."""
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top n_factors
    loadings = eigenvectors[:, :n_factors] * np.sqrt(np.maximum(eigenvalues[:n_factors], 0))

    # Promax rotation (simplified: use varimax as approximation)
    loadings = _varimax_rotation(loadings)

    # Compute variance explained
    var_explained = np.sum(loadings**2, axis=0)
    total_var = np.trace(corr_matrix)
    prop_var = var_explained / total_var
    cum_var = np.cumsum(prop_var)

    return loadings, var_explained, prop_var, cum_var


def _varimax_rotation(loadings, max_iter=100, tol=1e-6):
    """Varimax rotation for factor loadings."""
    n, k = loadings.shape
    rotation = np.eye(k)
    d = 0
    for _ in range(max_iter):
        Lam = loadings @ rotation
        grad = Lam * (Lam**2 - np.mean(Lam**2, axis=0))
        U, S, Vt = np.linalg.svd(loadings.T @ grad)
        rotation = U @ Vt
        d_new = np.sum(S)
        if abs(d_new - d) < tol:
            break
        d = d_new
    return loadings @ rotation


def run_efa_analysis(all_results):
    print("\n" + "=" * 70)
    print("ANALYSIS 1: EXPLORATORY FACTOR ANALYSIS")
    print("=" * 70)

    # ── Step 1: Item-level variance analysis ──
    print("\n--- Item-Level Variance Analysis (IPIP-NEO-120) ---")
    ipip_rows = []
    row_labels = []
    for model_name, model_data in all_results.items():
        for persona in model_data["results_by_persona"]:
            items = extract_item_responses(model_data, persona)
            ipip = items[items["scale"] == "IPIP-NEO-120"].sort_values("item_id")
            if len(ipip) == 120:
                ipip_rows.append(ipip["parsed_value"].values.astype(float))
                row_labels.append((model_name, persona))

    X_items = np.array(ipip_rows)
    item_ids = sorted(
        extract_item_responses(list(all_results.values())[0], "Default")
        .query("scale == 'IPIP-NEO-120'")["item_id"].tolist()
    )

    print(f"Item-level matrix: {X_items.shape[0]} obs x {X_items.shape[1]} items")

    col_vars = X_items.var(axis=0)
    n_zero_var = int(np.sum(col_vars < 1e-10))
    n_low_var = int(np.sum(col_vars < 0.1))
    n_decent_var = int(np.sum(col_vars >= 0.3))
    print(f"Zero-variance items: {n_zero_var}/{len(item_ids)}")
    print(f"Low-variance items (<0.1): {n_low_var}/{len(item_ids)}")
    print(f"Decent-variance items (>=0.3): {n_decent_var}/{len(item_ids)}")

    if n_zero_var > 50:
        print(f"\n*** CRITICAL: {n_zero_var}/120 items have zero variance — LLMs converge on same answers ***")

    # ── Step 2: Domain-level EFA ──
    print("\n--- Domain-Level Factor Analysis ---")
    domain_names = [
        "IPIP_Neuroticism", "IPIP_Extraversion", "IPIP_Openness", "IPIP_Agreeableness", "IPIP_Conscientiousness",
        "SD3_Machiavellianism", "SD3_Narcissism", "SD3_Psychopathy",
        "ZKPQ_Activity", "ZKPQ_Aggression", "ZKPQ_ImpulsiveSS", "ZKPQ_NeuroticismA", "ZKPQ_Sociability",
        "EPQR_Psychoticism", "EPQR_Extraversion", "EPQR_Neuroticism", "EPQR_Lie",
    ]

    domain_keys = [
        "IPIP-NEO-120::Neuroticism", "IPIP-NEO-120::Extraversion",
        "IPIP-NEO-120::Openness", "IPIP-NEO-120::Agreeableness",
        "IPIP-NEO-120::Conscientiousness",
        "SD3::Machiavellianism", "SD3::Narcissism", "SD3::Psychopathy",
        "ZKPQ-50-CC::Activity", "ZKPQ-50-CC::Aggression-Hostility",
        "ZKPQ-50-CC::Impulsive_Sensation_Seeking", "ZKPQ-50-CC::Neuroticism-Anxiety",
        "ZKPQ-50-CC::Sociability",
        "EPQR-A::Psychoticism", "EPQR-A::Extraversion",
        "EPQR-A::Neuroticism", "EPQR-A::Lie",
    ]

    domain_rows = []
    domain_labels = []
    for model_name, model_data in all_results.items():
        for persona in model_data["results_by_persona"]:
            ds = model_data["results_by_persona"][persona]["domain_scores"]
            row = [ds.get(key, {}).get("mean_score", np.nan) for key in domain_keys]
            domain_rows.append(row)
            domain_labels.append((model_name, persona))

    X_domain = np.array(domain_rows)
    valid_mask = ~np.any(np.isnan(X_domain), axis=1)
    X_domain_clean = X_domain[valid_mask]

    print(f"Domain-level matrix: {X_domain_clean.shape[0]} obs x {X_domain_clean.shape[1]} domains")

    # Standardize
    X_std = (X_domain_clean - X_domain_clean.mean(axis=0)) / (X_domain_clean.std(axis=0) + 1e-10)
    corr = np.corrcoef(X_std, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(corr)[::-1]

    print(f"\nEigenvalues:")
    for i, ev in enumerate(eigenvalues):
        print(f"  {domain_names[i]:>28s}: {ev:.3f}")

    n_factors_kaiser = int(np.sum(eigenvalues > 1.0))
    print(f"\nKaiser criterion: {n_factors_kaiser} factors")
    print(f"Expected: ~7-8 distinct factors")

    # Parallel analysis
    np.random.seed(42)
    n_pa_runs = 500
    n_eigs = X_domain_clean.shape[1]
    pa_eigs = np.zeros((n_pa_runs, n_eigs))
    for i in range(n_pa_runs):
        X_rand = np.column_stack([
            np.random.permutation(X_domain_clean[:, j]) for j in range(n_eigs)
        ])
        pa_eigs[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(X_rand, rowvar=False)))[::-1]

    pa_95 = np.percentile(pa_eigs, 95, axis=0)
    n_factors_pa = int(np.sum(eigenvalues > pa_95))
    ev_ratio = eigenvalues[0] / eigenvalues[1] if len(eigenvalues) > 1 and eigenvalues[1] > 0 else 0

    print(f"Parallel analysis: {n_factors_pa} factors")
    print(f"Eigenvalue ratio (1st/2nd): {ev_ratio:.2f} (>3.0 = dominant general factor)")
    print(f"First 3 eigenvalues explain: {sum(eigenvalues[:3])/sum(eigenvalues)*100:.1f}% of total variance")

    # Run EFA with n_factors_pa factors
    n_f = max(n_factors_pa, 3)
    loadings_5, var_exp, prop_var, cum_var = _principal_factor_analysis(corr, n_f)

    print(f"\n--- {n_f}-Factor Solution (Domain-Level EFA) ---")
    loadings_df = pd.DataFrame(
        loadings_5, index=domain_names,
        columns=[f"F{i+1}" for i in range(n_f)]
    )
    print("Factor loadings (sorted by absolute loading per factor):")
    print(loadings_df.round(3).to_string())

    print(f"\nVariance explained:")
    for i in range(n_f):
        print(f"  F{i+1}: {prop_var[i]*100:.1f}%")
    print(f"  Total: {cum_var[-1]*100:.1f}%")

    # Big Five alignment
    ipip_idx = [0, 1, 2, 3, 4]
    ipip_names = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]

    print("\n--- Big Five Factor Alignment ---")
    for i, idx in enumerate(ipip_idx):
        best_f = int(np.argmax(np.abs(loadings_5[idx, :])))
        best_l = loadings_5[idx, best_f]
        print(f"  {ipip_names[i]:>20s} -> F{best_f+1} (loading = {best_l:.3f})")

    assigned = [int(np.argmax(np.abs(loadings_5[idx, :]))) for idx in ipip_idx]
    n_unique = len(set(assigned))
    print(f"\nDistinct factor assignments: {n_unique}/5 {'PRESERVED' if n_unique == 5 else 'COLLAPSED'}")

    # ── Step 3: Per-model factor structure ──
    print("\n--- Per-Model Factor Structure (Domain-Level) ---")
    model_factor_results = []

    for model_name in sorted(all_results.keys()):
        model_rows = []
        for persona in all_results[model_name]["results_by_persona"]:
            ds = all_results[model_name]["results_by_persona"][persona]["domain_scores"]
            row = [ds.get(key, {}).get("mean_score", np.nan) for key in domain_keys]
            model_rows.append(row)

        X_m = np.array(model_rows)
        if np.any(np.isnan(X_m)):
            continue

        try:
            X_m_std = (X_m - X_m.mean(axis=0)) / (X_m.std(axis=0) + 1e-10)
            corr_m = np.corrcoef(X_m_std, rowvar=False)
            loadings_m, _, _, _ = _principal_factor_analysis(corr_m, n_f)

            max_cong = []
            used = set()
            for fi in range(n_f):
                cands = [(fj, np.dot(loadings_5[:, fi], loadings_m[:, fj]) /
                          (np.linalg.norm(loadings_5[:, fi]) * np.linalg.norm(loadings_m[:, fj]) + 1e-10))
                         for fj in range(n_f) if fj not in used]
                if cands:
                    best = max(cands, key=lambda x: abs(x[1]))
                    used.add(best[0])
                    max_cong.append(abs(best[1]))

            model_factor_results.append({
                "model": model_name,
                "mean_congruence": np.mean(max_cong) if max_cong else np.nan,
                "min_congruence": np.min(max_cong) if max_cong else np.nan,
            })
        except Exception:
            pass

    model_factor_df = pd.DataFrame(model_factor_results)
    if len(model_factor_df) > 0:
        print(model_factor_df.sort_values("mean_congruence", ascending=False).to_string(index=False))
        print(f"\nMean Tucker's phi: {model_factor_df['mean_congruence'].mean():.3f}")

    # Save
    loadings_df.to_csv(OUTPUT_DIR / "efa_domain_loadings.csv")
    pd.DataFrame({"domain": domain_names, "eigenvalue": eigenvalues}).to_csv(
        OUTPUT_DIR / "efa_eigenvalues.csv", index=False)
    if len(model_factor_df) > 0:
        model_factor_df.to_csv(OUTPUT_DIR / "efa_per_model_congruence.csv", index=False)

    return {
        "n_factors_pa": n_factors_pa,
        "n_factors_kaiser": n_factors_kaiser,
        "n_zero_var_items": n_zero_var,
        "eigenvalue_ratio": ev_ratio,
        "model_factor_df": model_factor_df,
        "loadings_df": loadings_df,
    }


# ─────────────────────────────────────────────────────────────
# Analysis 2: Pairwise Inconsistency Rate (PIR) + SDR
# ─────────────────────────────────────────────────────────────

def run_pir_sdr_analysis(all_results):
    print("\n" + "=" * 70)
    print("ANALYSIS 2: PAIRWISE INCONSISTENCY RATE + SOCIAL DESIRABILITY")
    print("=" * 70)

    sdr_directions = {
        ("IPIP-NEO-120", "Agreeableness"): +1,
        ("IPIP-NEO-120", "Conscientiousness"): +1,
        ("IPIP-NEO-120", "Extraversion"): +0.5,
        ("IPIP-NEO-120", "Openness"): +0.5,
        ("IPIP-NEO-120", "Neuroticism"): -1,
        ("SD3", "Machiavellianism"): -1,
        ("SD3", "Narcissism"): -0.5,
        ("SD3", "Psychopathy"): -1,
        ("EPQR-A", "Lie"): -0.5,
    }

    pir_rows = []
    sdr_rows = []

    for model_name in sorted(all_results.keys()):
        items_df = extract_item_responses(all_results[model_name], "Default")

        for (scale, domain), group in items_df.groupby(["scale", "domain"]):
            fwd = group[group["keyed"] == "+"]
            rev = group[group["keyed"] == "-"]
            if len(fwd) == 0 or len(rev) == 0:
                continue

            resp_format = group["response_format"].values[0]
            n_inconsistent = 0
            n_total_pairs = 0
            for _, fwd_row in fwd.iterrows():
                for _, rev_row in rev.iterrows():
                    n_total_pairs += 1
                    if resp_format == "likert_5":
                        if (fwd_row["parsed_value"] >= 3 and rev_row["parsed_value"] >= 3):
                            n_inconsistent += 1
                        elif (fwd_row["parsed_value"] <= 3 and rev_row["parsed_value"] <= 3):
                            n_inconsistent += 1
                    else:
                        if fwd_row["parsed_value"] == rev_row["parsed_value"]:
                            n_inconsistent += 1

            pir = n_inconsistent / n_total_pairs if n_total_pairs > 0 else np.nan
            pir_rows.append({
                "model": model_name,
                "scale": scale,
                "domain": domain,
                "pir": pir,
                "fwd_raw_mean": fwd["parsed_value"].mean(),
                "rev_raw_mean": rev["parsed_value"].mean(),
                "fwd_scored_mean": fwd["scored_value"].mean(),
                "rev_scored_mean": rev["scored_value"].mean(),
                "n_fwd": len(fwd),
                "n_rev": len(rev),
            })

        ds = all_results[model_name]["results_by_persona"]["Default"]["domain_scores"]
        sdr_composite = 0.0
        n_sdr_dims = 0
        for (scale, domain), direction in sdr_directions.items():
            key = f"{scale}::{domain}"
            if key in ds:
                score = ds[key]["mean_score"]
                if scale in ["IPIP-NEO-120", "SD3"]:
                    norm_score = (score - 1) / 4.0
                else:
                    norm_score = score
                sdr_composite += direction * norm_score
                n_sdr_dims += 1

        sdr_index = sdr_composite / n_sdr_dims if n_sdr_dims > 0 else np.nan

        likert_items = items_df[items_df["response_format"] == "likert_5"]
        acquiescence = (likert_items["parsed_value"].mean() - 3.0) / 2.0 if len(likert_items) > 0 else np.nan
        extreme_rate = ((likert_items["parsed_value"] == 1) | (likert_items["parsed_value"] == 5)).mean() if len(likert_items) > 0 else np.nan
        midpoint_rate = (likert_items["parsed_value"] == 3).mean() if len(likert_items) > 0 else np.nan

        lie_items = items_df[(items_df["scale"] == "EPQR-A") & (items_df["domain"] == "Lie")]
        lie_score = lie_items["scored_value"].mean() if len(lie_items) > 0 else np.nan

        sdr_rows.append({
            "model": model_name,
            "sdr_composite": sdr_index,
            "acquiescence": acquiescence,
            "extreme_response": extreme_rate,
            "midpoint_response": midpoint_rate,
            "lie_scale": lie_score,
        })

    pir_df = pd.DataFrame(pir_rows)
    sdr_df = pd.DataFrame(sdr_rows)

    print("\n--- PIR by Domain ---")
    pir_summary = pir_df.groupby(["scale", "domain"]).agg(
        mean_pir=("pir", "mean"), std_pir=("pir", "std"),
        min_pir=("pir", "min"), max_pir=("pir", "max"),
    ).reset_index()
    print(pir_summary.to_string(index=False))
    print(f"\nOverall mean PIR: {pir_df['pir'].mean():.3f}")

    print("\n--- SDR by Model ---")
    print(sdr_df.sort_values("sdr_composite", ascending=False).to_string(index=False))
    print(f"\nMean SDR composite: {sdr_df['sdr_composite'].mean():.3f}")

    # PIR x SDR cross-validation
    print("\n--- PIR x SDR Cross-Validation ---")
    pir_model = pir_df.groupby("model")["pir"].mean().reset_index()
    pir_model.columns = ["model", "mean_pir"]
    pir_sdr = pir_model.merge(sdr_df[["model", "sdr_composite"]], on="model")

    if len(pir_sdr) > 3:
        r_pir_sdr, p_pir_sdr = stats.spearmanr(pir_sdr["mean_pir"], pir_sdr["sdr_composite"])
        print(f"Spearman r(PIR, SDR) = {r_pir_sdr:.3f}, p = {p_pir_sdr:.4f}")
        if r_pir_sdr > 0:
            print("-> Higher SDR -> more inconsistency (supports alignment-distortion)")
        else:
            print("-> No positive PIR-SDR relationship")

    # PIR by persona
    print("\n--- PIR: Default vs MBTI Persona ---")
    persona_pir_rows = []
    for model_name in sorted(all_results.keys()):
        for persona in all_results[model_name]["results_by_persona"]:
            items_df = extract_item_responses(all_results[model_name], persona)
            for (scale, domain), group in items_df.groupby(["scale", "domain"]):
                fwd = group[group["keyed"] == "+"]
                rev = group[group["keyed"] == "-"]
                if len(fwd) == 0 or len(rev) == 0:
                    continue
                resp_format = group["response_format"].values[0]
                if resp_format == "likert_5":
                    agreement = 1.0 - abs(fwd["scored_value"].mean() - rev["scored_value"].mean()) / 4.0
                else:
                    agreement = 1.0 - abs(fwd["scored_value"].mean() - rev["scored_value"].mean())
                persona_pir_rows.append({
                    "model": model_name, "persona": persona,
                    "condition": "Default" if persona == "Default" else "MBTI",
                    "scale": scale, "domain": domain, "agreement": agreement,
                })

    persona_pir_df = pd.DataFrame(persona_pir_rows)
    default_pir = persona_pir_df[persona_pir_df["condition"] == "Default"].groupby("model")["agreement"].mean()
    mbti_pir = persona_pir_df[persona_pir_df["condition"] == "MBTI"].groupby("model")["agreement"].mean()

    print(f"Mean agreement (Default): {default_pir.mean():.3f}")
    print(f"Mean agreement (MBTI avg): {mbti_pir.mean():.3f}")
    t_stat, p_val = stats.ttest_rel(mbti_pir.values, default_pir.values)
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")

    print("\n--- MBTI Persona PIR (ranked by agreement) ---")
    persona_rank = persona_pir_df[persona_pir_df["condition"] == "MBTI"].groupby("persona")["agreement"].mean().sort_values()
    for persona, agreement in persona_rank.items():
        print(f"  {persona}: {agreement:.3f}")

    # PIR variance decomposition
    print("\n--- PIR Variance Decomposition ---")
    grand_mean = pir_df["pir"].mean()
    ss_total = ((pir_df["pir"] - grand_mean) ** 2).sum()
    ss_model = pir_df.groupby("model")["pir"].apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
    ss_domain = pir_df.groupby(["scale", "domain"])["pir"].apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()

    var_model = ss_model / ss_total * 100
    var_domain = ss_domain / ss_total * 100
    var_residual = 100 - var_model - var_domain

    print(f"  Model: {var_model:.1f}%")
    print(f"  Domain: {var_domain:.1f}%")
    print(f"  Residual: {var_residual:.1f}%")

    # Save
    pir_df.to_csv(OUTPUT_DIR / "pir_by_model_domain.csv", index=False)
    sdr_df.to_csv(OUTPUT_DIR / "sdr_by_model.csv", index=False)
    pir_sdr.to_csv(OUTPUT_DIR / "pir_sdr_crossvalidation.csv", index=False)
    persona_pir_df.to_csv(OUTPUT_DIR / "pir_by_persona.csv", index=False)

    return {
        "pir_df": pir_df,
        "sdr_df": sdr_df,
        "pir_sdr": pir_sdr,
        "var_model": var_model,
        "var_domain": var_domain,
    }


# ─────────────────────────────────────────────────────────────
# Analysis 3: Variance Decomposition
# ─────────────────────────────────────────────────────────────

def run_variance_decomposition(all_results):
    print("\n" + "=" * 70)
    print("ANALYSIS 3: VARIANCE DECOMPOSITION")
    print("=" * 70)

    df = build_full_response_matrix(all_results)

    df["norm_score"] = np.nan
    likert_mask = df["response_format"] == "likert_5"
    df.loc[likert_mask, "norm_score"] = (df.loc[likert_mask, "scored_value"] - 1) / 4.0
    binary_mask = df["response_format"].isin(["true_false", "yes_no"])
    df.loc[binary_mask, "norm_score"] = df.loc[binary_mask, "scored_value"]

    df["scale_domain"] = df["scale"] + "::" + df["domain"]

    results = {}
    for label, sub_df in [("Likert (IPIP+SD3)", df[df["response_format"] == "likert_5"]),
                           ("Binary (ZKPQ+EPQR)", df[df["response_format"].isin(["true_false", "yes_no"])])]:
        print(f"\n--- {label} ---")
        print(f"N: {len(sub_df)}, Models: {sub_df['model'].nunique()}, "
              f"Domains: {sub_df['scale_domain'].nunique()}, Items: {sub_df['item_id'].nunique()}")

        grand_mean = sub_df["norm_score"].mean()
        ss_total = ((sub_df["norm_score"] - grand_mean) ** 2).sum()

        ss_model = sub_df.groupby("model")["norm_score"].apply(
            lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
        ss_domain = sub_df.groupby("scale_domain")["norm_score"].apply(
            lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
        ss_persona = sub_df.groupby("persona")["norm_score"].apply(
            lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
        ss_item = sub_df.groupby("item_id")["norm_score"].apply(
            lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
        ss_residual = ss_total - ss_model - ss_domain - ss_persona - ss_item

        print("\nVariance decomposition:")
        for name, ss in [("Model", ss_model), ("Domain", ss_domain),
                         ("Persona", ss_persona), ("Item", ss_item),
                         ("Residual", ss_residual)]:
            pct = ss / ss_total * 100 if ss_total > 0 else 0
            print(f"  {name:>12s}: {pct:5.1f}%")

        results[label] = {
            "model": ss_model / ss_total if ss_total > 0 else 0,
            "domain": ss_domain / ss_total if ss_total > 0 else 0,
            "persona": ss_persona / ss_total if ss_total > 0 else 0,
            "item": ss_item / ss_total if ss_total > 0 else 0,
            "residual": ss_residual / ss_total if ss_total > 0 else 0,
        }

    print("\n--- Interpretation ---")
    for label, vc in results.items():
        total = sum(vc.values())
        model_pct = vc["model"] / total * 100
        domain_pct = vc["domain"] / total * 100
        item_pct = vc["item"] / total * 100
        print(f"\n  {label}:")
        print(f"    Model (machine personality): {model_pct:.1f}%")
        print(f"    Domain (trait): {domain_pct:.1f}%")
        print(f"    Item (stimulus): {item_pct:.1f}%")
        if model_pct < 10:
            print("    -> Model explains <10%: weak 'machine personality'")
        if domain_pct > 40:
            print("    -> Domain explains >40%: responses are trait-driven")

    var_df = pd.DataFrame([
        {"analysis": label, "component": comp, "fraction": var, "percentage": var * 100}
        for label, vc in results.items()
        for comp, var in vc.items()
    ])
    var_df.to_csv(OUTPUT_DIR / "variance_decomposition.csv", index=False)

    return results


# ─────────────────────────────────────────────────────────────
# Analysis 4: DIF + Convergent Validity
# ─────────────────────────────────────────────────────────────

def run_dif_analysis(all_results):
    print("\n" + "=" * 70)
    print("ANALYSIS 4: DIF + CONVERGENT VALIDITY")
    print("=" * 70)

    model_families = {
        "OpenAI": ["GPT_5.2", "GPT-5.5"],
        "Anthropic": ["Claude-Opus-4.6", "Claude-Sonnet-4.6"],
        "Google": ["Gemini_3-Pro-Preview", "Gemini-3-Flash-Preview",
                    "Gemini-3.1-Pro-Preview", "Gemini-3.1-Flash-Lite"],
        "Chinese": ["DeepSeek-V3.2", "DeepSeek-V4-Flash", "DeepSeek-V4-Pro",
                     "GLM-4.6V", "Kimi-K2.5", "Kimi-K2.6",
                     "MiniMax-M2.7", "Qwen3-235B-A22B",
                     "Qwen3.5-122B-A10B", "Qwen3.5-397B-A17B"],
    }

    all_items = []
    for model_name, model_data in all_results.items():
        family = "Unknown"
        for fam, models in model_families.items():
            if model_name in models:
                family = fam
                break
        items_df = extract_item_responses(model_data, "Default")
        items_df["model_family"] = family
        all_items.append(items_df)

    df = pd.concat(all_items, ignore_index=True)
    df_likert = df[df["response_format"] == "likert_5"].copy()

    # Item variance collapse
    print("\n--- Item Variance Collapse ---")
    item_var = df_likert.groupby("item_id").agg(
        variance=("parsed_value", "var"),
        mean=("parsed_value", "mean"),
        scale=("scale", "first"),
        domain=("domain", "first"),
        keyed=("keyed", "first"),
        item_text=("item_text", "first"),
    ).reset_index()

    low_var = item_var[item_var["variance"] < 0.3].sort_values("variance")
    print(f"Low-variance items (<0.3): {len(low_var)}/{len(item_var)}")

    if len(low_var) > 0:
        high = low_var[low_var["mean"] > 3.5]
        low = low_var[low_var["mean"] < 2.5]
        print(f"  High agreement (>3.5): {len(high)}, Low agreement (<2.5): {len(low)}")

    # Safety keywords
    safety_keywords = ["harm", "hurt", "lie", "cheat", "steal", "kill", "danger", "weapon",
                       "manipulate", "deceive", "cruel", "violent", "abuse", "threat",
                       "break", "rules", "law", "punish", "revenge", "angry"]

    safety_items = []
    for _, row in low_var.iterrows():
        text = row.get("item_text", "")
        if text and any(kw in text.lower() for kw in safety_keywords):
            safety_items.append(row)

    if safety_items:
        print(f"\nSafety-contaminated items: {len(safety_items)}")
        for row in safety_items:
            print(f"  {row['item_id']} ({row['domain']}): var={row['variance']:.3f}, mean={row['mean']:.2f}")

    # Chinese vs Western DIF
    print("\n--- Chinese vs Western DIF ---")
    df_likert["region"] = df_likert["model_family"].map(
        lambda x: "Chinese" if x == "Chinese" else "Western"
    )

    domain_dif = []
    for (scale, domain), group in df_likert.groupby(["scale", "domain"]):
        chinese = group[group["region"] == "Chinese"]["parsed_value"]
        western = group[group["region"] == "Western"]["parsed_value"]
        if len(chinese) < 5 or len(western) < 5:
            continue
        pooled_std = np.sqrt((chinese.var() + western.var()) / 2)
        d = (chinese.mean() - western.mean()) / pooled_std if pooled_std > 0 else 0
        t_stat, p_val = stats.ttest_ind(chinese, western)
        domain_dif.append({
            "scale": scale, "domain": domain,
            "chinese_mean": chinese.mean(), "western_mean": western.mean(),
            "cohens_d": d, "t_stat": t_stat, "p_value": p_val,
        })

    dif_df = pd.DataFrame(domain_dif).sort_values("cohens_d", key=abs, ascending=False)
    print(dif_df.to_string(index=False))
    sig = dif_df[dif_df["p_value"] < 0.05]
    print(f"\nSignificant (p < 0.05): {len(sig)}/{len(dif_df)}")

    # Convergent validity
    print("\n--- Convergent Validity ---")
    convergence_pairs = [
        ("IPIP-NEO-120", "Neuroticism", "ZKPQ-50-CC", "Neuroticism-Anxiety", "positive"),
        ("IPIP-NEO-120", "Extraversion", "ZKPQ-50-CC", "Sociability", "positive"),
        ("IPIP-NEO-120", "Extraversion", "EPQR-A", "Extraversion", "positive"),
        ("IPIP-NEO-120", "Neuroticism", "EPQR-A", "Neuroticism", "positive"),
        ("IPIP-NEO-120", "Agreeableness", "SD3", "Machiavellianism", "negative"),
        ("IPIP-NEO-120", "Agreeableness", "SD3", "Psychopathy", "negative"),
        ("IPIP-NEO-120", "Conscientiousness", "SD3", "Psychopathy", "negative"),
        ("EPQR-A", "Psychoticism", "SD3", "Psychopathy", "positive"),
    ]

    model_scores = {}
    for model_name, model_data in all_results.items():
        scores = {}
        domain_scores = model_data["results_by_persona"]["Default"]["domain_scores"]
        for key, val in domain_scores.items():
            scores[key] = val["mean_score"]
        model_scores[model_name] = scores

    conv_results = []
    for s1, d1, s2, d2, expected in convergence_pairs:
        k1, k2 = f"{s1}::{d1}", f"{s2}::{d2}"
        v1, v2 = [], []
        for model, scores in model_scores.items():
            if k1 in scores and k2 in scores:
                v1.append(scores[k1])
                v2.append(scores[k2])
        if len(v1) < 3:
            continue
        r_p, p_p = stats.pearsonr(v1, v2)
        r_s, p_s = stats.spearmanr(v1, v2)
        match = (r_p > 0 and expected == "positive") or (r_p < 0 and expected == "negative")
        conv_results.append({
            "pair": f"{d1} <-> {d2}", "expected": expected,
            "r_pearson": r_p, "p_pearson": p_p,
            "r_spearman": r_s, "p_spearman": p_s,
            "sign_match": match, "n_models": len(v1),
        })

    conv_df = pd.DataFrame(conv_results)
    print(conv_df.to_string(index=False))
    print(f"\nSign matches: {conv_df['sign_match'].sum()}/{len(conv_df)}")

    # Save
    item_var.to_csv(OUTPUT_DIR / "item_variance_analysis.csv", index=False)
    dif_df.to_csv(OUTPUT_DIR / "dif_model_family.csv", index=False)
    conv_df.to_csv(OUTPUT_DIR / "convergent_validity_enhanced.csv", index=False)
    if safety_items:
        pd.DataFrame(safety_items).to_csv(OUTPUT_DIR / "safety_contaminated_items.csv", index=False)

    return {
        "low_var_count": len(low_var),
        "safety_items_count": len(safety_items),
        "dif_df": dif_df,
        "conv_df": conv_df,
    }


# ─────────────────────────────────────────────────────────────
# Analysis 5: Response Style Quantification
# ─────────────────────────────────────────────────────────────

def run_response_style_analysis(all_results):
    print("\n" + "=" * 70)
    print("ANALYSIS 5: RESPONSE STYLE QUANTIFICATION")
    print("=" * 70)

    rows = []
    for model_name in sorted(all_results.keys()):
        for persona in all_results[model_name]["results_by_persona"]:
            items_df = extract_item_responses(all_results[model_name], persona)
            likert = items_df[items_df["response_format"] == "likert_5"]
            binary = items_df[items_df["response_format"].isin(["true_false", "yes_no"])]

            fwd_l = likert[likert["keyed"] == "+"]
            rev_l = likert[likert["keyed"] == "-"]
            acq = (fwd_l["parsed_value"].mean() - rev_l["parsed_value"].mean()) if len(fwd_l) > 0 and len(rev_l) > 0 else np.nan

            rows.append({
                "model": model_name,
                "persona": persona,
                "condition": "Default" if persona == "Default" else "MBTI",
                "acquiescence": acq,
                "extreme_response": ((likert["parsed_value"] == 1) | (likert["parsed_value"] == 5)).mean() if len(likert) > 0 else np.nan,
                "midpoint_response": (likert["parsed_value"] == 3).mean() if len(likert) > 0 else np.nan,
                "item_variance": likert["parsed_value"].var() if len(likert) > 0 else np.nan,
                "binary_agree_rate": binary["parsed_value"].mean() if len(binary) > 0 else np.nan,
            })

    rs_df = pd.DataFrame(rows)
    default_rs = rs_df[rs_df["condition"] == "Default"].copy()

    print("\n--- Response Styles (Default) ---")
    print(default_rs[["model", "acquiescence", "extreme_response", "midpoint_response", "item_variance"]].to_string(index=False))

    print("\n--- Response Style Intercorrelations ---")
    rs_cols = ["acquiescence", "extreme_response", "midpoint_response", "item_variance", "binary_agree_rate"]
    valid = default_rs[rs_cols].dropna()
    if len(valid) > 3:
        print(valid.corr(method="spearman").round(3).to_string())

    print("\n--- Default -> MBTI Shift ---")
    for col in ["acquiescence", "extreme_response", "midpoint_response"]:
        dv = rs_df[rs_df["condition"] == "Default"].groupby("model")[col].mean()
        mv = rs_df[rs_df["condition"] == "MBTI"].groupby("model")[col].mean()
        if len(dv) > 2 and len(mv) > 2:
            common = dv.index.intersection(mv.index)
            t, p = stats.ttest_rel(mv[common].values, dv[common].values)
            print(f"  {col}: delta = {(mv[common] - dv[common]).mean():.3f}, t = {t:.2f}, p = {p:.4f}")

    rs_df.to_csv(OUTPUT_DIR / "response_styles.csv", index=False)
    return rs_df


# ─────────────────────────────────────────────────────────────
# Analysis 6: Measurement Invariance
# ─────────────────────────────────────────────────────────────

def run_measurement_invariance(all_results):
    print("\n" + "=" * 70)
    print("ANALYSIS 6: MBTI PERSONA MEASUREMENT INVARIANCE")
    print("=" * 70)

    invariance_rows = []
    for model_name in sorted(all_results.keys()):
        default_items = extract_item_responses(all_results[model_name], "Default")
        default_ipip = default_items[default_items["scale"] == "IPIP-NEO-120"].sort_values("item_id")
        default_vec = default_ipip["parsed_value"].values.astype(float)

        for persona in all_results[model_name]["results_by_persona"]:
            if persona == "Default":
                continue
            persona_items = extract_item_responses(all_results[model_name], persona)
            persona_ipip = persona_items[persona_items["scale"] == "IPIP-NEO-120"].sort_values("item_id")
            persona_vec = persona_ipip["parsed_value"].values.astype(float)

            if len(default_vec) != len(persona_vec):
                continue

            r, p = stats.pearsonr(default_vec, persona_vec)
            cos_sim = np.dot(default_vec, persona_vec) / (np.linalg.norm(default_vec) * np.linalg.norm(persona_vec) + 1e-10)
            mad = np.mean(np.abs(default_vec - persona_vec))
            var_ratio = persona_vec.var() / (default_vec.var() + 1e-10)

            invariance_rows.append({
                "model": model_name, "persona": persona,
                "pearson_r": r, "cosine_sim": cos_sim,
                "mean_abs_diff": mad, "variance_ratio": var_ratio,
            })

    inv_df = pd.DataFrame(invariance_rows)

    print("\n--- Profile Similarity to Default (Pearson r) ---")
    model_inv = inv_df.groupby("model").agg(
        mean_r=("pearson_r", "mean"), std_r=("pearson_r", "std"),
        min_r=("pearson_r", "min"), mean_mad=("mean_abs_diff", "mean"),
    ).reset_index().sort_values("mean_r", ascending=False)
    print(model_inv.to_string(index=False))

    print(f"\nOverall mean r(Default, MBTI): {inv_df['pearson_r'].mean():.3f}")
    print(f"r > 0.5: {(inv_df['pearson_r'] > 0.5).sum()}/{len(inv_df)}")
    print(f"r > 0.8: {(inv_df['pearson_r'] > 0.8).sum()}/{len(inv_df)}")

    print("\n--- Persona-Level Invariance ---")
    persona_inv = inv_df.groupby("persona")["pearson_r"].mean().sort_values()
    print("Most divergent:")
    for p, r in persona_inv.head(5).items():
        print(f"  {p}: r = {r:.3f}")
    print("Most similar:")
    for p, r in persona_inv.tail(5).items():
        print(f"  {p}: r = {r:.3f}")

    mean_r = inv_df["pearson_r"].mean()
    if mean_r > 0.8:
        print("\n-> High configural invariance: persona shifts baseline but preserves structure")
    elif mean_r > 0.5:
        print("\n-> Moderate invariance: persona partially restructures response patterns")
    else:
        print("\n-> Low invariance: persona fundamentally restructures responses")

    inv_df.to_csv(OUTPUT_DIR / "persona_invariance.csv", index=False)
    model_inv.to_csv(OUTPUT_DIR / "persona_invariance_by_model.csv", index=False)

    return inv_df


# ─────────────────────────────────────────────────────────────
# Integrated Report
# ─────────────────────────────────────────────────────────────

def generate_integrated_report(all_results, efa_results, pir_results, var_results,
                                dif_results, rs_df, inv_df):
    lines = []
    lines.append("=" * 80)
    lines.append("DEEP PSYCHOMETRIC VALIDATION REPORT")
    lines.append("LLM Personality Measurement: Methodological Validation & Internal Contradictions")
    lines.append("=" * 80)
    lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Models: {len(all_results)}")
    lines.append(f"Personas: 17 (Default + 16 MBTI)")
    lines.append(f"Items: 221 (IPIP-NEO-120, SD3, ZKPQ-50-CC, EPQR-A)")
    lines.append(f"Total data points: {len(all_results) * 17 * 221:,}")
    lines.append("")
    lines.append("THEORETICAL FRAMEWORK")
    lines.append("-" * 40)
    lines.append("1. Measurement Invariance: Can human psychometric tools measure LLM 'personality'?")
    lines.append("2. Response Style Theory: LLM variance = trait + response style + error")
    lines.append("3. Social Desirability: Alignment creates systematic response distortion")
    lines.append("")

    lines.append("=" * 80)
    lines.append("FINDING 1: FACTOR STRUCTURE COLLAPSE")
    lines.append("=" * 80)
    lines.append(f"Zero-variance items (models all agree): {efa_results['n_zero_var_items']}/120 IPIP items")
    lines.append(f"Domain-level Kaiser criterion: {efa_results['n_factors_kaiser']} factors")
    lines.append(f"Domain-level Parallel analysis: {efa_results['n_factors_pa']} factors")
    lines.append(f"First/second eigenvalue ratio: {efa_results['eigenvalue_ratio']:.2f}")
    mf = efa_results.get("model_factor_df")
    if mf is not None and len(mf) > 0:
        lines.append(f"Mean Tucker's phi (per-model): {mf['mean_congruence'].mean():.3f}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("FINDING 2: REVERSE-ITEM INCONSISTENCY")
    lines.append("=" * 80)
    pir_df = pir_results["pir_df"]
    lines.append(f"Mean PIR: {pir_df['pir'].mean():.3f}")
    lines.append(f"PIR variance — model: {pir_results['var_model']:.1f}%, domain: {pir_results['var_domain']:.1f}%")
    pir_sdr = pir_results["pir_sdr"]
    if len(pir_sdr) > 3:
        r, p = stats.spearmanr(pir_sdr["mean_pir"], pir_sdr["sdr_composite"])
        lines.append(f"PIR x SDR: r = {r:.3f}, p = {p:.4f}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("FINDING 3: VARIANCE DECOMPOSITION")
    lines.append("=" * 80)
    for label, vc in var_results.items():
        total = sum(vc.values())
        lines.append(f"\n{label}:")
        for comp, var in vc.items():
            pct = var / total * 100 if total > 0 else 0
            lines.append(f"  {comp}: {pct:.1f}%")
    lines.append("")

    lines.append("=" * 80)
    lines.append("FINDING 4: DIF + CONVERGENT VALIDITY")
    lines.append("=" * 80)
    lines.append(f"Low-variance items: {dif_results['low_var_count']}")
    lines.append(f"Safety-contaminated: {dif_results['safety_items_count']}")
    sig = dif_results['dif_df'][dif_results['dif_df']['p_value'] < 0.05]
    lines.append(f"Significant DIF: {len(sig)}/{len(dif_results['dif_df'])}")
    conv = dif_results['conv_df']
    lines.append(f"Convergent sign matches: {conv['sign_match'].sum()}/{len(conv)}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("FINDING 5: PERSONA MEASUREMENT INVARIANCE")
    lines.append("=" * 80)
    lines.append(f"Mean r(Default, MBTI): {inv_df['pearson_r'].mean():.3f}")
    lines.append(f"r > 0.8: {(inv_df['pearson_r'] > 0.8).sum()}/{len(inv_df)}")
    lines.append("")

    report = "\n".join(lines)
    with open(OUTPUT_DIR / "psychometric_validation_report.txt", "w") as f:
        f.write(report)
    print(report)
    return report


def main():
    print("Loading experiment results...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} models\n")

    print("=" * 80)
    print("DEEP PSYCHOMETRIC VALIDATION PIPELINE")
    print("=" * 80)

    efa_results = run_efa_analysis(all_results)
    pir_results = run_pir_sdr_analysis(all_results)
    var_results = run_variance_decomposition(all_results)
    dif_results = run_dif_analysis(all_results)
    rs_df = run_response_style_analysis(all_results)
    inv_df = run_measurement_invariance(all_results)

    generate_integrated_report(
        all_results, efa_results, pir_results, var_results,
        dif_results, rs_df, inv_df
    )

    print(f"\n\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
