"""
LLM Personality Consistency Analysis
=====================================
Analyzes whether LLMs produce internally consistent personality profiles
when responding to validated psychometric instruments.

Key analyses:
1. Within-domain forward vs reverse item consistency
2. Forward-reverse correlation across models
3. Cross-scale trait convergence (overlapping constructs)
4. Lie scale / social desirability detection
5. MBTI persona consistency
6. Acquiescence bias analysis
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_results():
    results = {}
    for fpath in sorted(RESULTS_DIR.glob("exp_mbti_*.json")):
        model_name = fpath.stem.replace("exp_mbti_", "")
        with open(fpath) as f:
            results[model_name] = json.load(f)
    return results


def extract_item_responses(model_data, persona="Default"):
    resp_data = model_data["results_by_persona"][persona]["responses"]
    items = []
    for r in resp_data:
        items.append({
            "item_id": r["item_id"],
            "scale": r["scale"],
            "domain": r["domain"],
            "facet": r["facet"],
            "item_text": r["item_text"],
            "keyed": r["keyed"],
            "parsed_value": r["parsed_value"],
            "scored_value": r["scored_value"],
            "response_format": r["response_format"],
            "parse_failed": r.get("parse_failed", False),
        })
    return pd.DataFrame(items)


def compute_forward_reverse_correlation_across_models(all_results):
    """For each domain, correlate forward and reverse subscale means across models.

    If consistent: models scoring high on forward also score high on reverse (after scoring).
    Correlation should be POSITIVE.

    If inconsistent (alignment-driven agreement): forward and reverse diverge.
    """
    rows = []
    for model_name, model_data in all_results.items():
        items_df = extract_item_responses(model_data, persona="Default")
        for (scale, domain), group in items_df.groupby(["scale", "domain"]):
            fwd = group[group["keyed"] == "+"]
            rev = group[group["keyed"] == "-"]
            if len(fwd) == 0 or len(rev) == 0:
                continue
            rows.append({
                "model": model_name,
                "scale": scale,
                "domain": domain,
                "fwd_scored_mean": fwd["scored_value"].mean(),
                "rev_scored_mean": rev["scored_value"].mean(),
                "domain_scored_mean": group["scored_value"].mean(),
                "fwd_raw_mean": fwd["parsed_value"].mean(),
                "rev_raw_mean": rev["parsed_value"].mean(),
                "n_forward": len(fwd),
                "n_reverse": len(rev),
            })

    df = pd.DataFrame(rows)
    corr_results = []
    for (scale, domain), group in df.groupby(["scale", "domain"]):
        if len(group) < 3:
            continue
        r = np.corrcoef(group["fwd_scored_mean"], group["rev_scored_mean"])[0, 1]
        corr_results.append({
            "scale": scale,
            "domain": domain,
            "n_models": len(group),
            "fwd_rev_correlation": r,
            "mean_fwd_scored": group["fwd_scored_mean"].mean(),
            "mean_rev_scored": group["rev_scored_mean"].mean(),
            "mean_fwd_raw": group["fwd_raw_mean"].mean(),
            "mean_rev_raw": group["rev_raw_mean"].mean(),
            "mean_gap": abs(group["fwd_scored_mean"] - group["rev_scored_mean"]).mean(),
        })
    return pd.DataFrame(corr_results), df


def compute_item_level_conflict(all_results):
    """Find specific domains where LLMs show contradictory responses."""
    conflict_rows = []
    for model_name, model_data in all_results.items():
        items_df = extract_item_responses(model_data, persona="Default")

        for (scale, domain), group in items_df.groupby(["scale", "domain"]):
            fwd = group[group["keyed"] == "+"]
            rev = group[group["keyed"] == "-"]
            if len(fwd) == 0 or len(rev) == 0:
                continue

            fwd_raw_mean = fwd["parsed_value"].mean()
            rev_raw_mean = rev["parsed_value"].mean()
            fwd_scored_mean = fwd["scored_value"].mean()
            rev_scored_mean = rev["scored_value"].mean()

            resp_format = group["response_format"].values[0]
            if "likert" in resp_format:
                scale_range = 4.0
            else:
                scale_range = 1.0

            agreement = 1.0 - abs(fwd_scored_mean - rev_scored_mean) / scale_range

            if "likert" in resp_format:
                acquiescence = (fwd_raw_mean + rev_raw_mean) / 2 - 3.0
            else:
                acquiescence = (fwd_raw_mean + rev_raw_mean) / 2 - 0.5

            conflict_rows.append({
                "model": model_name,
                "scale": scale,
                "domain": domain,
                "n_fwd": len(fwd),
                "n_rev": len(rev),
                "fwd_raw_mean": fwd_raw_mean,
                "rev_raw_mean": rev_raw_mean,
                "fwd_scored_mean": fwd_scored_mean,
                "rev_scored_mean": rev_scored_mean,
                "agreement": agreement,
                "acquiescence": acquiescence,
                "response_format": resp_format,
            })

    return pd.DataFrame(conflict_rows)


def compute_cross_scale_convergence(all_results):
    """Check if overlapping constructs across scales converge."""
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

    results = []
    for scale1, dom1, scale2, dom2, expected_dir in convergence_pairs:
        key1 = f"{scale1}::{dom1}"
        key2 = f"{scale2}::{dom2}"
        vals1, vals2 = [], []
        for model, scores in model_scores.items():
            if key1 in scores and key2 in scores:
                vals1.append(scores[key1])
                vals2.append(scores[key2])
        if len(vals1) < 3:
            continue
        r = np.corrcoef(vals1, vals2)[0, 1]
        sign_match = (r > 0 and expected_dir == "positive") or (r < 0 and expected_dir == "negative")
        results.append({
            "scale1": scale1,
            "domain1": dom1,
            "scale2": scale2,
            "domain2": dom2,
            "expected": expected_dir,
            "r": r,
            "n_models": len(vals1),
            "sign_match": sign_match,
        })
    return pd.DataFrame(results)


def compute_lie_scale_analysis(all_results):
    """Analyze EPQR-A Lie scale and social desirability indicators."""
    results = []
    for model_name, model_data in all_results.items():
        items_df = extract_item_responses(model_data, persona="Default")
        lie_items = items_df[(items_df["scale"] == "EPQR-A") & (items_df["domain"] == "Lie")]
        if len(lie_items) == 0:
            continue

        lie_score = lie_items["scored_value"].mean()

        ipip = items_df[items_df["scale"] == "IPIP-NEO-120"]
        fwd_mean = ipip[ipip["keyed"] == "+"]["parsed_value"].mean() if len(ipip[ipip["keyed"] == "+"]) > 0 else np.nan
        rev_mean = ipip[ipip["keyed"] == "-"]["parsed_value"].mean() if len(ipip[ipip["keyed"] == "-"]) > 0 else np.nan
        acquiescence = fwd_mean - rev_mean if not np.isnan(fwd_mean) and not np.isnan(rev_mean) else np.nan

        likert_items = items_df[items_df["response_format"] == "likert_5"]
        midpoint_rate = (likert_items["parsed_value"] == 3).mean() if len(likert_items) > 0 else np.nan
        extreme_rate = ((likert_items["parsed_value"] == 1) | (likert_items["parsed_value"] == 5)).mean() if len(likert_items) > 0 else np.nan

        sd3 = items_df[items_df["scale"] == "SD3"]
        mach = sd3[sd3["domain"] == "Machiavellianism"]["scored_value"].mean() if len(sd3[sd3["domain"] == "Machiavellianism"]) > 0 else np.nan
        narc = sd3[sd3["domain"] == "Narcissism"]["scored_value"].mean() if len(sd3[sd3["domain"] == "Narcissism"]) > 0 else np.nan
        psych = sd3[sd3["domain"] == "Psychopathy"]["scored_value"].mean() if len(sd3[sd3["domain"] == "Psychopathy"]) > 0 else np.nan

        results.append({
            "model": model_name,
            "lie_score": lie_score,
            "acquiescence_bias": acquiescence,
            "midpoint_rate": midpoint_rate,
            "extreme_rate": extreme_rate,
            "sd3_machiavellianism": mach,
            "sd3_narcissism": narc,
            "sd3_psychopathy": psych,
        })
    return pd.DataFrame(results)


def compute_mbti_persona_consistency(all_results):
    """Check if MBTI persona-driven responses show theoretically expected patterns."""
    rows = []
    for model_name, model_data in all_results.items():
        for persona, pdata in model_data["results_by_persona"].items():
            if persona == "Default":
                continue
            ei, sn, tf, jp = persona[0], persona[1], persona[2], persona[3]
            ds = pdata["domain_scores"]
            row = {
                "model": model_name,
                "persona": persona,
                "EI": ei, "SN": sn, "TF": tf, "JP": jp,
            }
            for key, val in ds.items():
                row[key] = val["mean_score"]
            rows.append(row)

    df = pd.DataFrame(rows)

    checks = {
        "E > I on Extraversion": (
            lambda md: md[md["EI"] == "E"]["IPIP-NEO-120::Extraversion"].mean()
                       > md[md["EI"] == "I"]["IPIP-NEO-120::Extraversion"].mean(),
            lambda md: (
                md[md["EI"] == "E"]["IPIP-NEO-120::Extraversion"].mean(),
                md[md["EI"] == "I"]["IPIP-NEO-120::Extraversion"].mean(),
            ),
        ),
        "J > P on Conscientiousness": (
            lambda md: md[md["JP"] == "J"]["IPIP-NEO-120::Conscientiousness"].mean()
                       > md[md["JP"] == "P"]["IPIP-NEO-120::Conscientiousness"].mean(),
            lambda md: (
                md[md["JP"] == "J"]["IPIP-NEO-120::Conscientiousness"].mean(),
                md[md["JP"] == "P"]["IPIP-NEO-120::Conscientiousness"].mean(),
            ),
        ),
        "F > T on Agreeableness": (
            lambda md: md[md["TF"] == "F"]["IPIP-NEO-120::Agreeableness"].mean()
                       > md[md["TF"] == "T"]["IPIP-NEO-120::Agreeableness"].mean(),
            lambda md: (
                md[md["TF"] == "F"]["IPIP-NEO-120::Agreeableness"].mean(),
                md[md["TF"] == "T"]["IPIP-NEO-120::Agreeableness"].mean(),
            ),
        ),
        "N > S on Openness": (
            lambda md: md[md["SN"] == "N"]["IPIP-NEO-120::Openness"].mean()
                       > md[md["SN"] == "S"]["IPIP-NEO-120::Openness"].mean(),
            lambda md: (
                md[md["SN"] == "N"]["IPIP-NEO-120::Openness"].mean(),
                md[md["SN"] == "S"]["IPIP-NEO-120::Openness"].mean(),
            ),
        ),
    }

    patterns = []
    for model in df["model"].unique():
        md = df[df["model"] == model]
        for check_name, (check_fn, val_fn) in checks.items():
            try:
                correct = check_fn(md)
                vals = val_fn(md)
                row = {"model": model, "check": check_name, "correct": correct}
                if "E > I" in check_name:
                    row["group_A_mean"] = vals[0]; row["group_B_mean"] = vals[1]
                elif "J > P" in check_name:
                    row["group_A_mean"] = vals[0]; row["group_B_mean"] = vals[1]
                elif "F > T" in check_name:
                    row["group_A_mean"] = vals[0]; row["group_B_mean"] = vals[1]
                elif "N > S" in check_name:
                    row["group_A_mean"] = vals[0]; row["group_B_mean"] = vals[1]
                patterns.append(row)
            except Exception:
                pass

    return df, pd.DataFrame(patterns)


def compute_inter_dimension_correlations(all_results, persona="Default"):
    """Compute correlations between all domain pairs across models."""
    model_vectors = {}
    for model_name, model_data in all_results.items():
        ds = model_data["results_by_persona"][persona]["domain_scores"]
        vec = {key: val["mean_score"] for key, val in ds.items()}
        model_vectors[model_name] = vec

    df = pd.DataFrame(model_vectors).T
    corr_matrix = df.corr()
    return corr_matrix, df


def generate_report(all_results):
    """Generate the full analysis report."""
    lines = []
    lines.append("=" * 80)
    lines.append("LLM PERSONALITY CONSISTENCY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Models analyzed: {len(all_results)}")
    lines.append(f"Models: {', '.join(sorted(all_results.keys()))}")
    lines.append(f"Personas per model: 17 (Default + 16 MBTI)")
    lines.append(f"Items per condition: 221 (4 scales, 17 domains)")
    lines.append("")

    # --- 1. Forward-Reverse Correlation ---
    lines.append("-" * 80)
    lines.append("1. FORWARD vs REVERSE ITEM CONSISTENCY (correlation across models)")
    lines.append("-" * 80)
    corr_df, detail_df = compute_forward_reverse_correlation_across_models(all_results)
    lines.append(corr_df.to_string(index=False))
    lines.append("")
    n_positive = (corr_df["fwd_rev_correlation"] > 0).sum()
    n_total = len(corr_df)
    lines.append(f"Positive correlation (consistent): {n_positive}/{n_total} ({100*n_positive/n_total:.1f}%)")
    lines.append(f"Mean correlation: {corr_df['fwd_rev_correlation'].mean():.3f}")
    lines.append("")

    # --- 2. Item-level conflict ---
    lines.append("-" * 80)
    lines.append("2. ITEM-LEVEL CONFLICT ANALYSIS (per model x domain)")
    lines.append("-" * 80)
    conflict_df = compute_item_level_conflict(all_results)
    summary = conflict_df.groupby(["scale", "domain"]).agg(
        mean_agreement=("agreement", "mean"),
        std_agreement=("agreement", "std"),
        mean_acquiescence=("acquiescence", "mean"),
        mean_fwd_scored=("fwd_scored_mean", "mean"),
        mean_rev_scored=("rev_scored_mean", "mean"),
    ).reset_index()
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("agreement: 1.0 = perfect forward/reverse consistency")
    lines.append("acquiescence: positive = agreeing with all items regardless of direction")
    lines.append("")

    # --- 3. Lie scale ---
    lines.append("-" * 80)
    lines.append("3. LIE SCALE & SOCIAL DESIRABILITY")
    lines.append("-" * 80)
    lie_df = compute_lie_scale_analysis(all_results)
    lines.append(lie_df.sort_values("lie_score", ascending=False).to_string(index=False))
    lines.append("")
    lines.append(f"Mean Lie score: {lie_df['lie_score'].mean():.3f} (0-1 scale)")
    lines.append(f"Mean Acquiescence bias: {lie_df['acquiescence_bias'].mean():.3f}")
    lines.append(f"Mean Midpoint rate: {lie_df['midpoint_rate'].mean():.3f}")
    lines.append("")

    # --- 4. Cross-scale convergence ---
    lines.append("-" * 80)
    lines.append("4. CROSS-SCALE TRAIT CONVERGENCE")
    lines.append("-" * 80)
    conv_df = compute_cross_scale_convergence(all_results)
    lines.append(conv_df.to_string(index=False))
    lines.append("")
    n_match = conv_df["sign_match"].sum()
    n_total_c = len(conv_df)
    lines.append(f"Sign match: {n_match}/{n_total_c} ({100*n_match/n_total_c:.1f}%)")
    lines.append("")

    # --- 5. MBTI persona consistency ---
    lines.append("-" * 80)
    lines.append("5. MBTI PERSONA CONSISTENCY")
    lines.append("-" * 80)
    mbti_df, pattern_df = compute_mbti_persona_consistency(all_results)
    if len(pattern_df) > 0:
        pattern_summary = pattern_df.groupby("check").agg(
            n_models=("correct", "count"),
            n_correct=("correct", "sum"),
        ).reset_index()
        pattern_summary["rate"] = pattern_summary["n_correct"] / pattern_summary["n_models"]
        lines.append(pattern_summary.to_string(index=False))
        lines.append("")
        per_model = pattern_df.groupby("model").agg(
            total=("correct", "count"),
            correct=("correct", "sum"),
        ).reset_index()
        per_model["rate"] = per_model["correct"] / per_model["total"]
        lines.append("Per-model consistency:")
        lines.append(per_model.sort_values("rate", ascending=False).to_string(index=False))
    lines.append("")

    # --- 6. Key findings ---
    lines.append("=" * 80)
    lines.append("KEY FINDINGS SUMMARY")
    lines.append("=" * 80)

    mean_agreement = conflict_df["agreement"].mean()
    mean_acq = conflict_df["acquiescence"].mean()
    mean_lie = lie_df["lie_score"].mean()
    convergence_rate = n_match / n_total_c if n_total_c > 0 else 0

    lines.append(f"1. Forward-Reverse Agreement (mean): {mean_agreement:.3f}")
    lines.append(f"2. Acquiescence Bias (mean): {mean_acq:.3f}")
    lines.append(f"3. Lie Scale Score (mean): {mean_lie:.3f}")
    lines.append(f"4. Cross-Scale Convergence: {convergence_rate:.1%}")
    if len(pattern_df) > 0:
        lines.append(f"5. MBTI Persona Consistency: {pattern_df['correct'].mean():.1%}")

    lines.append("")
    model_agreement = conflict_df.groupby("model")["agreement"].mean().sort_values()
    lines.append("Least consistent models:")
    for m, a in model_agreement.head(5).items():
        lines.append(f"  {m}: {a:.3f}")
    lines.append("Most consistent models:")
    for m, a in model_agreement.tail(5).items():
        lines.append(f"  {m}: {a:.3f}")

    return "\n".join(lines)


def save_outputs(all_results):
    report = generate_report(all_results)
    with open(OUTPUT_DIR / "consistency_report.txt", "w") as f:
        f.write(report)
    print(report)

    conflict_df = compute_item_level_conflict(all_results)
    conflict_df.to_csv(OUTPUT_DIR / "item_level_conflict.csv", index=False)

    corr_df, detail_df = compute_forward_reverse_correlation_across_models(all_results)
    corr_df.to_csv(OUTPUT_DIR / "forward_reverse_correlation.csv", index=False)
    detail_df.to_csv(OUTPUT_DIR / "forward_reverse_detail.csv", index=False)

    lie_df = compute_lie_scale_analysis(all_results)
    lie_df.to_csv(OUTPUT_DIR / "lie_scale_analysis.csv", index=False)

    conv_df = compute_cross_scale_convergence(all_results)
    conv_df.to_csv(OUTPUT_DIR / "cross_scale_convergence.csv", index=False)

    corr_matrix, score_df = compute_inter_dimension_correlations(all_results)
    corr_matrix.to_csv(OUTPUT_DIR / "inter_dimension_correlations.csv")
    score_df.to_csv(OUTPUT_DIR / "model_domain_scores.csv")

    mbti_df, pattern_df = compute_mbti_persona_consistency(all_results)
    mbti_df.to_csv(OUTPUT_DIR / "mbti_persona_scores.csv", index=False)
    pattern_df.to_csv(OUTPUT_DIR / "mbti_persona_consistency.csv", index=False)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    print("Loading experiment results...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} models")
    print()
    save_outputs(all_results)
