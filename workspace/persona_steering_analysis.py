#!/usr/bin/env python3
"""Persona steering analyses for the LLM psychometric battery.

This script is intentionally offline: it only reads completed experiment JSON
files and the static battery specification. It does not call any model API.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RESULTS_DIR = Path("results")
ITEMS_PATH = Path("data/items_battery.json")
OUT_DIR = Path("analysis_output")
FIG_DIR = Path("figures")
FINAL_DIR = Path("workspace/final")

OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

MBTI_ORDER = [
    "ISTJ",
    "ISFJ",
    "INFJ",
    "INTJ",
    "ISTP",
    "ISFP",
    "INFP",
    "INTP",
    "ESTP",
    "ESFP",
    "ENFP",
    "ENTP",
    "ESTJ",
    "ESFJ",
    "ENFJ",
    "ENTJ",
]

AXES = ["EI", "SN", "TF", "JP"]
AXIS_LABELS = {
    "EI": "E minus I",
    "SN": "N minus S",
    "TF": "F minus T",
    "JP": "J minus P",
}

SCALE_RANGES = {
    "likert_5": 4.0,
    "true_false": 1.0,
    "yes_no": 1.0,
}

PALETTE = {
    "blue": "#3B6EA8",
    "orange": "#D47A2A",
    "green": "#4C8A67",
    "red": "#B85C5A",
    "purple": "#8E6FAE",
    "cyan": "#5AA6B8",
    "gray": "#7D8790",
    "lgray": "#E8ECEF",
    "black": "#1D2329",
    "paper": "#FBFAF7",
}

COL_W = 3.35
FULL_W = 6.95

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "figure.titlesize": 9.5,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["black"],
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A9B0B7")
    ax.spines["bottom"].set_color("#A9B0B7")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E6E8EB", linewidth=0.55)
        ax.set_axisbelow(True)


def clean_label(label: str) -> str:
    return (
        label.replace("IPIP-NEO-120::", "IPIP ")
        .replace("ZKPQ-50-CC::", "ZKPQ ")
        .replace("EPQR-A::", "EPQR ")
        .replace("SD3::", "SD3 ")
        .replace("Impulsive_Sensation_Seeking", "ImpSS")
        .replace("Aggression-Hostility", "AggHost")
        .replace("Neuroticism-Anxiety", "N-Anx")
        .replace("Machiavellianism", "Mach")
        .replace("Conscientiousness", "Consc")
        .replace("Agreeableness", "Agree")
        .replace("Extraversion", "Extra")
        .replace("Neuroticism", "Neuro")
        .replace("Psychoticism", "Psychot")
        .replace("Psychopathy", "Psychop")
        .replace("Narcissism", "Narc")
    )


def short_model(name: str) -> str:
    replacements = {
        "Claude-": "Claude ",
        "DeepSeek-": "DS ",
        "Gemini-": "Gem ",
        "GPT-": "GPT ",
        "Qwen3.5-": "Qwen ",
        "Qwen3-": "Qwen ",
        "-Preview": "",
        "Preview": "",
        "Flash-Lite": "FL",
        "-A22B": "",
        "-A10B": "",
        "-A17B": "",
    }
    out = name
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def unit(vec: np.ndarray) -> np.ndarray:
    n = norm(vec)
    if n == 0 or not np.isfinite(n):
        return np.zeros_like(vec)
    return vec / n


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def euclidean_per_dim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / math.sqrt(len(a)))


def mahalanobis(a: np.ndarray, b: np.ndarray, inv_cov: np.ndarray) -> float:
    d = a - b
    val = float(d @ inv_cov @ d.T)
    return math.sqrt(max(val, 0.0) / len(d))


def axis_code(persona: str) -> dict[str, int]:
    return {
        "EI": 1 if persona[0] == "E" else -1,
        "SN": 1 if persona[1] == "N" else -1,
        "TF": 1 if persona[2] == "F" else -1,
        "JP": 1 if persona[3] == "J" else -1,
    }


def load_results() -> dict[str, dict]:
    files = sorted(RESULTS_DIR.glob("exp_mbti_*.json"))
    if not files:
        raise FileNotFoundError(f"No result files found in {RESULTS_DIR}")
    data = {}
    for path in files:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        data[payload["model_name"]] = payload
    return dict(sorted(data.items()))


def load_items() -> pd.DataFrame:
    with ITEMS_PATH.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = []
    for item in payload["items"]:
        rows.append(
            {
                "item_id": item["id"],
                "scale": item["scale"],
                "domain": item["domain"],
                "facet": item.get("facet"),
                "keyed": item["keyed"],
                "response_format": item["response_format"],
                "item_text": item["text"],
            }
        )
    return pd.DataFrame(rows)


def domain_order_from_results(all_results: dict[str, dict]) -> list[str]:
    first_model = next(iter(all_results))
    first_persona = next(iter(all_results[first_model]["results_by_persona"]))
    return list(all_results[first_model]["results_by_persona"][first_persona]["domain_scores"].keys())


def build_domain_frame(all_results: dict[str, dict], domains: list[str]) -> pd.DataFrame:
    rows = []
    for model, payload in all_results.items():
        for persona, pdata in payload["results_by_persona"].items():
            row = {"model": model, "persona": persona}
            for domain in domains:
                row[domain] = float(pdata["domain_scores"][domain]["mean_score"])
            rows.append(row)
    return pd.DataFrame(rows)


def zscore_domain_frame(score_df: pd.DataFrame, domains: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    means = score_df[domains].mean()
    stds = score_df[domains].std(ddof=0).replace(0, 1.0)
    z_df = score_df[["model", "persona"]].copy()
    z_df[domains] = (score_df[domains] - means) / stds
    return z_df, means, stds


def make_profile_map(z_df: pd.DataFrame, domains: list[str]) -> dict[tuple[str, str], np.ndarray]:
    profiles = {}
    for _, row in z_df.iterrows():
        profiles[(row["model"], row["persona"])] = row[domains].to_numpy(dtype=float)
    return profiles


def make_theory_axes(domains: list[str]) -> dict[str, np.ndarray]:
    idx = {domain: i for i, domain in enumerate(domains)}

    def empty() -> np.ndarray:
        return np.zeros(len(domains), dtype=float)

    def add(vec: np.ndarray, domain: str, value: float) -> None:
        if domain in idx:
            vec[idx[domain]] += value

    # Positive directions are E, N, F, and J respectively.
    axes = {axis: empty() for axis in AXES}

    add(axes["EI"], "IPIP-NEO-120::Extraversion", 1.0)
    add(axes["EI"], "EPQR-A::Extraversion", 1.0)
    add(axes["EI"], "ZKPQ-50-CC::Sociability", 1.0)
    add(axes["EI"], "ZKPQ-50-CC::Activity", 0.75)

    add(axes["SN"], "IPIP-NEO-120::Openness", 1.0)

    add(axes["TF"], "IPIP-NEO-120::Agreeableness", 1.0)
    add(axes["TF"], "SD3::Machiavellianism", -0.75)
    add(axes["TF"], "SD3::Psychopathy", -0.75)
    add(axes["TF"], "ZKPQ-50-CC::Aggression-Hostility", -0.75)

    add(axes["JP"], "IPIP-NEO-120::Conscientiousness", 1.0)
    add(axes["JP"], "ZKPQ-50-CC::Impulsive_Sensation_Seeking", -1.0)
    add(axes["JP"], "EPQR-A::Psychoticism", -0.75)

    return {axis: unit(vec) for axis, vec in axes.items()}


def theory_target(persona: str, theory_axes: dict[str, np.ndarray]) -> np.ndarray:
    codes = axis_code(persona)
    target = np.zeros_like(next(iter(theory_axes.values())))
    for axis, sign in codes.items():
        target += sign * theory_axes[axis]
    return target


def empirical_centroid(
    profiles: dict[tuple[str, str], np.ndarray], models: list[str], personas: list[str], held_model: str, persona: str
) -> np.ndarray:
    vecs = [profiles[(m, persona)] for m in models if m != held_model]
    return np.mean(vecs, axis=0)


def empirical_group_mean(
    profiles: dict[tuple[str, str], np.ndarray],
    models: list[str],
    personas: list[str],
    held_model: str,
    selector,
) -> np.ndarray:
    vecs = []
    for model in models:
        if model == held_model:
            continue
        for persona in personas:
            if selector(persona):
                vecs.append(profiles[(model, persona)])
    return np.mean(vecs, axis=0)


def empirical_axis(
    profiles: dict[tuple[str, str], np.ndarray], models: list[str], personas: list[str], held_model: str, axis: str
) -> np.ndarray:
    if axis == "EI":
        pos = empirical_group_mean(profiles, models, personas, held_model, lambda p: p[0] == "E")
        neg = empirical_group_mean(profiles, models, personas, held_model, lambda p: p[0] == "I")
    elif axis == "SN":
        pos = empirical_group_mean(profiles, models, personas, held_model, lambda p: p[1] == "N")
        neg = empirical_group_mean(profiles, models, personas, held_model, lambda p: p[1] == "S")
    elif axis == "TF":
        pos = empirical_group_mean(profiles, models, personas, held_model, lambda p: p[2] == "F")
        neg = empirical_group_mean(profiles, models, personas, held_model, lambda p: p[2] == "T")
    elif axis == "JP":
        pos = empirical_group_mean(profiles, models, personas, held_model, lambda p: p[3] == "J")
        neg = empirical_group_mean(profiles, models, personas, held_model, lambda p: p[3] == "P")
    else:
        raise ValueError(axis)
    return pos - neg


def vector_metrics(delta: np.ndarray, target: np.ndarray) -> dict[str, float]:
    dim = math.sqrt(len(delta))
    delta_norm = norm(delta)
    target_norm = norm(target)
    if delta_norm == 0 or target_norm == 0:
        return {
            "cosine": np.nan,
            "projection": np.nan,
            "projection_per_dim": np.nan,
            "off_target_leakage": np.nan,
            "target_norm": target_norm,
        }
    target_unit = target / target_norm
    projection = float(np.dot(delta, target_unit))
    leakage = math.sqrt(max(delta_norm**2 - projection**2, 0.0))
    return {
        "cosine": float(projection / delta_norm),
        "projection": projection,
        "projection_per_dim": projection / dim,
        "off_target_leakage": leakage / dim,
        "target_norm": target_norm / dim,
    }


def covariance_inverse(profiles: dict[tuple[str, str], np.ndarray], models: list[str], personas: list[str]) -> np.ndarray:
    x = np.array([profiles[(model, persona)] for model in models for persona in personas], dtype=float)
    cov = np.cov(x, rowvar=False)
    shrink = 0.05 * np.trace(cov) / cov.shape[0]
    return np.linalg.pinv(cov + np.eye(cov.shape[0]) * shrink)


def nearest_centroid(
    vec: np.ndarray,
    centroids: dict[str, np.ndarray],
    center: np.ndarray,
    inv_cov: np.ndarray,
    metric: str,
) -> tuple[str, float, float]:
    distances = []
    for persona, centroid in centroids.items():
        if metric == "z_euclidean":
            dist = euclidean_per_dim(vec, centroid)
        elif metric == "cosine":
            dist = cosine_distance(vec - center, centroid - center)
        elif metric == "mahalanobis":
            dist = mahalanobis(vec, centroid, inv_cov)
        else:
            raise ValueError(metric)
        distances.append((persona, dist))
    distances.sort(key=lambda x: x[1])
    margin = distances[1][1] - distances[0][1] if len(distances) > 1 else np.nan
    return distances[0][0], float(distances[0][1]), float(margin)


def run_default_bias(
    profiles: dict[tuple[str, str], np.ndarray],
    models: list[str],
    personas: list[str],
    theory_axes: dict[str, np.ndarray],
    inv_cov: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for model in models:
        default_vec = profiles[(model, "Default")]
        centroids = {p: empirical_centroid(profiles, models, personas, model, p) for p in personas}
        center = np.mean(list(centroids.values()), axis=0)

        nearest = {}
        for metric in ["z_euclidean", "cosine", "mahalanobis"]:
            pred, dist, margin = nearest_centroid(default_vec, centroids, center, inv_cov, metric)
            nearest[f"nearest_{metric}"] = pred
            nearest[f"nearest_{metric}_distance"] = dist
            nearest[f"nearest_{metric}_margin"] = margin

        row = {"model": model, **nearest}
        empirical_code = []
        theory_code = []
        for axis in AXES:
            emp_axis = unit(empirical_axis(profiles, models, personas, model, axis))
            th_axis = unit(theory_axes[axis])
            centered = default_vec - center
            emp_projection = float(np.dot(centered, emp_axis))
            th_projection = float(np.dot(centered, th_axis))
            row[f"empirical_{axis}_projection"] = emp_projection
            row[f"theory_{axis}_projection"] = th_projection
            if axis == "EI":
                empirical_code.append("E" if emp_projection >= 0 else "I")
                theory_code.append("E" if th_projection >= 0 else "I")
            elif axis == "SN":
                empirical_code.append("N" if emp_projection >= 0 else "S")
                theory_code.append("N" if th_projection >= 0 else "S")
            elif axis == "TF":
                empirical_code.append("F" if emp_projection >= 0 else "T")
                theory_code.append("F" if th_projection >= 0 else "T")
            elif axis == "JP":
                empirical_code.append("J" if emp_projection >= 0 else "P")
                theory_code.append("J" if th_projection >= 0 else "P")
        row["default_empirical_axis_type"] = "".join(empirical_code)
        row["default_theory_axis_type"] = "".join(theory_code)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "default_bias_index.csv", index=False)
    return df


def run_target_adherence(
    profiles: dict[tuple[str, str], np.ndarray],
    models: list[str],
    personas: list[str],
    theory_axes: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for model in models:
        default_vec = profiles[(model, "Default")]
        default_centroid = empirical_centroid(profiles, models, ["Default"], model, "Default")
        for persona in personas:
            persona_vec = profiles[(model, persona)]
            delta = persona_vec - default_vec
            empirical_target = empirical_centroid(profiles, models, personas, model, persona) - default_centroid
            th_target = theory_target(persona, theory_axes)
            emp = vector_metrics(delta, empirical_target)
            th = vector_metrics(delta, th_target)
            row = {
                "model": model,
                "persona": persona,
                "psd": norm(delta) / math.sqrt(len(delta)),
                "delta_norm": norm(delta),
                "empirical_cosine": emp["cosine"],
                "empirical_projection": emp["projection"],
                "empirical_projection_per_dim": emp["projection_per_dim"],
                "empirical_off_target_leakage": emp["off_target_leakage"],
                "empirical_target_norm": emp["target_norm"],
                "theory_cosine": th["cosine"],
                "theory_projection": th["projection"],
                "theory_projection_per_dim": th["projection_per_dim"],
                "theory_off_target_leakage": th["off_target_leakage"],
                "theory_target_norm": th["target_norm"],
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "persona_target_adherence.csv", index=False)
    return df


def run_persona_fidelity(
    profiles: dict[tuple[str, str], np.ndarray],
    models: list[str],
    personas: list[str],
    inv_cov: np.ndarray,
    domain_indices: list[int] | None = None,
    subset_name: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    if domain_indices is None:
        domain_indices = list(range(len(next(iter(profiles.values())))))

    def sub(vec: np.ndarray) -> np.ndarray:
        return vec[domain_indices]

    # Metric-specific covariance for ablations.
    x = np.array([sub(profiles[(model, persona)]) for model in models for persona in personas], dtype=float)
    cov = np.cov(x, rowvar=False)
    shrink = 0.05 * np.trace(cov) / cov.shape[0] if cov.ndim == 2 else 0.05
    local_inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * shrink) if cov.ndim == 2 else np.eye(len(domain_indices))

    for model in models:
        centroids = {p: sub(empirical_centroid(profiles, models, personas, model, p)) for p in personas}
        center = np.mean(list(centroids.values()), axis=0)
        for actual in personas:
            vec = sub(profiles[(model, actual)])
            row = {"subset": subset_name, "model": model, "actual_persona": actual}
            for metric in ["z_euclidean", "cosine", "mahalanobis"]:
                pred, dist, margin = nearest_centroid(vec, centroids, center, local_inv, metric)
                row[f"predicted_{metric}"] = pred
                row[f"distance_{metric}"] = dist
                row[f"margin_{metric}"] = margin
                row[f"correct_{metric}"] = pred == actual
            detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    if subset_name == "all":
        detail_df.to_csv(OUT_DIR / "persona_fidelity_detail.csv", index=False)

    sparse = (
        detail_df.groupby(["actual_persona", "predicted_z_euclidean"])
        .size()
        .reset_index(name="count")
        .rename(columns={"predicted_z_euclidean": "predicted_persona"})
    )
    full_index = pd.MultiIndex.from_product(
        [MBTI_ORDER, MBTI_ORDER], names=["actual_persona", "predicted_persona"]
    )
    confusion = (
        sparse.set_index(["actual_persona", "predicted_persona"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    totals = confusion.groupby("actual_persona")["count"].transform("sum")
    confusion["rate"] = np.where(totals > 0, confusion["count"] / totals, 0.0)
    if subset_name == "all":
        confusion.to_csv(OUT_DIR / "persona_fidelity_confusion.csv", index=False)
    return detail_df, confusion


def run_cross_scale_coherence(
    profiles: dict[tuple[str, str], np.ndarray], models: list[str], personas: list[str], domains: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    constructs = {
        "Extraversion": [
            ("IPIP-NEO-120::Extraversion", 1),
            ("EPQR-A::Extraversion", 1),
            ("ZKPQ-50-CC::Sociability", 1),
            ("ZKPQ-50-CC::Activity", 1),
        ],
        "Neuroticism": [
            ("IPIP-NEO-120::Neuroticism", 1),
            ("EPQR-A::Neuroticism", 1),
            ("ZKPQ-50-CC::Neuroticism-Anxiety", 1),
        ],
        "Agreeableness_vs_Antagonism": [
            ("IPIP-NEO-120::Agreeableness", 1),
            ("SD3::Machiavellianism", -1),
            ("SD3::Psychopathy", -1),
            ("ZKPQ-50-CC::Aggression-Hostility", -1),
        ],
        "Conscientiousness_vs_Disinhibition": [
            ("IPIP-NEO-120::Conscientiousness", 1),
            ("ZKPQ-50-CC::Impulsive_Sensation_Seeking", -1),
            ("EPQR-A::Psychoticism", -1),
            ("SD3::Psychopathy", -1),
        ],
    }
    idx = {domain: i for i, domain in enumerate(domains)}
    rows = []
    pair_rows = []

    long_delta = []
    for model in models:
        default_vec = profiles[(model, "Default")]
        for persona in personas:
            delta = profiles[(model, persona)] - default_vec
            for construct, members in constructs.items():
                adjusted = []
                raw = []
                labels = []
                for domain, sign in members:
                    if domain not in idx:
                        continue
                    val = float(delta[idx[domain]])
                    raw.append(val)
                    adjusted.append(float(sign * val))
                    labels.append(domain)
                    long_delta.append(
                        {
                            "model": model,
                            "persona": persona,
                            "construct": construct,
                            "domain": domain,
                            "adjusted_delta": float(sign * val),
                        }
                    )
                adjusted_arr = np.array(adjusted, dtype=float)
                signs = np.sign(adjusted_arr[np.abs(adjusted_arr) > 1e-9])
                total_pairs = 0
                same_pairs = 0
                for i in range(len(signs)):
                    for j in range(i + 1, len(signs)):
                        total_pairs += 1
                        same_pairs += int(signs[i] == signs[j])
                sign_match = same_pairs / total_pairs if total_pairs else np.nan
                denom = float(np.mean(np.abs(adjusted_arr))) if len(adjusted_arr) else np.nan
                coherence = abs(float(np.mean(adjusted_arr))) / denom if denom and np.isfinite(denom) else np.nan
                rows.append(
                    {
                        "model": model,
                        "persona": persona,
                        "construct": construct,
                        "n_domains": len(adjusted_arr),
                        "mean_adjusted_delta": float(np.mean(adjusted_arr)),
                        "mean_abs_adjusted_delta": denom,
                        "sign_match_rate": sign_match,
                        "coherence_score": coherence,
                    }
                )

    delta_df = pd.DataFrame(long_delta)
    for construct, cdf in delta_df.groupby("construct"):
        pivot = cdf.pivot_table(index=["model", "persona"], columns="domain", values="adjusted_delta")
        cols = list(pivot.columns)
        for i, left in enumerate(cols):
            for right in cols[i + 1 :]:
                pair = pivot[[left, right]].dropna()
                if len(pair) < 3:
                    r, p = np.nan, np.nan
                else:
                    r, p = stats.pearsonr(pair[left], pair[right])
                pair_rows.append(
                    {
                        "construct": construct,
                        "domain_1": left,
                        "domain_2": right,
                        "r_delta": r,
                        "p_delta": p,
                        "n": len(pair),
                    }
                )

    df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    summary = (
        df.groupby("construct")
        .agg(
            mean_sign_match=("sign_match_rate", "mean"),
            mean_coherence=("coherence_score", "mean"),
            mean_abs_delta=("mean_abs_adjusted_delta", "mean"),
        )
        .reset_index()
    )
    df.to_csv(OUT_DIR / "cross_scale_persona_coherence.csv", index=False)
    summary.to_csv(OUT_DIR / "cross_scale_persona_coherence_summary.csv", index=False)
    pair_df.to_csv(OUT_DIR / "cross_scale_delta_correlations.csv", index=False)
    return df, pair_df


def run_factorial_effects(
    profiles: dict[tuple[str, str], np.ndarray], models: list[str], personas: list[str], domains: list[str]
) -> pd.DataFrame:
    terms = ["EI", "SN", "TF", "JP", "EI_SN", "EI_TF", "EI_JP", "SN_TF", "SN_JP", "TF_JP"]
    design_rows = []
    for model in models:
        default_vec = profiles[(model, "Default")]
        for persona in personas:
            codes = axis_code(persona)
            row = {
                "model": model,
                "persona": persona,
                "EI": codes["EI"],
                "SN": codes["SN"],
                "TF": codes["TF"],
                "JP": codes["JP"],
            }
            row["EI_SN"] = row["EI"] * row["SN"]
            row["EI_TF"] = row["EI"] * row["TF"]
            row["EI_JP"] = row["EI"] * row["JP"]
            row["SN_TF"] = row["SN"] * row["TF"]
            row["SN_JP"] = row["SN"] * row["JP"]
            row["TF_JP"] = row["TF"] * row["JP"]
            delta = profiles[(model, persona)] - default_vec
            for i, domain in enumerate(domains):
                row[domain] = delta[i]
            design_rows.append(row)

    design = pd.DataFrame(design_rows)
    x = np.column_stack([np.ones(len(design))] + [design[t].to_numpy(dtype=float) for t in terms])
    names = ["Intercept"] + terms
    rows = []
    for domain in domains:
        y = design[domain].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        yhat = x @ beta
        resid = y - yhat
        df_resid = len(y) - x.shape[1]
        s2 = float((resid @ resid) / df_resid)
        xtx_inv = np.linalg.pinv(x.T @ x)
        se = np.sqrt(np.diag(xtx_inv) * s2)
        tvals = beta / se
        pvals = 2 * stats.t.sf(np.abs(tvals), df_resid)
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) @ (y - y.mean())))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        for name, coef, se_val, t_val, p_val in zip(names, beta, se, tvals, pvals):
            rows.append(
                {
                    "domain": domain,
                    "term": name,
                    "coef": coef,
                    "se": se_val,
                    "t": t_val,
                    "p": p_val,
                    "r2": r2,
                    "n": len(y),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "factorial_mbti_effects.csv", index=False)
    return df


def build_item_frame(all_results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model, payload in all_results.items():
        for persona, pdata in payload["results_by_persona"].items():
            for response in pdata["responses"]:
                rows.append(
                    {
                        "model": model,
                        "persona": persona,
                        "item_id": response["item_id"],
                        "scale": response["scale"],
                        "domain": response["domain"],
                        "facet": response.get("facet"),
                        "keyed": response["keyed"],
                        "response_format": response["response_format"],
                        "item_text": response.get("item_text", ""),
                        "parsed_value": response.get("parsed_value"),
                        "scored_value": response.get("scored_value"),
                        "parse_failed": bool(response.get("parse_failed")),
                    }
                )
    df = pd.DataFrame(rows)
    df["parsed_value"] = pd.to_numeric(df["parsed_value"], errors="coerce")
    df["scored_value"] = pd.to_numeric(df["scored_value"], errors="coerce")
    return df


def text_flags(text: str, scale: str, domain: str) -> dict[str, bool]:
    low = text.lower()
    dark = scale == "SD3" or domain in {"Psychopathy", "Machiavellianism", "Aggression-Hostility", "Psychoticism"}
    safety_terms = [
        "drug",
        "danger",
        "dangerous",
        "sex",
        "revenge",
        "payback",
        "nasty",
        "law",
        "fight",
        "hit",
        "manipulat",
        "cheat",
        "crime",
        "criminal",
        "hurt",
    ]
    self_terms = ["myself", "body", "physical", "sex", "sleep", "eat", "food", "emotion", "feel"]
    return {
        "dark_or_antisocial_flag": dark,
        "safety_sensitive_flag": any(term in low for term in safety_terms),
        "ai_self_or_embodied_flag": any(term in low for term in self_terms),
    }


def side_sign(value: float, response_format: str) -> int:
    if not np.isfinite(value):
        return 0
    if response_format == "likert_5":
        if value > 3:
            return 1
        if value < 3:
            return -1
        return 0
    if value > 0.5:
        return 1
    if value < 0.5:
        return -1
    return 0


def run_item_plasticity(item_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    default = item_df[item_df["persona"] == "Default"].copy()
    mbti = item_df[item_df["persona"] != "Default"].copy()
    merged = mbti.merge(
        default[
            [
                "model",
                "item_id",
                "parsed_value",
                "scored_value",
            ]
        ].rename(columns={"parsed_value": "default_parsed", "scored_value": "default_scored"}),
        on=["model", "item_id"],
        how="left",
    )
    merged["range_width"] = merged["response_format"].map(SCALE_RANGES).fillna(1.0)
    valid = merged.dropna(subset=["parsed_value", "default_parsed", "scored_value", "default_scored"]).copy()
    valid["raw_delta_norm"] = (valid["parsed_value"] - valid["default_parsed"]) / valid["range_width"]
    valid["scored_delta_norm"] = (valid["scored_value"] - valid["default_scored"]) / valid["range_width"]
    valid["abs_raw_delta_norm"] = valid["raw_delta_norm"].abs()
    valid["abs_scored_delta_norm"] = valid["scored_delta_norm"].abs()
    valid["value_changed"] = valid["parsed_value"] != valid["default_parsed"]
    valid["directional_flip"] = [
        side_sign(v, fmt) != side_sign(d, fmt) and side_sign(v, fmt) != 0 and side_sign(d, fmt) != 0
        for v, d, fmt in zip(valid["parsed_value"], valid["default_parsed"], valid["response_format"])
    ]

    rows = []
    for item_id, g in valid.groupby("item_id"):
        first = g.iloc[0]
        flags = text_flags(first["item_text"], first["scale"], first["domain"])
        rows.append(
            {
                "item_id": item_id,
                "scale": first["scale"],
                "domain": first["domain"],
                "facet": first["facet"],
                "keyed": first["keyed"],
                "response_format": first["response_format"],
                "item_text": first["item_text"],
                "n_valid_pairs": len(g),
                "mean_abs_raw_delta": g["abs_raw_delta_norm"].mean(),
                "mean_abs_scored_delta": g["abs_scored_delta_norm"].mean(),
                "value_change_rate": g["value_changed"].mean(),
                "directional_flip_rate": g["directional_flip"].mean(),
                "mean_raw_delta": g["raw_delta_norm"].mean(),
                "mean_scored_delta": g["scored_delta_norm"].mean(),
                "default_mean_raw": g["default_parsed"].mean(),
                "persona_mean_raw": g["parsed_value"].mean(),
                **flags,
            }
        )

    out = pd.DataFrame(rows).sort_values("mean_abs_raw_delta", ascending=False)
    q10 = out["mean_abs_raw_delta"].quantile(0.10)
    q90 = out["mean_abs_raw_delta"].quantile(0.90)
    out["plasticity_class"] = np.where(
        out["mean_abs_raw_delta"] >= q90,
        "plastic",
        np.where(out["mean_abs_raw_delta"] <= q10, "locked", "intermediate"),
    )
    out["plasticity_rank"] = out["mean_abs_raw_delta"].rank(ascending=False, method="first").astype(int)
    out.to_csv(OUT_DIR / "item_plasticity.csv", index=False)
    valid.to_csv(OUT_DIR / "item_plasticity_long.csv", index=False)
    return out, valid


def run_directional_pair_coherence(valid_item_delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, persona, scale, domain), g in valid_item_delta.groupby(["model", "persona", "scale", "domain"]):
        fwd = g[g["keyed"] == "+"]
        rev = g[g["keyed"] == "-"]
        if len(fwd) == 0 or len(rev) == 0:
            continue
        raw_pairs = []
        scored_pairs = []
        for _, frow in fwd.iterrows():
            for _, rrow in rev.iterrows():
                sf = np.sign(frow["raw_delta_norm"])
                sr = np.sign(rrow["raw_delta_norm"])
                ssf = np.sign(frow["scored_delta_norm"])
                ssr = np.sign(rrow["scored_delta_norm"])
                if sf != 0 and sr != 0:
                    raw_pairs.append(sf != sr)
                if ssf != 0 and ssr != 0:
                    scored_pairs.append(ssf == ssr)
        rows.append(
            {
                "model": model,
                "persona": persona,
                "scale": scale,
                "domain": domain,
                "n_forward": len(fwd),
                "n_reverse": len(rev),
                "n_raw_nonzero_pairs": len(raw_pairs),
                "n_scored_nonzero_pairs": len(scored_pairs),
                "directional_pair_coherence": float(np.mean(raw_pairs)) if raw_pairs else np.nan,
                "scored_pair_alignment": float(np.mean(scored_pairs)) if scored_pairs else np.nan,
                "mean_forward_raw_delta": fwd["raw_delta_norm"].mean(),
                "mean_reverse_raw_delta": rev["raw_delta_norm"].mean(),
                "mean_forward_scored_delta": fwd["scored_delta_norm"].mean(),
                "mean_reverse_scored_delta": rev["scored_delta_norm"].mean(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "directional_pair_coherence.csv", index=False)
    return df


def run_robustness(
    profiles: dict[tuple[str, str], np.ndarray],
    models: list[str],
    personas: list[str],
    domains: list[str],
    inv_cov: np.ndarray,
) -> pd.DataFrame:
    subsets = {
        "all": list(range(len(domains))),
        "likert_only": [i for i, d in enumerate(domains) if d.startswith("IPIP-NEO-120") or d.startswith("SD3")],
        "binary_only": [i for i, d in enumerate(domains) if d.startswith("ZKPQ-50-CC") or d.startswith("EPQR-A")],
        "ipip_only": [i for i, d in enumerate(domains) if d.startswith("IPIP-NEO-120")],
    }
    rows = []
    for subset_name, indices in subsets.items():
        detail, _ = run_persona_fidelity(profiles, models, personas, inv_cov, indices, subset_name)
        for metric in ["z_euclidean", "cosine", "mahalanobis"]:
            rows.append(
                {
                    "subset": subset_name,
                    "metric": metric,
                    "top1_accuracy": detail[f"correct_{metric}"].mean(),
                    "mean_margin": detail[f"margin_{metric}"].mean(),
                    "n": len(detail),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "persona_fidelity_robustness.csv", index=False)
    return df


def bootstrap_ci(default_df: pd.DataFrame, adherence_df: pd.DataFrame, coherence_df: pd.DataFrame) -> pd.DataFrame:
    model_names = sorted(default_df["model"].unique())
    per_model_default = default_df.set_index("model")["nearest_z_euclidean_distance"]
    per_model_adherence = adherence_df.groupby("model")["empirical_cosine"].mean()
    per_model_coherence = coherence_df.groupby("model")["coherence_score"].mean()
    metrics = {
        "default_nearest_distance": per_model_default,
        "target_empirical_cosine": per_model_adherence,
        "cross_scale_coherence": per_model_coherence,
    }
    rows = []
    for name, series in metrics.items():
        observed = float(series.mean())
        boot = []
        values = series.reindex(model_names).to_numpy(dtype=float)
        for _ in range(1000):
            idx = RNG.integers(0, len(values), len(values))
            boot.append(float(np.nanmean(values[idx])))
        rows.append(
            {
                "statistic": name,
                "observed": observed,
                "ci_2.5": float(np.nanpercentile(boot, 2.5)),
                "ci_97.5": float(np.nanpercentile(boot, 97.5)),
                "bootstrap_unit": "model",
                "n_bootstrap": 1000,
                "seed": 42,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "persona_steering_bootstrap_ci.csv", index=False)
    return df


def savefig(fig: plt.Figure, filename: str) -> None:
    for directory in [FIG_DIR, FINAL_DIR]:
        fig.savefig(directory / filename, dpi=320, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def plot_default_bias_and_adherence(default_df: pd.DataFrame, adherence_df: pd.DataFrame) -> None:
    models = default_df.sort_values("nearest_z_euclidean_distance")["model"].tolist()
    heat = default_df.set_index("model").loc[
        models, [f"empirical_{axis}_projection" for axis in AXES]
    ]
    fig = plt.figure(figsize=(FULL_W, 4.35))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.34, 1.02], height_ratios=[0.68, 1.42], wspace=0.34, hspace=0.52)
    ax1 = fig.add_subplot(gs[:, 0])
    vmax = np.nanmax(np.abs(heat.to_numpy()))
    im = ax1.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax1.set_xticks(np.arange(-0.5, len(AXES), 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax1.grid(which="minor", color="white", linewidth=0.55)
    ax1.tick_params(which="minor", bottom=False, left=False)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels([short_model(m) for m in models], fontsize=8)
    ax1.set_xticks(range(len(AXES)))
    ax1.set_xticklabels(["E-I", "N-S", "F-T", "J-P"], fontsize=8)
    ax1.set_title("Default bias on empirical axes", loc="left", pad=5, fontweight="bold")
    panel_label(ax1, "A")
    cbar = fig.colorbar(im, ax=ax1, shrink=0.58, pad=0.025)
    cbar.set_label("Projection", rotation=90, labelpad=6)
    cbar.outline.set_linewidth(0.45)

    ax2 = fig.add_subplot(gs[0, 1])
    counts = default_df["nearest_z_euclidean"].value_counts().reindex(MBTI_ORDER).dropna()
    ax2.bar(counts.index, counts.values, color=PALETTE["blue"], width=0.68)
    ax2.set_title("Nearest default centroid", loc="left", pad=5, fontweight="bold")
    panel_label(ax2, "B")
    ax2.set_ylabel("Models")
    ax2.tick_params(axis="x", rotation=0, labelsize=6.5)
    ax2.set_ylim(0, max(counts.values) + 1.0)
    clean_axes(ax2, "y")

    ax3 = fig.add_subplot(gs[1, 1])
    means = adherence_df.groupby("model")["empirical_cosine"].mean().sort_values(ascending=False)
    colors = [PALETTE["green"] if v >= 0.8 else PALETTE["orange"] for v in means.values]
    ax3.barh([short_model(m) for m in means.index], means.values, color=colors, height=0.7)
    ax3.invert_yaxis()
    ax3.axvline(means.mean(), color=PALETTE["black"], linewidth=0.7, linestyle="--")
    ax3.set_xlim(0.52, 0.95)
    ax3.set_xlabel("Cosine with empirical target")
    ax3.set_title("Target adherence", loc="left", pad=5, fontweight="bold")
    panel_label(ax3, "C")
    ax3.tick_params(axis="y", labelsize=5.6)
    clean_axes(ax3, "x")
    fig.align_ylabels([ax1, ax2, ax3])
    savefig(fig, "fig11_default_bias_and_adherence.png")


def plot_confusion(confusion: pd.DataFrame, detail: pd.DataFrame) -> None:
    matrix = confusion.pivot_table(
        index="actual_persona", columns="predicted_persona", values="count", fill_value=0
    ).reindex(index=MBTI_ORDER, columns=MBTI_ORDER, fill_value=0)
    accuracy = detail["correct_z_euclidean"].mean()
    data = matrix.to_numpy(dtype=float)
    mask = np.ma.masked_where(data == 0, data)
    fig, ax = plt.subplots(figsize=(COL_W, 3.25))
    ax.imshow(np.zeros_like(data), cmap=matplotlib.colors.ListedColormap(["#F7F8FA"]), vmin=0, vmax=1)
    im = ax.imshow(mask, cmap="Blues", vmin=0, vmax=18, interpolation="nearest")
    # Mark the rare off-diagonal confusions in a warm outline.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > 0 and i != j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=PALETTE["red"], linewidth=1.1))
    ax.set_xticks(range(len(MBTI_ORDER)))
    ax.set_xticklabels(MBTI_ORDER, rotation=45, ha="right", fontsize=5.8)
    ax.set_yticks(range(len(MBTI_ORDER)))
    ax.set_yticklabels(MBTI_ORDER, fontsize=5.8)
    ax.set_xticks(np.arange(-0.5, len(MBTI_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MBTI_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(MBTI_ORDER)):
        for j in range(len(MBTI_ORDER)):
            val = int(matrix.iloc[i, j])
            if val:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=5.8, color="white" if val >= 10 else PALETTE["black"])
    ax.set_xlabel("Predicted centroid")
    ax.set_ylabel("Prompted persona")
    ax.set_title(f"Persona fidelity ({accuracy:.1%} top-1)", loc="left", pad=5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("Models", labelpad=5)
    cbar.outline.set_linewidth(0.45)
    savefig(fig, "fig12_persona_fidelity_confusion.png")


def plot_cross_scale(coherence_df: pd.DataFrame) -> None:
    summary = coherence_df.groupby("construct").agg(
        coherence=("coherence_score", "mean"), sign_match=("sign_match_rate", "mean")
    )
    order = ["Extraversion", "Neuroticism", "Agreeableness_vs_Antagonism", "Conscientiousness_vs_Disinhibition"]
    summary = summary.reindex([o for o in order if o in summary.index])
    persona_heat = coherence_df.pivot_table(index="construct", columns="persona", values="coherence_score", aggfunc="mean")
    persona_heat = persona_heat.reindex(index=summary.index, columns=MBTI_ORDER)
    pretty = {
        "Extraversion": "Extra.",
        "Neuroticism": "Neuro.",
        "Agreeableness_vs_Antagonism": "Agree. vs Antag.",
        "Conscientiousness_vs_Disinhibition": "Consc. vs Disinh.",
    }
    fig = plt.figure(figsize=(FULL_W, 3.15))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.28], hspace=0.42)
    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(summary))
    ax1.hlines(y, summary["sign_match"], summary["coherence"], color="#B8C0C8", linewidth=1.4)
    ax1.scatter(summary["sign_match"], y, color=PALETTE["orange"], s=22, label="Sign match", zorder=3)
    ax1.scatter(summary["coherence"], y, color=PALETTE["purple"], s=22, label="Coherence", zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([pretty.get(s, s) for s in summary.index], fontsize=7)
    ax1.set_xlim(0.65, 0.94)
    ax1.set_xlabel("Mean score")
    ax1.set_title("Construct-level agreement", loc="left", pad=5, fontweight="bold")
    panel_label(ax1, "A")
    ax1.legend(loc="lower right", ncol=2, frameon=False)
    clean_axes(ax1, "x")

    ax2 = fig.add_subplot(gs[1, 0])
    im = ax2.imshow(persona_heat.to_numpy(), cmap="viridis", vmin=0.65, vmax=1, aspect="auto", interpolation="nearest")
    ax2.set_xticks(np.arange(-0.5, len(MBTI_ORDER), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(persona_heat.index), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=0.5)
    ax2.tick_params(which="minor", bottom=False, left=False)
    ax2.set_yticks(range(len(persona_heat.index)))
    ax2.set_yticklabels([pretty.get(s, s) for s in persona_heat.index], fontsize=7)
    ax2.set_xticks(range(len(MBTI_ORDER)))
    ax2.set_xticklabels(MBTI_ORDER, rotation=0, fontsize=6.5)
    ax2.set_title("Coherence by prompted persona", loc="left", pad=5, fontweight="bold")
    panel_label(ax2, "B")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.018, pad=0.012)
    cbar.set_label("Coherence", labelpad=5)
    cbar.outline.set_linewidth(0.45)
    savefig(fig, "fig13_cross_scale_persona_coherence.png")


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    var = np.var(coords, axis=0)
    ratio = var / np.var(centered, axis=0).sum()
    return coords, ratio


def plot_vector_field(
    profiles: dict[tuple[str, str], np.ndarray], models: list[str], personas: list[str]
) -> None:
    keys = [(model, persona) for model in models for persona in ["Default"] + personas]
    x = np.array([profiles[k] for k in keys])
    coords, ratio = pca_2d(x)
    coord_map = {key: coords[i] for i, key in enumerate(keys)}
    default_centroid = np.mean([coord_map[(m, "Default")] for m in models], axis=0)
    persona_centroids = {p: np.mean([coord_map[(m, p)] for m in models], axis=0) - default_centroid for p in personas}

    fig, ax = plt.subplots(figsize=(COL_W, 3.05))
    for persona, c in persona_centroids.items():
        color = PALETTE["blue"] if persona.startswith("I") else PALETTE["orange"]
        alpha = 0.85 if persona[3] == "P" else 0.65
        ax.annotate(
            "",
            xy=(c[0], c[1]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.85, alpha=alpha, shrinkA=0, shrinkB=2),
            zorder=2,
        )
        ax.scatter(c[0], c[1], s=18, c=color, edgecolor="white", linewidth=0.45, zorder=3)
        ha = "left" if c[0] >= 0 else "right"
        va = "bottom" if c[1] >= 0 else "top"
        ax.annotate(
            persona,
            (c[0], c[1]),
            xytext=(2 if ha == "left" else -2, 1 if va == "bottom" else -1),
            textcoords="offset points",
            fontsize=5.9,
            ha=ha,
            va=va,
        )
    ax.scatter(0, 0, s=46, marker="*", c=PALETTE["black"], zorder=5)
    ax.axhline(0, color="#D8DDE2", linewidth=0.6)
    ax.axvline(0, color="#D8DDE2", linewidth=0.6)
    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}% variance)")
    ax.set_title("Persona centroid shifts", loc="left", pad=5, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    clean_axes(ax, None)
    savefig(fig, "fig14_persona_vector_field.png")


def plot_item_plasticity(item_plasticity: pd.DataFrame) -> None:
    top = item_plasticity.sort_values("mean_abs_raw_delta", ascending=False).head(10)
    bottom = item_plasticity.sort_values("mean_abs_raw_delta", ascending=True).head(10)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.25), sharex=False)
    for ax, data, title, color in [
        (axes[0], top.iloc[::-1], "Most plastic items", PALETTE["red"]),
        (axes[1], bottom.iloc[::-1], "Most locked items", PALETTE["blue"]),
    ]:
        labels = [f"{r.item_id}  {str(r.item_text)[:28]}" for r in data.itertuples()]
        ax.barh(labels, data["mean_abs_raw_delta"], color=color, height=0.68)
        ax.set_title(title, loc="left", pad=5, fontweight="bold")
        ax.set_xlabel("Mean absolute shift")
        ax.tick_params(axis="y", labelsize=5.8)
        clean_axes(ax, "x")
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    savefig(fig, "figA1_item_plasticity.png")


def plot_interactions(factorial_df: pd.DataFrame) -> None:
    interactions = ["EI_JP", "TF_JP", "SN_TF", "EI_TF", "SN_JP", "EI_SN"]
    df = factorial_df[factorial_df["term"].isin(interactions)]
    pivot = df.pivot_table(index="domain", columns="term", values="coef")
    order = pivot.abs().max(axis=1).sort_values(ascending=False).index[:17]
    pivot = pivot.loc[order, interactions]
    fig, ax = plt.subplots(figsize=(COL_W, 3.25))
    vmax = np.nanmax(np.abs(pivot.to_numpy()))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, len(interactions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([clean_label(d) for d in pivot.index], fontsize=5.8)
    ax.set_xticks(range(len(interactions)))
    ax.set_xticklabels([i.replace("_", "x") for i in interactions], rotation=45, ha="right", fontsize=6.3)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=4.8, color="white" if abs(val) > vmax * 0.55 else PALETTE["black"])
    ax.set_title("MBTI interaction effects", loc="left", pad=5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("Coefficient", labelpad=5)
    cbar.outline.set_linewidth(0.45)
    savefig(fig, "figA2_mbti_interactions.png")


def plot_robustness(robustness_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    subsets = list(robustness_df["subset"].unique())
    metrics = ["z_euclidean", "cosine", "mahalanobis"]
    x = np.arange(len(subsets))
    width = 0.24
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]
    for i, metric in enumerate(metrics):
        vals = robustness_df[robustness_df["metric"] == metric].set_index("subset").reindex(subsets)["top1_accuracy"]
        ax.bar(x + (i - 1) * width, vals, width, label=metric, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["All", "Likert", "Binary", "IPIP"], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Fidelity robustness", loc="left", pad=5, fontweight="bold")
    ax.legend(fontsize=6.2, frameon=False, ncol=1, loc="lower left")
    clean_axes(ax, "y")
    savefig(fig, "figA3_robustness_sensitivity.png")


def write_summary(
    all_results: dict[str, dict],
    default_df: pd.DataFrame,
    adherence_df: pd.DataFrame,
    fidelity_detail: pd.DataFrame,
    coherence_df: pd.DataFrame,
    item_plasticity: pd.DataFrame,
    pair_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
) -> None:
    n_models = len(all_results)
    n_personas = len(MBTI_ORDER) + 1
    n_items = len(load_items())
    n_points = n_models * n_personas * n_items
    missing = sum(
        1
        for payload in all_results.values()
        for pdata in payload["results_by_persona"].values()
        for response in pdata["responses"]
        if response.get("parsed_value") is None
    )
    nearest_counts = default_df["nearest_z_euclidean"].value_counts()
    top_nearest = nearest_counts.index[0]
    top_nearest_n = int(nearest_counts.iloc[0])
    empirical_cos = adherence_df["empirical_cosine"].mean()
    theory_cos = adherence_df["theory_cosine"].mean()
    leakage = adherence_df["empirical_off_target_leakage"].mean()
    accuracy = fidelity_detail["correct_z_euclidean"].mean()
    most_confused = (
        fidelity_detail[fidelity_detail["predicted_z_euclidean"] != fidelity_detail["actual_persona"]]
        .groupby(["actual_persona", "predicted_z_euclidean"])
        .size()
        .sort_values(ascending=False)
    )
    most_confused_text = "none"
    if len(most_confused):
        (actual, pred), count = most_confused.index[0], int(most_confused.iloc[0])
        most_confused_text = f"{actual} -> {pred} ({count} models)"
    coherence_mean = coherence_df["coherence_score"].mean()
    sign_match = coherence_df["sign_match_rate"].mean()
    locked = int((item_plasticity["plasticity_class"] == "locked").sum())
    plastic = int((item_plasticity["plasticity_class"] == "plastic").sum())
    pair_coherence = pair_df["directional_pair_coherence"].mean()

    lines = [
        "PERSONA STEERING ANALYSIS SUMMARY",
        "=" * 42,
        f"Models: {n_models}",
        f"Personas: {n_personas} (Default + 16 MBTI)",
        f"Items: {n_items}",
        f"Expected item-level responses: {n_points:,}",
        f"Missing parsed values: {missing}",
        "",
        "Default bias",
        "-" * 42,
        f"Most common nearest persona centroid: {top_nearest} ({top_nearest_n}/{n_models} models)",
        f"Mean nearest-centroid distance: {default_df['nearest_z_euclidean_distance'].mean():.3f}",
        "",
        "Target adherence",
        "-" * 42,
        f"Mean empirical target cosine: {empirical_cos:.3f}",
        f"Mean theory target cosine: {theory_cos:.3f}",
        f"Mean empirical off-target leakage: {leakage:.3f}",
        "",
        "Persona fidelity",
        "-" * 42,
        f"Top-1 accuracy (z-Euclidean LOO centroids): {accuracy:.3f}",
        f"Most common confusion: {most_confused_text}",
        "",
        "Cross-scale coherence",
        "-" * 42,
        f"Mean coherence score: {coherence_mean:.3f}",
        f"Mean sign-match rate: {sign_match:.3f}",
        "",
        "Item mechanism",
        "-" * 42,
        f"Plastic items: {plastic}; locked items: {locked}",
        f"Mean directional pair coherence: {pair_coherence:.3f}",
        "",
        "Bootstrap CIs",
        "-" * 42,
    ]
    for row in bootstrap_df.itertuples(index=False):
        lines.append(f"{row.statistic}: {row.observed:.3f} [{row._2:.3f}, {row._3:.3f}]")

    (OUT_DIR / "persona_steering_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Loading completed experiment results...")
    all_results = load_results()
    items = load_items()
    domains = domain_order_from_results(all_results)
    models = sorted(all_results.keys())
    personas = [p for p in MBTI_ORDER if p in all_results[models[0]]["results_by_persona"]]

    observed = sum(len(pdata["responses"]) for payload in all_results.values() for pdata in payload["results_by_persona"].values())
    expected = len(models) * (len(personas) + 1) * len(items)
    missing = sum(
        1
        for payload in all_results.values()
        for pdata in payload["results_by_persona"].values()
        for response in pdata["responses"]
        if response.get("parsed_value") is None
    )
    print(f"Data integrity: observed={observed:,}, expected={expected:,}, missing_parsed={missing}")
    if observed != expected:
        raise RuntimeError(f"Unexpected item count: observed {observed}, expected {expected}")

    score_df = build_domain_frame(all_results, domains)
    z_df, _, _ = zscore_domain_frame(score_df, domains)
    profiles = make_profile_map(z_df, domains)
    inv_cov = covariance_inverse(profiles, models, personas)
    theory_axes = make_theory_axes(domains)

    print("Running default bias analysis...")
    default_df = run_default_bias(profiles, models, personas, theory_axes, inv_cov)
    print("Running target adherence analysis...")
    adherence_df = run_target_adherence(profiles, models, personas, theory_axes)
    print("Running persona fidelity analysis...")
    fidelity_detail, confusion = run_persona_fidelity(profiles, models, personas, inv_cov)
    print("Running cross-scale coherence analysis...")
    coherence_df, pair_corr_df = run_cross_scale_coherence(profiles, models, personas, domains)
    print("Running factorial MBTI effects...")
    factorial_df = run_factorial_effects(profiles, models, personas, domains)
    print("Running item plasticity and pair coherence...")
    item_df = build_item_frame(all_results)
    item_plasticity, valid_item_delta = run_item_plasticity(item_df)
    directional_pairs = run_directional_pair_coherence(valid_item_delta)
    print("Running robustness and bootstrap checks...")
    robustness_df = run_robustness(profiles, models, personas, domains, inv_cov)
    bootstrap_df = bootstrap_ci(default_df, adherence_df, coherence_df)

    print("Generating figures...")
    plot_default_bias_and_adherence(default_df, adherence_df)
    plot_confusion(confusion, fidelity_detail)
    plot_cross_scale(coherence_df)
    plot_vector_field(profiles, models, personas)
    plot_item_plasticity(item_plasticity)
    plot_interactions(factorial_df)
    plot_robustness(robustness_df)

    write_summary(
        all_results,
        default_df,
        adherence_df,
        fidelity_detail,
        coherence_df,
        item_plasticity,
        directional_pairs,
        bootstrap_df,
    )

    print("Done.")
    print((OUT_DIR / "persona_steering_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
