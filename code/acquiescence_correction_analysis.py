#!/usr/bin/env python3
"""
Acquiescence-correction re-analysis (rebuttal for Reviewer ys6N).

Question: if acquiescence is the single root cause of the psychometric failures
reported in the paper, does REMOVING it (via the textbook within-respondent
mean-centering correction) recover coherent measurement?

We recompute Cronbach's alpha per IPIP-NEO-120 domain under three conditions:
  A. baseline   = raw parsed_value, no reverse scoring (the paper's method)
  B. trait      = reverse-scored (scored_value), no acquiescence removal
  C. corrected  = acquiescence-corrected (within-respondent mean-centered
                  across all 120 IPIP items) THEN reverse-scored

Purely on the EXISTING 375,700-response dataset (paper's 20 models). No new
model calls. The two qwen3-0.6b base/instruct files are excluded so n=20.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
IPIP = "IPIP-NEO-120"
DOMAINS = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
EXCLUDE = ("qwen3-0.6b-base", "qwen3-0.6b-instruct")  # extra base/instruct comparison


def load_all():
    res = {}
    for fp in sorted(DATA_DIR.glob("exp_mbti_*.json")):
        name = fp.stem.replace("exp_mbti_", "")
        if name in EXCLUDE:
            continue
        res[name] = json.load(open(fp))
    return res


def _mean_field(r, field):
    if "samples" in r and r["samples"]:
        vals = [s[field] for s in r["samples"] if s.get(field) is not None]
        return sum(vals) / len(vals) if vals else None
    return r.get(field)


def extract(model_data, persona):
    rows = []
    for r in model_data["results_by_persona"][persona]["responses"]:
        rows.append({
            "item_id": r["item_id"], "scale": r["scale"], "domain": r["domain"],
            "keyed": r["keyed"], "response_format": r["response_format"],
            "parsed_value": _mean_field(r, "parsed_value"),
            "scored_value": _mean_field(r, "scored_value"),
        })
    df = pd.DataFrame(rows)
    df["parsed_value"] = pd.to_numeric(df["parsed_value"], errors="coerce")
    df["scored_value"] = pd.to_numeric(df["scored_value"], errors="coerce")
    null = df["parsed_value"].isna()
    for idx in df[null].index:
        row = df.loc[idx]
        same = df[(df.domain == row.domain) & (df.scale == row.scale) & df.parsed_value.notna()]
        if len(same):
            med = same.parsed_value.median()
            df.at[idx, "parsed_value"] = med
            df.at[idx, "scored_value"] = med if row.keyed == "+" else (6 - med if row.response_format == "likert_5" else med)
    return df


def cronbach_alpha(X: np.ndarray) -> float:
    """X: (n_respondents, k_items). Paper's method: var ddof=1."""
    k = X.shape[1]
    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return 0.0
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def main() -> None:
    all_results = load_all()
    cond_A = {d: [] for d in DOMAINS}
    cond_B = {d: [] for d in DOMAINS}
    cond_C = {d: [] for d in DOMAINS}
    delta_raw, acq_index = [], []
    n_models = 0

    for model_name, mdata in all_results.items():
        stacks_A = {d: [] for d in DOMAINS}
        stacks_B = {d: [] for d in DOMAINS}
        stacks_C = {d: [] for d in DOMAINS}
        model_deltas, model_acq = [], []

        for persona in mdata["results_by_persona"]:
            df = extract(mdata, persona)
            ip = df[df["scale"] == IPIP].sort_values("item_id").reset_index(drop=True)
            if len(ip) != 120:
                continue
            raw = ip["parsed_value"].values.astype(float)
            scored = ip["scored_value"].values.astype(float)
            keyed = ip["keyed"].values
            domain = ip["domain"].values

            mu = raw.mean()
            cor = raw - mu
            cor_trait = np.where(keyed == "+", cor, -cor)

            for d in DOMAINS:
                mask = domain == d
                if mask.sum() < 3:
                    continue
                stacks_A[d].append(raw[mask])
                stacks_B[d].append(scored[mask])
                stacks_C[d].append(cor_trait[mask])

            fwd = raw[keyed == "+"]
            rev = raw[keyed == "-"]
            model_deltas.append(fwd.mean() - rev.mean())
            model_acq.append((mu - 3.0) / 2.0)

        if any(len(stacks_A[d]) < 3 for d in DOMAINS):
            continue
        n_models += 1
        for d in DOMAINS:
            cond_A[d].append(cronbach_alpha(np.array(stacks_A[d])))
            cond_B[d].append(cronbach_alpha(np.array(stacks_B[d])))
            cond_C[d].append(cronbach_alpha(np.array(stacks_C[d])))
        delta_raw.append(np.mean(model_deltas))
        acq_index.append(np.mean(model_acq))

    print(f"\nModels analysed: {n_models}")
    print(f"Mean acquiescence index (IPIP, (mean-3)/2): {np.mean(acq_index):+.3f}  "
          f"[{min(acq_index):+.3f}, {max(acq_index):+.3f}]")
    print(f"Mean raw forward-reverse gap Δ (IPIP): {np.mean(delta_raw):+.3f}\n")

    print(f"{'Domain':18s} | {'A baseline':>11s} | {'B trait':>9s} | {'C corrected':>11s} | {'C−A':>7s}")
    print("-" * 66)
    rows = []
    for d in DOMAINS:
        a, b, c = np.mean(cond_A[d]), np.mean(cond_B[d]), np.mean(cond_C[d])
        rows.append({"domain": d, "A_baseline_raw": round(a, 4),
                     "B_trait_aligned": round(b, 4),
                     "C_acquiescence_corrected": round(c, 4)})
        print(f"{d:18s} | {a:11.3f} | {b:9.3f} | {c:11.3f} | {c-a:+7.3f}")
    print("-" * 66)
    meanA = np.mean([np.mean(cond_A[d]) for d in DOMAINS])
    meanB = np.mean([np.mean(cond_B[d]) for d in DOMAINS])
    meanC = np.mean([np.mean(cond_C[d]) for d in DOMAINS])
    print(f"{'MEAN over 5 domains':18s} | {meanA:11.3f} | {meanB:9.3f} | {meanC:11.3f}")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "acquiescence_correction_alpha.csv", index=False)
    print(f"\nSaved: results/acquiescence_correction_alpha.csv")


if __name__ == "__main__":
    main()
