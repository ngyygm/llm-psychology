#!/usr/bin/env python3
"""
Per-persona breakdown of MBTI persona steering: which archetypes does each
model steer toward most reliably, and which does it miss?

Outputs:
  results/corrected_analysis/mbti_per_persona.csv
  results/corrected_analysis/mbti_dimension_shifts.csv

Used in Appendix M of the paper to give reviewers a granular view of where
persona steering succeeds and fails.
"""

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from run_mbti_personas import MBTI_TYPES

OUT_DIR = Path("results/corrected_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BFI = ["bfi.extraversion", "bfi.agreeableness", "bfi.conscientiousness",
       "bfi.neuroticism", "bfi.openness"]
DIM_MAP = {"bfi.extraversion": "E", "bfi.agreeableness": "A",
           "bfi.conscientiousness": "C", "bfi.neuroticism": "N_FFM",
           "bfi.openness": "O"}


def load_mbti():
    rows = []
    for f in sorted(Path("results/mbti_persona/corrected").glob("*.json")):
        try:
            rows.extend(json.loads(f.read_text()))
        except Exception:
            pass
    return pd.DataFrame(rows)


def load_default():
    rows = []
    for f in sorted(Path("results/vendor_exp/corrected").glob("*.json")):
        if "checkpoint" in f.name:
            continue
        try:
            rows.extend(json.loads(f.read_text()))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    main = df[(df["study"].isin([1, 2, 6])) &
              (df["thinking_mode"].fillna("chat") == "chat")]
    if "prompt_variant" in main.columns:
        main = main[(main["prompt_variant"].fillna("") == "") |
                    (main["prompt_variant"] == "default")]
    return main.groupby("model_id")[BFI].mean().to_dict("index")


def per_persona_table(mbti_df, default_means):
    rows = []
    for (model_id, persona), sub in mbti_df.groupby(["model_id", "mbti_type"]):
        base = default_means.get(model_id, {})
        if not base:
            continue
        means = sub[BFI].mean().to_dict()
        n_pred, n_hit = 0, 0
        per_dim = {}
        for col in BFI:
            short = DIM_MAP[col]
            pred = MBTI_TYPES[persona]["big5_pred"].get(short, 0)
            obs = means[col] - base[col]
            per_dim[f"shift_{short}"] = round(obs, 3)
            if pred == 0 or abs(obs) < 0.05:
                continue
            n_pred += 1
            n_hit += int(np.sign(obs) == np.sign(pred))
        rows.append({
            "model_id": model_id, "mbti_type": persona,
            "n_predictions": n_pred, "n_hits": n_hit,
            "hit_rate": round(n_hit / n_pred, 3) if n_pred else None,
            **per_dim,
        })
    return pd.DataFrame(rows)


def main():
    mbti_df = load_mbti()
    if mbti_df.empty:
        print("No MBTI data; run run_mbti_personas.py first.")
        return
    default_means = load_default()
    table = per_persona_table(mbti_df, default_means)
    print(f"\n{'Model':<25s} {'Best persona':<10s} {'Worst persona':<12s} mean hit_rate")
    for model_id, sub in table.groupby("model_id"):
        sub = sub.dropna(subset=["hit_rate"])
        if sub.empty:
            continue
        best = sub.loc[sub["hit_rate"].idxmax()]
        worst = sub.loc[sub["hit_rate"].idxmin()]
        mean_hr = sub["hit_rate"].mean()
        print(f"  {model_id:<25s} {best['mbti_type']} ({best['hit_rate']:.2f})  "
              f"{worst['mbti_type']} ({worst['hit_rate']:.2f})  {mean_hr:.3f}")
    out_path = OUT_DIR / "mbti_per_persona.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved {len(table)} rows -> {out_path}")


if __name__ == "__main__":
    main()
