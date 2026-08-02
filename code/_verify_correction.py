#!/usr/bin/env python3
"""Verify scored_value reverse-scoring + domain-level EFA with/without correction."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from acquiescence_correction_analysis import load_all, extract, EXCLUDE

DATA_DIR = Path(__file__).resolve().parent / "data"
DOMAIN_ORDER = ["Neuroticism", "Extraversion", "Openness", "Agreeableness",
                "Conscientiousness", "Sociability", "Activity",
                "ImpulsiveSS", "Aggression-Hostility", "Psychoticism",
                "Machiavellianism", "Narcissism", "Psychopathy",
                "Aggression", "Lie", "Extraversion-s", "Anxiety"]


def pca_eigvars(corr):
    eigs = np.linalg.eigvalsh(corr.values)
    return np.sort(eigs)[::-1]


def parallel_analysis(n_obs, n_var, n_iter=200, seed=0):
    rng = np.random.default_rng(seed)
    ev = np.zeros(n_var)
    for _ in range(n_iter):
        X = rng.standard_normal((n_obs, n_var))
        c = np.corrcoef(X, rowvar=False)
        ev = np.maximum(ev, np.sort(np.linalg.eigvalsh(c))[::-1])
    return ev


def main():
    all_results = load_all()

    # ---- 1. sanity: scored_value reverse-scoring ----
    mdata = all_results["Claude-Opus-4.6"]
    df = extract(mdata, "Default")
    ip = df[df["scale"] == "IPIP-NEO-120"]
    rev = ip[ip["keyed"] == "-"]
    fwd = ip[ip["keyed"] == "+"]
    rev_check = np.allclose(rev["scored_value"].values, 6 - rev["parsed_value"].values)
    fwd_check = np.allclose(fwd["scored_value"].values, fwd["parsed_value"].values)
    print("=== scored_value sanity (IPIP, Claude-Opus Default) ===")
    print(f"  forward items: scored==parsed? {fwd_check}")
    print(f"  reverse items: scored==6-parsed? {rev_check}")
    print(f"  n forward={len(fwd)}, n reverse={len(rev)}")

    # ---- 2. domain-level EFA: scored vs acquiescence-corrected ----
    rows_s, rows_c = [], []
    for model_name, md in all_results.items():
        for persona in md["results_by_persona"]:
            d = extract(md, persona)
            # acquiescence index per respondent over ALL 120 IPIP likert items
            ip = d[d["scale"] == "IPIP-NEO-120"]
            mu = ip["parsed_value"].mean()
            for _, r in d.iterrows():
                raw = r["parsed_value"]
                cor = raw - mu
                cor_trait = cor if r["keyed"] == "+" else -cor
                rows_s.append({"model": model_name, "persona": persona,
                               "domain": f'{r["scale"]}|{r["domain"]}', "scored": r["scored_value"]})
                rows_c.append({"model": model_name, "persona": persona,
                               "domain": f'{r["scale"]}|{r["domain"]}', "scored": cor_trait})
    S = pd.DataFrame(rows_s).pivot_table(index=["model", "persona"], columns="domain", values="scored")
    C = pd.DataFrame(rows_c).pivot_table(index=["model", "persona"], columns="domain", values="scored")
    S = S.dropna(); C = C.dropna()
    print(f"\n=== domain-level EFA (n_obs={len(S)}, n_domains={S.shape[1]}) ===")
    for label, M in [("scored (paper)", S), ("acquiescence-corrected", C)]:
        corr = M.corr()
        eigs = pca_eigvars(corr)
        n_kaiser = int((eigs > 1).sum())
        pa = parallel_analysis(len(M), M.shape[1])
        n_pa = int((eigs > pa).sum())
        print(f"\n  [{label}]")
        print(f"    top eigenvalues: {np.round(eigs[:6], 2)}")
        print(f"    Kaiser factors (eig>1): {n_kaiser}")
        print(f"    Parallel-analysis factors: {n_pa}")


if __name__ == "__main__":
    main()
