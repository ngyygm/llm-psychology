#!/usr/bin/env python3
"""Verify the three rebuttal analyses directly from the raw data.

1. Table 4  — per-model forward/reverse trait-endorsement rates and Δ, by instrument (Default).
2. 17-domain α-vs-reverse-%  — Cronbach's α (paper's raw method) per (scale,domain), averaged
   across models, correlated with reverse-item percentage.
3. k=1..5 convergence  — α (Spearman ρ with k=5 across models), PIR, and EFA factor structure.

Reuses load_all / extract / cronbach_alpha from acquiescence_correction_analysis so the α method
matches the paper exactly. All numbers written to results/ and printed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from acquiescence_correction_analysis import load_all, extract, cronbach_alpha

RESULTS_DIR = Path(__file__).resolve().parent / "results"
IPIP, SD3, ZKPQ, EPQR = "IPIP-NEO-120", "SD3", "ZKPQ-50-CC", "EPQR-A"
SCALES = [IPIP, SD3, ZKPQ, EPQR]
IPIP5 = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
KS = [1, 2, 3, 4, 5]


def _mean_first_k(r, field, k):
    vals = [s[field] for s in r["samples"][:k] if s.get(field) is not None]
    return sum(vals) / len(vals) if vals else None


def extract_k(model_data, persona, k):
    """`extract` but averaging only the first k samples (for convergence)."""
    rows = []
    for r in model_data["results_by_persona"][persona]["responses"]:
        rows.append({"item_id": r["item_id"], "scale": r["scale"], "domain": r["domain"],
                     "keyed": r["keyed"], "response_format": r["response_format"],
                     "parsed_value": _mean_first_k(r, "parsed_value", k),
                     "scored_value": _mean_first_k(r, "scored_value", k)})
    df = pd.DataFrame(rows)
    for col in ("parsed_value", "scored_value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    null = df["parsed_value"].isna()
    for idx in df[null].index:
        row = df.loc[idx]
        same = df[(df.domain == row.domain) & (df.scale == row.scale) & df.parsed_value.notna()]
        if len(same):
            med = same.parsed_value.median()
            df.at[idx, "parsed_value"] = med
            df.at[idx, "scored_value"] = med if row.keyed == "+" else (6 - med if row.response_format == "likert_5" else med)
    return df


def agree_rate(df):
    """Raw agreement rate per item, matching the paper's definition (round2_fixes.py):
    likert: parsed>=3 (incl. neutral midpoint); binary True/Yes: parsed==1.
    Key-independent — Table 4 splits by keyed direction afterward."""
    t = pd.Series(np.nan, index=df.index)
    likert = df["response_format"] == "likert_5"
    t[likert] = (df.loc[likert, "parsed_value"] >= 3).astype(float)
    t[~likert] = (df.loc[~likert, "parsed_value"] == 1).astype(float)
    return t


def pir_overall(df):
    pis = []
    for (sc, dom), grp in df.groupby(["scale", "domain"]):
        fwd, rev = grp[grp["keyed"] == "+"], grp[grp["keyed"] == "-"]
        if len(fwd) == 0 or len(rev) == 0:
            continue
        rf = grp["response_format"].iloc[0]
        n_inc = n_tot = 0
        fv, rv = fwd["parsed_value"].values, rev["parsed_value"].values
        for a in fv:
            for b in rv:
                n_tot += 1
                if rf == "likert_5":
                    if (a >= 3 and b >= 3) or (a <= 3 and b <= 3):
                        n_inc += 1
                elif a == b:
                    n_inc += 1
        pis.append(n_inc / n_tot if n_tot else np.nan)
    return float(np.nanmean(pis))


def pca_eig(corr):
    return np.sort(np.linalg.eigvalsh(corr))[::-1]


def parallel_pa(n_obs, n_var, n_iter=200, seed=0):
    rng = np.random.default_rng(seed)
    ev = np.zeros(n_var)
    for _ in range(n_iter):
        ev = np.maximum(ev, np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n_obs, n_var)), rowvar=False)))[::-1])
    return ev


# ───────────────────────── 1. Table 4 ─────────────────────────
def table4(allr):
    rows = []
    for m, md in allr.items():
        df = extract(md, "Default")
        df["t"] = agree_rate(df)
        rec = {"model": m}
        for sc in SCALES:
            sdf = df[df["scale"] == sc]
            fr, rr = sdf[sdf["keyed"] == "+"]["t"].mean(), sdf[sdf["keyed"] == "-"]["t"].mean()
            rec[f"{sc}_Fwd"], rec[f"{sc}_Rev"], rec[f"{sc}_Δ"] = fr, rr, fr - rr
        rec["Overall_Fwd"] = df[df["keyed"] == "+"]["t"].mean()
        rec["Overall_Rev"] = df[df["keyed"] == "-"]["t"].mean()
        rec["Overall_Δ"] = rec["Overall_Fwd"] - rec["Overall_Rev"]
        rows.append(rec)
    tbl = pd.DataFrame(rows)
    means = tbl.drop(columns="model").mean().to_dict()
    print("\n=== TABLE 4 (per-model Fwd/Rev/Δ, Default; mean over 20 models) ===")
    for sc in SCALES + ["Overall"]:
        print(f"  {sc:14s} Fwd={means[f'{sc}_Fwd']:.2f}  Rev={means[f'{sc}_Rev']:.2f}  Δ={means[f'{sc}_Δ']:+.2f}")
    print("  spot-check vs rebuttal (Claude-Opus IPIP 0.67/0.29/+0.38; GPT-5.2; DeepSeek-V3.2):")
    for m in tbl["model"]:
        if any(s in m for s in ["Claude-Opus-4.6", "GPT_5.2", "DeepSeek-V3.2"]):
            r = tbl[tbl["model"] == m].iloc[0]
            print(f"    {m:20s} IPIP {r['IPIP-NEO-120_Fwd']:.2f}/{r['IPIP-NEO-120_Rev']:.2f}/{r['IPIP-NEO-120_Δ']:+.2f} | "
                  f"SD3 Δ={r['SD3_Δ']:+.2f} ZKPQ Δ={r['ZKPQ-50-CC_Δ']:+.2f} EPQR Δ={r['EPQR-A_Δ']:+.2f}")
    tbl.to_csv(RESULTS_DIR / "table4_fwd_rev_delta.csv", index=False)
    print(f"  saved results/table4_fwd_rev_delta.csv ({len(tbl)} models)")


# ───────────────────── 2. 17-domain α vs reverse-% ─────────────────────
def alpha_reverse_17(allr):
    df0 = extract(next(iter(allr.values())), "Default")
    g = df0.groupby(["scale", "domain"]).agg(
        n_items=("keyed", "size"), n_rev=("keyed", lambda s: (s == "-").sum())).reset_index()
    rows = []
    for _, gr in g.iterrows():
        sc, dom, n_items, n_rev = gr["scale"], gr["domain"], int(gr["n_items"]), int(gr["n_rev"])
        if n_items < 3:
            continue
        model_alphas = []
        for m, md in allr.items():
            stack = []
            for persona in md["results_by_persona"]:
                d = extract(md, persona)
                dd = d[(d["scale"] == sc) & (d["domain"] == dom)].sort_values("item_id")
                if len(dd) >= 3:
                    stack.append(dd["parsed_value"].values.astype(float))
            if len(stack) >= 3:
                a = cronbach_alpha(np.array(stack))
                if np.isfinite(a):
                    model_alphas.append(a)
        if model_alphas:
            rows.append({"scale": sc, "domain": dom, "n_items": n_items, "n_rev": n_rev,
                         "reverse_pct": n_rev / n_items, "alpha_mean": float(np.mean(model_alphas))})
    df = pd.DataFrame(rows)
    df["ipip"] = df["scale"] == IPIP

    def rep(sub, label):
        if len(sub) < 3:
            return
        rho, p = stats.spearmanr(sub["alpha_mean"], sub["reverse_pct"])
        print(f"  {label:36s} N={len(sub):2d}  ρ={rho:+.3f}  p={p:.4f}")

    print("\n=== 17-DOMAIN α vs reverse-item % ===")
    rep(df, "All domains")
    rep(df[df["alpha_mean"] > 0.02], "Non-degenerate (α>0.02)")
    rep(df[df["ipip"]], "IPIP-5 (original)")
    rep(df[(~df["ipip"]) & (df["alpha_mean"] > 0.02)], "Non-IPIP non-degenerate")
    print("  per-domain:")
    for _, r in df.sort_values("reverse_pct").iterrows():
        print(f"    {r['scale']:13s} {r['domain']:22s} rev%={r['reverse_pct']:.2f} α={r['alpha_mean']:.3f}")
    df.to_csv(RESULTS_DIR / "alpha_reverse_17domains.csv", index=False)
    print(f"  saved results/alpha_reverse_17domains.csv ({len(df)} domains)")


# ───────────────────── 3. k=1..5 convergence ─────────────────────
def k_convergence(allr):
    amk = {k: {d: {} for d in IPIP5} for k in KS}   # α per (k,domain,model)
    pmk = {k: {} for k in KS}                        # PIR per (k,model)
    for k in KS:
        for m, md in allr.items():
            stacks = {d: [] for d in IPIP5}
            pirs = []
            for persona in md["results_by_persona"]:
                df = extract_k(md, persona, k)
                for d in IPIP5:
                    dd = df[(df["scale"] == IPIP) & (df["domain"] == d)].sort_values("item_id")
                    if len(dd) >= 3:
                        stacks[d].append(dd["parsed_value"].values.astype(float))
                pirs.append(pir_overall(df))
            for d in IPIP5:
                if len(stacks[d]) >= 3:
                    amk[k][d][m] = cronbach_alpha(np.array(stacks[d]))
            pmk[k][m] = float(np.nanmean(pirs))

    print("\n=== k-CONVERGENCE: α (Spearman ρ with k=5 across 20 models) ===")
    arows = []
    for d in IPIP5:
        rec = {"domain": d}
        common = [m for m in amk[5][d] if m in amk[1][d]]
        a5 = [amk[5][d][m] for m in common]
        for k in KS:
            rec[f"k{k}"] = 1.0 if k == 5 else stats.spearmanr(a5, [amk[k][d][m] for m in common]).correlation
        arows.append(rec)
        print(f"  {d:18s} " + "  ".join(f"k{k}={rec[f'k{k}']:.3f}" for k in KS))
    pd.DataFrame(arows).to_csv(RESULTS_DIR / "k_convergence_alpha.csv", index=False)

    print("\n=== k-CONVERGENCE: PIR ===")
    common_p = list(pmk[5].keys())
    p5 = np.array([pmk[5][m] for m in common_p])
    prows = []
    for k in KS:
        pk = np.array([pmk[k][m] for m in common_p])
        rho = 1.0 if k == 5 else stats.spearmanr(p5, pk).correlation
        mae = float(np.mean(np.abs(p5 - pk)))
        prows.append({"k": k, "mean_PIR": float(np.mean(list(pmk[k].values()))), "rho_k5": rho, "MAE": mae})
        print(f"  k={k}  mean PIR={float(np.mean(list(pmk[k].values()))):.3f}  ρ(k,5)={rho:.3f}  MAE={mae:.3f}")
    pd.DataFrame(prows).to_csv(RESULTS_DIR / "k_convergence_pir.csv", index=False)

    print("\n=== k-CONVERGENCE: EFA factor structure ===")
    erows = []
    for k in [1, 3, 5]:
        rm = []
        for m, md in allr.items():
            for persona in md["results_by_persona"]:
                df = extract_k(md, persona, k)
                for (sc, dom), grp in df.groupby(["scale", "domain"]):
                    rm.append({"mp": (m, persona), "domain": f"{sc}|{dom}", "scored": grp["scored_value"].mean()})
        M = pd.DataFrame(rm).pivot_table(index="mp", columns="domain", values="scored").dropna()
        eigs = pca_eig(M.corr())
        n_kaiser = int((eigs > 1).sum())
        n_pa = int((eigs > parallel_pa(len(M), M.shape[1])).sum())
        erows.append({"k": k, "n_obs": len(M), "kaiser": n_kaiser, "pa": n_pa,
                      "top3": ", ".join(f"{e:.2f}" for e in eigs[:3])})
        print(f"  k={k}  n={len(M)}  Kaiser={n_kaiser}  PA={n_pa}  top3=[{erows[-1]['top3']}]")
    pd.DataFrame(erows).to_csv(RESULTS_DIR / "k_convergence_efa.csv", index=False)
    print("  saved results/k_convergence_{alpha,pir,efa}.csv")


def main():
    allr = load_all()
    print(f"models loaded: {len(allr)}")
    table4(allr)
    alpha_reverse_17(allr)
    k_convergence(allr)


if __name__ == "__main__":
    main()
