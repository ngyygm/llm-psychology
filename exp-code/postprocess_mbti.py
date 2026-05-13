#!/usr/bin/env python3
"""
Post-processing for the MBTI battery run.

Two passes:
  1. Recover thinking-model parse failures.
     Some thinking-mode models emit their full reasoning trace into a
     separate ``reasoning_content`` channel and never finish writing the
     visible answer in ``content`` — leaving raw_response empty. We re-parse
     those records using the same last-occurrence heuristic on
     reasoning_content, then update parsed_value / scored_value /
     parse_failed in place. The original raw_response (empty string) and
     full_response (the complete API blob) are preserved for provenance.

  2. Rebuild domain_scores per persona × scale × domain (so any newly
     recovered values land in the aggregate) and regenerate
     ``results/summary.csv`` from scratch covering every per-model JSON
     in results/.

Run:
    python postprocess_mbti.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mbti_experiment as r  # reuse parse_response, apply_reverse_scoring


CSV_FIELDS = r.CSV_FIELDS


def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def recover_record(rec: dict) -> bool:
    """Try to recover a parse_failed record from full_response.reasoning_content
    (or any analogous channel). Mutates rec in place. Returns True if recovered.
    """
    if not rec.get("parse_failed"):
        return False
    full = rec.get("full_response")
    candidate_text = ""
    if isinstance(full, dict):
        candidate_text = full.get("reasoning_content") or ""
        if not candidate_text:
            parts = full.get("parts") or full.get("content") or []
            if isinstance(parts, list):
                bits: list[str] = []
                for p in parts:
                    if isinstance(p, dict):
                        for k in ("text", "thinking", "content"):
                            v = p.get(k)
                            if isinstance(v, str):
                                bits.append(v)
                candidate_text = "\n".join(bits)
    elif isinstance(full, list):
        bits = []
        for blk in full:
            if isinstance(blk, dict):
                for k in ("text", "thinking", "content"):
                    v = blk.get(k)
                    if isinstance(v, str):
                        bits.append(v)
        candidate_text = "\n".join(bits)

    if not candidate_text:
        return False

    parsed, used = r.parse_response(candidate_text, rec["response_format"])
    if parsed is None:
        return False
    rec["parsed_value"] = parsed
    rec["scored_value"] = r.apply_reverse_scoring(used, rec["keyed"], rec["response_format"])
    rec["parse_failed"] = False
    rec["recovery"] = {
        "source": "reasoning_content",
        "recovered_value": parsed,
    }
    return True


def rebuild_domain_scores(records: list[dict]) -> dict:
    return r._aggregate_domain_scores(records)


def process_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        state = json.load(f)

    n_recovered = 0
    n_pf_before = 0
    n_pf_after = 0
    for persona, blk in state.get("results_by_persona", {}).items():
        for rec in blk.get("responses", []):
            if rec.get("parse_failed"):
                n_pf_before += 1
                if recover_record(rec):
                    n_recovered += 1
                else:
                    n_pf_after += 1
        blk["domain_scores"] = rebuild_domain_scores(blk["responses"])

    state.setdefault("postprocessing", {})
    state["postprocessing"]["recovered_from_reasoning_content"] = n_recovered
    state["postprocessing"]["parse_failed_before"] = n_pf_before
    state["postprocessing"]["parse_failed_after"] = n_pf_after

    if n_recovered:
        _atomic_write_json(path, state)
    return {
        "model": state.get("model_name"),
        "n_pf_before": n_pf_before,
        "n_recovered": n_recovered,
        "n_pf_after": n_pf_after,
    }


def rebuild_summary_csv() -> int:
    rows = 0
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()
    for path in sorted(RESULTS_DIR.glob("exp_mbti_*.json")):
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if rows == 0:
                w.writeheader()
            for persona, blk in state["results_by_persona"].items():
                for _, dom in blk["domain_scores"].items():
                    w.writerow({
                        "model_name": state["model_name"],
                        "persona": persona,
                        "scale": dom["scale"],
                        "domain": dom["domain"],
                        "n_items": dom["n_items"],
                        "mean_score": dom["mean_score"],
                        "std_score": dom["std_score"],
                        "min_score": dom["min_score"],
                        "max_score": dom["max_score"],
                        "completion_status": state["completion_status"],
                    })
                    rows += 1
    return rows


def main() -> int:
    print("== Pass 1: recover parse_failed records via reasoning_content ==")
    summaries = []
    for path in sorted(RESULTS_DIR.glob("exp_mbti_*.json")):
        s = process_file(path)
        summaries.append(s)
        if s["n_pf_before"] or s["n_recovered"]:
            print(f"  {s['model']:<32}  pf_before={s['n_pf_before']:>4}  "
                  f"recovered={s['n_recovered']:>4}  pf_after={s['n_pf_after']:>4}")
    total_pf_before = sum(s["n_pf_before"] for s in summaries)
    total_recov = sum(s["n_recovered"] for s in summaries)
    total_pf_after = sum(s["n_pf_after"] for s in summaries)
    print(f"  TOTAL                       pf_before={total_pf_before:>4}  "
          f"recovered={total_recov:>4}  pf_after={total_pf_after:>4}")

    print()
    print(f"== Pass 2: rebuild {SUMMARY_CSV.name} ==")
    n = rebuild_summary_csv()
    print(f"  wrote {n} summary rows → {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
