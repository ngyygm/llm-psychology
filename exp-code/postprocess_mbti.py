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

from run_mbti_experiment import migrate_state

CSV_FIELDS = r.CSV_FIELDS


def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _try_recover_from_full_response(full) -> str:
    """Extract candidate text from a full_response blob for re-parsing."""
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
    return candidate_text


def recover_sample(sample: dict, response_format: str, keyed: str) -> bool:
    """Try to recover a parse_failed sample from its full_response. Mutates in place."""
    if not sample.get("parse_failed"):
        return False
    candidate_text = _try_recover_from_full_response(sample.get("full_response"))
    if not candidate_text:
        return False
    parsed, used = r.parse_response(candidate_text, response_format)
    if parsed is None:
        return False
    sample["parsed_value"] = parsed
    sample["scored_value"] = r.apply_reverse_scoring(used, keyed, response_format)
    sample["parse_failed"] = False
    sample["recovery"] = {"source": "reasoning_content", "recovered_value": parsed}
    return True


def recover_record(rec: dict) -> int:
    """Try to recover parse_failed samples in a v2 record. Returns count recovered."""
    if "samples" in rec:
        n = 0
        for sample in rec["samples"]:
            if recover_sample(sample, rec["response_format"], rec["keyed"]):
                n += 1
        return n
    # v1 flat record fallback
    if not rec.get("parse_failed"):
        return 0
    candidate_text = _try_recover_from_full_response(rec.get("full_response"))
    if not candidate_text:
        return 0
    parsed, used = r.parse_response(candidate_text, rec["response_format"])
    if parsed is None:
        return 0
    rec["parsed_value"] = parsed
    rec["scored_value"] = r.apply_reverse_scoring(used, rec["keyed"], rec["response_format"])
    rec["parse_failed"] = False
    rec["recovery"] = {"source": "reasoning_content", "recovered_value": parsed}
    return 1


def rebuild_domain_scores(records: list[dict]) -> dict:
    return r._aggregate_domain_scores(records)


def process_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        state = json.load(f)

    migrate_state(state)

    n_recovered = 0
    n_pf_before = 0
    for persona, blk in state.get("results_by_persona", {}).items():
        for rec in blk.get("responses", []):
            if "samples" in rec:
                for s in rec["samples"]:
                    if s.get("parse_failed"):
                        n_pf_before += 1
            elif rec.get("parse_failed"):
                n_pf_before += 1
            recovered = recover_record(rec)
            n_recovered += recovered
        blk["domain_scores"] = rebuild_domain_scores(blk["responses"])

    n_pf_after = n_pf_before - n_recovered

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
