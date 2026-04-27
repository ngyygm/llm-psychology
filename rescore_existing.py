#!/usr/bin/env python3
"""
Rescore all existing experiment records using the corrected BFI reverse-scoring.

The original `run_model_experiments.py` had an off-by-one in BFI reverse-scoring
that targeted positive-keyed items instead of negative-keyed ones. Saved item
arrays therefore have positive items flipped (6 - x) and negative items left raw.
This script:
  1. reads every result JSON in results/vendor_exp and results/mbti_persona,
  2. reconstructs raw 1-5 item ratings by undoing the buggy reversal,
  3. applies the corrected reverse-scoring (BFI_REVERSE in run_model_experiments),
  4. recomputes the 9 dimension scores,
  5. writes mirrored files into <input_dir>/corrected/ with a `_scoring` field
     marking the file version.

We do NOT modify the original JSONs. To switch downstream analysis to corrected
data, point it at results/vendor_exp/corrected/.
"""

import json
import shutil
from pathlib import Path

from run_model_experiments import (
    BFI_44_ITEMS,
    BFI_REVERSE,
    HEXACO_H_ITEMS,
    SCHWARTZ_VALUES_ITEMS,
    COGNITIVE_STYLE_ITEMS,
    CULTURAL_DIMENSIONS_ITEMS,
)

# Recreate the buggy reverse positions (rev_idx - 1 in old code).
BUGGY_REVERSE = {
    trait: [(i - 1) % len(BFI_44_ITEMS[trait]) if (i - 1) >= 0 else (i - 1)
            for i in lst]
    for trait, lst in {
        "extraversion":      [1, 3, 5, 7],
        "agreeableness":     [3, 4, 6, 7],
        "conscientiousness": [1, 3, 5, 7],
        "neuroticism":       [0, 6],
        "openness":          [1, 3, 5],
    }.items()
}
# For neuroticism the buggy list had [0, 6] -> rev_idx-1 = -1 (wraps to 7) and 5.
BUGGY_REVERSE["neuroticism"] = [7, 5]

VENDOR_DIR = Path("results/vendor_exp")
MBTI_DIR = Path("results/mbti_persona")


def undo_bug(items: dict) -> dict:
    """Recover raw 1-5 ratings from a buggy-scored items dict."""
    out = {k: list(v) for k, v in items.items()}
    for trait, positions in BUGGY_REVERSE.items():
        key = f"bfi_{trait}"
        if key not in out:
            continue
        for pos in positions:
            if -len(out[key]) <= pos < len(out[key]):
                out[key][pos] = 6 - out[key][pos]
    return out


def correct_score(raw_items: dict) -> dict:
    """Apply correct reverse-scoring and compute 9 dimension scores."""
    items = {k: list(v) for k, v in raw_items.items()}
    for trait, positions in BFI_REVERSE.items():
        key = f"bfi_{trait}"
        if key not in items:
            continue
        for pos in positions:
            if 0 <= pos < len(items[key]):
                items[key][pos] = 6 - items[key][pos]
    bfi = {t: round(sum(items[f"bfi_{t}"]) / len(items[f"bfi_{t}"]), 4)
           for t in BFI_44_ITEMS}
    hexaco = (round(6 - sum(items["hexaco_h"]) / len(items["hexaco_h"]), 4)
              if "hexaco_h" in items else None)
    sv = items.get("schwartz_values", [])
    collectivism = (round((sv[1] + sv[2] - sv[0] - sv[3]) / 4 + 2.5, 4)
                    if len(sv) >= 4 else None)
    cs = items.get("cognitive_style", [])
    intuition = (round((cs[0] + (6 - cs[1]) + (6 - cs[2]) + cs[3]) / 4, 4)
                 if len(cs) >= 4 else None)
    cd = items.get("cultural_dimensions", [])
    ua = (round((cd[0] + (6 - cd[1]) + cd[2] + (6 - cd[3])) / 4, 4)
          if len(cd) >= 4 else None)
    return {
        "items_corrected": items,
        "bfi.extraversion": bfi["extraversion"],
        "bfi.agreeableness": bfi["agreeableness"],
        "bfi.conscientiousness": bfi["conscientiousness"],
        "bfi.neuroticism": bfi["neuroticism"],
        "bfi.openness": bfi["openness"],
        "hexaco_h": hexaco,
        "collectivism": collectivism,
        "intuition": intuition,
        "uncertainty_avoidance": ua,
    }


def rescore_record(rec: dict) -> dict:
    items = rec.get("items")
    if not items:
        return rec
    # MBTI data was collected with the corrected runner, so its saved items
    # already reflect the correct reverse-scoring. Skip the undo-bug step to
    # avoid double-reversing; just recompute dimension scores via correct_score.
    already_corrected = (rec.get("study") == "mbti_persona"
                         or rec.get("_scoring") == "corrected_v1")
    raw = items if already_corrected else undo_bug(items)
    # correct_score re-applies the correct reverse, but if the data was already
    # corrected we have to skip its reverse step too. Simplest approach: for
    # already-corrected records, just recompute aggregates from the stored items
    # and the already-correct state.
    if already_corrected:
        new_scores = _aggregate_without_reverse(items)
    else:
        new_scores = correct_score(raw)
    out = dict(rec)
    out["items"] = new_scores.pop("items_corrected")
    out.update(new_scores)
    out["_scoring"] = "corrected_v1"
    return out


def _aggregate_without_reverse(items: dict) -> dict:
    """Aggregate dimension scores assuming `items` are already correctly
    reverse-scored (no further flipping)."""
    bfi = {t: round(sum(items[f"bfi_{t}"]) / len(items[f"bfi_{t}"]), 4)
           for t in BFI_44_ITEMS if f"bfi_{t}" in items}
    hexaco = (round(6 - sum(items["hexaco_h"]) / len(items["hexaco_h"]), 4)
              if "hexaco_h" in items else None)
    sv = items.get("schwartz_values", [])
    collectivism = (round((sv[1] + sv[2] - sv[0] - sv[3]) / 4 + 2.5, 4)
                    if len(sv) >= 4 else None)
    cs = items.get("cognitive_style", [])
    intuition = (round((cs[0] + (6 - cs[1]) + (6 - cs[2]) + cs[3]) / 4, 4)
                 if len(cs) >= 4 else None)
    cd = items.get("cultural_dimensions", [])
    ua = (round((cd[0] + (6 - cd[1]) + cd[2] + (6 - cd[3])) / 4, 4)
          if len(cd) >= 4 else None)
    return {
        "items_corrected": items,
        "bfi.extraversion": bfi.get("extraversion"),
        "bfi.agreeableness": bfi.get("agreeableness"),
        "bfi.conscientiousness": bfi.get("conscientiousness"),
        "bfi.neuroticism": bfi.get("neuroticism"),
        "bfi.openness": bfi.get("openness"),
        "hexaco_h": hexaco,
        "collectivism": collectivism,
        "intuition": intuition,
        "uncertainty_avoidance": ua,
    }


def rescore_directory(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    n_files = n_records = 0
    for f in sorted(src.glob("*.json")):
        if "checkpoint" in f.name:
            shutil.copy2(f, dst / f.name)
            continue
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
            continue
        if not isinstance(data, list):
            shutil.copy2(f, dst / f.name)
            continue
        rescored = [rescore_record(r) for r in data]
        out_path = dst / f.name
        out_path.write_text(json.dumps(rescored, indent=2))
        n_files += 1
        n_records += len(rescored)
    print(f"  -> rescored {n_records} records across {n_files} files into {dst}")


def main():
    print("Rescoring results/vendor_exp/ ...")
    rescore_directory(VENDOR_DIR, VENDOR_DIR / "corrected")
    if MBTI_DIR.exists() and any(MBTI_DIR.glob("*.json")):
        print("\nRescoring results/mbti_persona/ ...")
        rescore_directory(MBTI_DIR, MBTI_DIR / "corrected")


if __name__ == "__main__":
    main()
