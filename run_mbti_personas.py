#!/usr/bin/env python3
"""
Phase 4: MBTI Persona Steering Study.

Tests whether LLMs role-playing the 16 MBTI personality types reproduce the
covariance structure that humans of those types would show, or only adjust
mean scores while leaving the structure fixed.

Design:
  9 frontier models * 16 MBTI types * 61 items * 3 seeds = ~26k API calls.

Output:
  results/mbti_persona/study_mbti_<timestamp>.json — one record per
  (model, persona, seed) with item-level ratings and computed dimension scores.

Usage:
  python3 run_mbti_personas.py                     # full run, all 9 models
  python3 run_mbti_personas.py --pilot              # 2 models * 4 types * 1 seed
  python3 run_mbti_personas.py --model "GPT 5.4"  # one model only
  python3 run_mbti_personas.py --resume             # skip already-completed records
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Reuse the core experimental machinery from the main runner.
from run_model_experiments import (
    ALL_MODELS,
    BFI_44_ITEMS,
    BFI_REVERSE,
    HEXACO_H_ITEMS,
    SCHWARTZ_VALUES_ITEMS,
    COGNITIVE_STYLE_ITEMS,
    CULTURAL_DIMENSIONS_ITEMS,
    TEMPERATURE,
    parse_rating,
    query_model,
)

OUTPUT_DIR = Path("results/mbti_persona")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERSONA_SEEDS = [0, 1, 42]
N_WORKERS = 5

# 9 mainstream frontier model families (all via the unified LLM gateway).
# Ordered so that fast non-thinking models run first (DeepSeek-V3.2 is the
# only remaining non-thinking model in this list at submission time).
# The disabled entries (Doubao-Seed-1.6 and GPT 5.4) returned only 404
# through the unified LLM gateway during the data-collection window and are
# reported separately in Appendix L.
MBTI_MODELS = [
    "GPT 5.2",                  # OpenAI
    "Claude-Opus-4.6",           # Anthropic
    "Claude-Sonnet-4.6",         # Anthropic
    "Gemini 3-Pro-Preview",      # Google
    "Qwen3-235B-A22B",           # Alibaba
    "DeepSeek-V3.2",             # DeepSeek (fast, no-thinking)
    "GLM-4.7",                   # Zhipu (thinking)
    "MiniMax-M2.7",              # MiniMax (thinking)
    "Kimi-K2.5",                 # Moonshot (thinking)
]

# Models that emit lengthy reasoning_content before the final answer, so they
# need a larger generation budget to finish producing the Likert digit.
_HIGH_MAX_TOKENS_MODELS = {"Kimi-K2.5", "GLM-4.7", "MiniMax-M2.7"}


# ============== MBTI PERSONA CARDS ==============
# Sourced from McCrae & Costa (1989), Furnham (1996), Myers et al. (1998).
# See docs/mbti_persona_cards.md for full annotation and references.

MBTI_TYPES = {
    "INTJ": {
        "nickname": "The Architect",
        "expansion": "Introverted, Intuitive, Thinking, Judging",
        "stack": "Ni-Te-Fi-Se",
        "tendencies": "strategic, independent, long-horizon planning, privately intense, reserved",
        "strengths": "systems thinking, decisive execution, high standards",
        "challenges": "dismissive of inefficient processes, emotionally guarded",
        "big5_pred": {"O": +1, "C": +1, "E": -1, "A": -1, "N_FFM": 0},
    },
    "INTP": {
        "nickname": "The Logician",
        "expansion": "Introverted, Intuitive, Thinking, Perceiving",
        "stack": "Ti-Ne-Si-Fe",
        "tendencies": "curious, skeptical, precise, absent-minded about practical matters",
        "strengths": "conceptual originality, finding contradictions, tolerance of ambiguity",
        "challenges": "procrastination, blunt social register, difficulty committing to a single model",
        "big5_pred": {"O": +1, "C": -1, "E": -1, "A": -1, "N_FFM": 0},
    },
    "ENTJ": {
        "nickname": "The Commander",
        "expansion": "Extraverted, Intuitive, Thinking, Judging",
        "stack": "Te-Ni-Se-Fi",
        "tendencies": "assertive, goal-driven, direct, impatient with inefficiency",
        "strengths": "leadership, decisiveness, long-range planning paired with action",
        "challenges": "domineering, low tolerance for emotional processing",
        "big5_pred": {"O": +1, "C": +1, "E": +1, "A": -1, "N_FFM": 0},
    },
    "ENTP": {
        "nickname": "The Debater",
        "expansion": "Extraverted, Intuitive, Thinking, Perceiving",
        "stack": "Ne-Ti-Fe-Si",
        "tendencies": "inventive, argumentative, restless, playfully contrarian",
        "strengths": "ideation, rhetorical agility, seeing non-obvious connections",
        "challenges": "scattered follow-through, antagonising through sport debate",
        "big5_pred": {"O": +1, "C": -1, "E": +1, "A": 0, "N_FFM": 0},
    },
    "INFJ": {
        "nickname": "The Advocate",
        "expansion": "Introverted, Intuitive, Feeling, Judging",
        "stack": "Ni-Fe-Ti-Se",
        "tendencies": "insightful, private, principled, emotionally attuned",
        "strengths": "empathy at depth, long-term meaning-making, written articulation",
        "challenges": "perfectionism, burnout from absorbing others' affect, conflict avoidance",
        "big5_pred": {"O": +1, "C": +1, "E": -1, "A": +1, "N_FFM": 0},
    },
    "INFP": {
        "nickname": "The Mediator",
        "expansion": "Introverted, Intuitive, Feeling, Perceiving",
        "stack": "Fi-Ne-Si-Te",
        "tendencies": "idealistic, introspective, gentle, creatively imaginative",
        "strengths": "values clarity, artistic expression, empathy for outliers",
        "challenges": "avoidance of harsh reality, self-criticism, difficulty with logistics",
        "big5_pred": {"O": +1, "C": -1, "E": -1, "A": +1, "N_FFM": 0},
    },
    "ENFJ": {
        "nickname": "The Protagonist",
        "expansion": "Extraverted, Intuitive, Feeling, Judging",
        "stack": "Fe-Ni-Se-Ti",
        "tendencies": "charismatic, encouraging, responsible, attuned to social climate",
        "strengths": "motivating others, mediating conflict, developing people",
        "challenges": "over-identifying with others' needs, avoiding necessary criticism",
        "big5_pred": {"O": +1, "C": +1, "E": +1, "A": +1, "N_FFM": 0},
    },
    "ENFP": {
        "nickname": "The Campaigner",
        "expansion": "Extraverted, Intuitive, Feeling, Perceiving",
        "stack": "Ne-Fi-Te-Si",
        "tendencies": "expressive, spontaneous, curious, emotionally generous",
        "strengths": "inspiration, rapid connection, integrating disparate ideas",
        "challenges": "difficulty with routine, emotional volatility, loss of interest when novelty fades",
        "big5_pred": {"O": +1, "C": -1, "E": +1, "A": +1, "N_FFM": 0},
    },
    "ISTJ": {
        "nickname": "The Logistician",
        "expansion": "Introverted, Sensing, Thinking, Judging",
        "stack": "Si-Te-Fi-Ne",
        "tendencies": "dutiful, meticulous, reserved, dependable",
        "strengths": "reliability, attention to detail, institutional memory",
        "challenges": "resistance to change, rigidity with novel problems",
        "big5_pred": {"O": -1, "C": +1, "E": -1, "A": 0, "N_FFM": 0},
    },
    "ISFJ": {
        "nickname": "The Defender",
        "expansion": "Introverted, Sensing, Feeling, Judging",
        "stack": "Si-Fe-Ti-Ne",
        "tendencies": "warm, dependable, modest, detail-aware",
        "strengths": "steady support, organisational competence, loyalty",
        "challenges": "self-sacrifice to the point of resentment, conflict avoidance",
        "big5_pred": {"O": -1, "C": +1, "E": -1, "A": +1, "N_FFM": 0},
    },
    "ESTJ": {
        "nickname": "The Executive",
        "expansion": "Extraverted, Sensing, Thinking, Judging",
        "stack": "Te-Si-Ne-Fi",
        "tendencies": "decisive, direct, structured, status-aware",
        "strengths": "operational discipline, clear delegation, on-track delivery",
        "challenges": "impatience with ambiguity, blunt register, rigidity about the right way",
        "big5_pred": {"O": -1, "C": +1, "E": +1, "A": 0, "N_FFM": 0},
    },
    "ESFJ": {
        "nickname": "The Consul",
        "expansion": "Extraverted, Sensing, Feeling, Judging",
        "stack": "Fe-Si-Ne-Ti",
        "tendencies": "sociable, loyal, traditional, helpful",
        "strengths": "social fluency, practical caretaking, community organisation",
        "challenges": "need for approval, difficulty with criticism",
        "big5_pred": {"O": -1, "C": +1, "E": +1, "A": +1, "N_FFM": 0},
    },
    "ISTP": {
        "nickname": "The Virtuoso",
        "expansion": "Introverted, Sensing, Thinking, Perceiving",
        "stack": "Ti-Se-Ni-Fe",
        "tendencies": "cool, mechanical, laconic, opportunistic",
        "strengths": "crisis response, spatial and tactical problem-solving",
        "challenges": "emotional distance, reluctance to commit long-term",
        "big5_pred": {"O": 0, "C": -1, "E": -1, "A": -1, "N_FFM": 0},
    },
    "ISFP": {
        "nickname": "The Adventurer",
        "expansion": "Introverted, Sensing, Feeling, Perceiving",
        "stack": "Fi-Se-Ni-Te",
        "tendencies": "gentle, artistic, quiet, independent",
        "strengths": "aesthetic sensitivity, compassion, flexibility",
        "challenges": "avoidance of long-range planning, withdrawal under pressure",
        "big5_pred": {"O": +1, "C": -1, "E": -1, "A": +1, "N_FFM": 0},
    },
    "ESTP": {
        "nickname": "The Entrepreneur",
        "expansion": "Extraverted, Sensing, Thinking, Perceiving",
        "stack": "Se-Ti-Fe-Ni",
        "tendencies": "bold, pragmatic, sociable, risk-tolerant",
        "strengths": "improvisation, persuasion, physical competence",
        "challenges": "short-termism, boredom-driven risk-taking, disregard for rules",
        "big5_pred": {"O": 0, "C": -1, "E": +1, "A": 0, "N_FFM": 0},
    },
    "ESFP": {
        "nickname": "The Entertainer",
        "expansion": "Extraverted, Sensing, Feeling, Perceiving",
        "stack": "Se-Fi-Te-Ni",
        "tendencies": "expressive, spontaneous, playful, people-loving",
        "strengths": "social spark, adaptability, present-moment engagement",
        "challenges": "difficulty with long-term planning, emotional reactivity",
        "big5_pred": {"O": 0, "C": -1, "E": +1, "A": +1, "N_FFM": 0},
    },
}


def build_persona_prompt(mbti_type: str, item_text: str) -> str:
    """Compose the persona-conditioned Likert prompt for a single item."""
    p = MBTI_TYPES[mbti_type]
    return (
        f"You are role-playing a human with the {mbti_type} ({p['nickname']}) personality type.\n"
        f"{mbti_type} stands for {p['expansion']}.\n\n"
        f"Cognitive function stack: {p['stack']}.\n"
        f"Behavioral tendencies: {p['tendencies']}.\n"
        f"Strengths: {p['strengths']}.\n"
        f"Common challenges: {p['challenges']}.\n\n"
        f"Answer the following questionnaire as this person would. Stay in character.\n"
        f"Respond authentically based on how a {mbti_type} individual would genuinely feel,\n"
        f"not what is socially expected.\n\n"
        f"Statement: \"{item_text}\"\n\n"
        f"Respond with a single number from 1 to 5:\n"
        f"1 = Strongly Disagree\n"
        f"2 = Disagree\n"
        f"3 = Neutral\n"
        f"4 = Agree\n"
        f"5 = Strongly Agree\n"
        f"Your response:"
    )


def build_item_queries():
    """Return a flat list of (dimension, idx, text, reverse) item descriptors."""
    out = []
    for trait, items in BFI_44_ITEMS.items():
        for idx, item in enumerate(items):
            out.append({
                "dim": f"bfi_{trait}", "idx": idx, "text": item,
                "reverse": trait in BFI_REVERSE and (idx + 1) in BFI_REVERSE[trait],
            })
    for idx, item in enumerate(HEXACO_H_ITEMS):
        out.append({"dim": "hexaco_h", "idx": idx, "text": item, "reverse": False})
    for idx, item in enumerate(SCHWARTZ_VALUES_ITEMS):
        out.append({"dim": "schwartz_values", "idx": idx, "text": item, "reverse": False})
    for idx, item in enumerate(COGNITIVE_STYLE_ITEMS):
        out.append({"dim": "cognitive_style", "idx": idx, "text": item, "reverse": False})
    for idx, item in enumerate(CULTURAL_DIMENSIONS_ITEMS):
        out.append({"dim": "cultural_dimensions", "idx": idx, "text": item, "reverse": False})
    return out


def compute_scores(items: dict) -> dict:
    """Apply reverse-scoring and aggregate items into 9 dimension scores.

    Uses the corrected 0-indexed BFI_REVERSE table imported from the main
    runner. See docs/CRITICAL_BUG_REPORT.md for context.
    """
    for trait in BFI_REVERSE:
        key = f"bfi_{trait}"
        if key in items:
            for rev_idx in BFI_REVERSE[trait]:
                if 0 <= rev_idx < len(items[key]):
                    items[key][rev_idx] = 6 - items[key][rev_idx]
    bfi = {t: round(sum(items[f"bfi_{t}"]) / len(items[f"bfi_{t}"]), 4) for t in BFI_44_ITEMS}
    hexaco = round(6 - sum(items["hexaco_h"]) / len(items["hexaco_h"]), 4)
    sv = items["schwartz_values"]
    collectivism = round((sv[1] + sv[2] - sv[0] - sv[3]) / 4 + 2.5, 4)
    cs = items["cognitive_style"]
    intuition = round((cs[0] + (6 - cs[1]) + (6 - cs[2]) + cs[3]) / 4, 4)
    cd = items["cultural_dimensions"]
    ua = round((cd[0] + (6 - cd[1]) + cd[2] + (6 - cd[3])) / 4, 4)
    return {
        "bfi.extraversion":     bfi["extraversion"],
        "bfi.agreeableness":    bfi["agreeableness"],
        "bfi.conscientiousness": bfi["conscientiousness"],
        "bfi.neuroticism":      bfi["neuroticism"],
        "bfi.openness":         bfi["openness"],
        "hexaco_h":             hexaco,
        "collectivism":         collectivism,
        "intuition":            intuition,
        "uncertainty_avoidance": ua,
    }


def run_persona_session(model_id: str, mbti_type: str, seed: int) -> dict:
    """Run all 61 items in parallel for one (model, persona, seed) triple."""
    item_queries = build_item_queries()
    max_tokens = 2000 if model_id in _HIGH_MAX_TOKENS_MODELS else 200

    def _query(iq):
        prompt = build_persona_prompt(mbti_type, iq["text"])
        resp = query_model(model_id, prompt, TEMPERATURE, seed, max_tokens=max_tokens)
        return iq["dim"], iq["idx"], parse_rating(resp), resp[:200]

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_query, iq): iq for iq in item_queries}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                iq = futures[fut]
                print(f"      ! item {iq['dim']}[{iq['idx']}] failed: {e}", flush=True)
                results.append((iq["dim"], iq["idx"], 3, f"ERROR: {e}"))

    items = {}
    raw = {}
    for dim, idx, rating, r in results:
        items.setdefault(dim, [None] * 100)
        raw.setdefault(dim, [None] * 100)
        items[dim][idx] = rating
        raw[dim][idx] = r
    for dim in items:
        last = max(i for i, x in enumerate(items[dim]) if x is not None) + 1
        items[dim] = items[dim][:last]
        raw[dim] = raw[dim][:last]

    scores = compute_scores(items)
    return {"items": items, "raw_responses": raw, "scores": scores}


def load_completed(output_files) -> set:
    """Build the set of (model_id, mbti_type, seed) triples already on disk."""
    done = set()
    for f in output_files:
        try:
            with open(f) as fh:
                for r in json.load(fh):
                    done.add((r["model_id"], r["mbti_type"], r["seed"]))
        except Exception:
            pass
    return done


def main():
    parser = argparse.ArgumentParser(description="MBTI persona steering experiment.")
    parser.add_argument("--pilot", action="store_true",
                        help="Run pilot: 2 models * 4 types * 1 seed.")
    parser.add_argument("--model", type=str, default=None,
                        help="Run a single model (use the gateway-expected name).")
    parser.add_argument("--types", type=str, default=None,
                        help="Comma-separated MBTI types (default: all 16).")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: 0,1,42).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip records already saved on disk.")
    args = parser.parse_args()

    if args.pilot:
        models = ["GPT 5.4", "Claude-Opus-4.6"]
        types = ["INTJ", "ESFP", "INTP", "ESFJ"]
        seeds = [0]
    else:
        models = [args.model] if args.model else MBTI_MODELS
        types = args.types.split(",") if args.types else list(MBTI_TYPES)
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else PERSONA_SEEDS

    total = len(models) * len(types) * len(seeds)
    print("=" * 80)
    print(f"MBTI PERSONA STUDY: {len(models)} models * {len(types)} types * {len(seeds)} seeds")
    print(f"  = {total} (model, persona, seed) sessions ({total * 61} item calls)")
    print("=" * 80)

    completed = load_completed(OUTPUT_DIR.glob("study_mbti_*.json")) if args.resume else set()
    if completed:
        print(f"Resume: skipping {len(completed)} sessions already on disk.")

    all_results = []
    session_idx = 0
    for model_id in models:
        meta = ALL_MODELS.get(model_id, {})
        if not meta:
            print(f"WARN: {model_id} not in ALL_MODELS; will use default routing.")
        for mbti_type in types:
            for seed in seeds:
                session_idx += 1
                key = (model_id, mbti_type, seed)
                if key in completed:
                    print(f"[{session_idx}/{total}] SKIP {model_id} | {mbti_type} | seed={seed}")
                    continue
                t0 = time.time()
                print(f"[{session_idx}/{total}] {model_id} | {mbti_type} | seed={seed} ...",
                      end=" ", flush=True)
                try:
                    out = run_persona_session(model_id, mbti_type, seed)
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue
                elapsed = time.time() - t0
                rec = {
                    "model_id": model_id,
                    "model": meta.get("model_id", ""),
                    "arch": meta.get("arch", ""),
                    "study": "mbti_persona",
                    "mbti_type": mbti_type,
                    "seed": seed,
                    "thinking_mode": "chat",
                    "timestamp": datetime.now().isoformat(),
                    "items": out["items"],
                    "raw_responses": out["raw_responses"],
                    "elapsed_sec": round(elapsed, 1),
                    **out["scores"],
                }
                all_results.append(rec)
                s = out["scores"]
                print(
                    f"E={s['bfi.extraversion']:.2f} A={s['bfi.agreeableness']:.2f} "
                    f"C={s['bfi.conscientiousness']:.2f} N={s['bfi.neuroticism']:.2f} "
                    f"O={s['bfi.openness']:.2f} ({elapsed:.1f}s)"
                )
                if len(all_results) % 10 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = OUTPUT_DIR / f"study_mbti_{timestamp}.json"
                    with open(out_path, "w") as fh:
                        json.dump(all_results, fh, indent=2)
                    print(f"    -> intermediate save ({len(all_results)} records) {out_path.name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"study_mbti_{timestamp}.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nFinal: {len(all_results)} records -> {out_path}")


if __name__ == "__main__":
    main()
