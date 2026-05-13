#!/usr/bin/env python3
"""
MBTI Role-Play × 221-Item Personality Battery — OpenAI-compatible API runner.

For every (model × persona × item) triple:
  1. Build a system prompt that pins the model into one of 16 MBTI personas.
     A 17th "Default" persona is also run with NO system prompt at all,
     serving as the no-persona control condition.
  2. Ask the model to answer one battery item in the requested response format.
  3. Parse + reverse-score the answer.
  4. Stream the result into results/exp_mbti_{MODEL}.json incrementally.
  5. Emit a single domain-level summary CSV (results/summary.csv) covering
     every model.

Configure via environment variables:
    OPENAI_API_KEY    — API key (required)
    OPENAI_BASE_URL   — endpoint base; default https://api.openai.com/v1
                        Override for vLLM (http://localhost:8000/v1),
                        Together, Anyscale, Groq, etc.

Concurrency model:
  * Inside one model: 16 in-flight HTTP requests (sliding window via
    ThreadPoolExecutor.submit + as_completed).
  * Across models: 5 models running at the same time.

Failure handling:
  * Every request is retried up to MAX_RETRIES with RETRY_WAIT_SECONDS gap.
  * If a model emits >= MODEL_FATAL_THRESHOLD consecutive permanent failures,
    the model is aborted and an entry is appended to model_error_log.md.

Run modes:
    python run_mbti_experiment.py --test           # 1 model × 10 items smoke test
    python run_mbti_experiment.py --all            # all models in MODELS
    python run_mbti_experiment.py --model NAME     # one model, full 17 × 221
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# ============================================================================
#                              CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
BATTERY_PATH = REPO_ROOT / "data" / "items_battery.json"
MBTI_DOC_PATH = HERE / "mbti-doc.md"
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"
ERROR_LOG_PATH = HERE / "model_error_log.md"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
CHAT_URL = f"{OPENAI_BASE_URL}/chat/completions"

REQUEST_TIMEOUT = 120
MAX_RETRIES = 10
RETRY_WAIT_SECONDS = 3
MODEL_FATAL_THRESHOLD = 8  # consecutive permanent failures → abort the model
BATCH_SIZE = 16            # in-flight requests per model
PARALLEL_MODELS = 5        # how many models run simultaneously
TEMPERATURE = 0.7
MAX_TOKENS = 8192          # generous headroom for thinking-mode models

# Customize with the model ids exposed by your endpoint.
MODELS: list[str] = [
    "gpt-4o-mini",
    "gpt-4o",
]

MBTI_ORDER = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ",
]

# "Default" is a baseline control persona: no MBTI prompt at all, the model
# simply receives the questionnaire item directly. Useful as the "no-persona"
# anchor when comparing against the 16 MBTI conditions.
DEFAULT_PERSONA = "Default"
PERSONA_ORDER = [DEFAULT_PERSONA] + MBTI_ORDER

# ============================================================================
#                          LOGGING / ERROR LOG
# ============================================================================

def sanitize(name: str) -> str:
    """Filesystem-safe model name: keep alnum/dot/dash/underscore."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def setup_logger(model_name: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    safe = sanitize(model_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"mbti_{safe}_{ts}.log"
    logger = logging.getLogger(f"mbti.{safe}.{ts}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(f"[{model_name}] %(message)s"))
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.info("Log file: %s", log_path)
    return logger


_ERROR_LOG_LOCK = threading.Lock()


def append_error_log(model_name: str, error_info: str) -> None:
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with _ERROR_LOG_LOCK:
        new_file = not ERROR_LOG_PATH.exists()
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            if new_file:
                f.write("# Per-model fatal error log\n\n")
                f.write("Each entry records a model whose run was aborted because the API "
                        "returned permanent failures.\n\n")
            f.write(f"## `{model_name}` — {stamp}\n\n")
            f.write("```\n")
            f.write(error_info.strip() + "\n")
            f.write("```\n\n")

# ============================================================================
#                          BATTERY + MBTI LOADERS
# ============================================================================

def load_battery() -> dict:
    with open(BATTERY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_mbti() -> dict[str, str]:
    """Parse mbti-doc.md → { type: description }."""
    raw = MBTI_DOC_PATH.read_text(encoding="utf-8").strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", raw) if b.strip()]
    out: dict[str, str] = {}
    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        if not lines:
            continue
        head, *body = lines
        if head in MBTI_ORDER:
            out[head] = " ".join(body)
    missing = [t for t in MBTI_ORDER if t not in out]
    if missing:
        raise RuntimeError(f"MBTI doc missing types: {missing}")
    return out

# ============================================================================
#                              PROMPTING
# ============================================================================

def make_persona_prompt(persona: str, mbti_description: str | None) -> str:
    """Build the system prompt for a persona.

    For ``persona == "Default"`` we return an empty string, meaning no
    persona-shaping instructions are sent at all — the model just sees the
    raw questionnaire item. For any of the 16 MBTI types we render a
    role-play prompt anchored on the matching mbti-doc.md description.
    """
    if persona == DEFAULT_PERSONA:
        return ""
    assert mbti_description is not None, f"missing description for persona {persona!r}"
    return (
        f"You are role-playing as a person whose Myers-Briggs Type Indicator "
        f"(MBTI) personality is {persona}.\n\n"
        f"{persona} personality description:\n"
        f"{mbti_description}\n\n"
        f"You must answer ALL of the following personality questionnaire items "
        f"strictly in character as a {persona} person. Your answers must be "
        f"entirely consistent with the {persona} personality profile described "
        f"above. Stay in character at all times. Do not break character. Do not "
        f"reveal that you are an AI. Reply ONLY with the exact format the question "
        f"asks for (a single number 1-5, or True/False, or Yes/No) — no "
        f"explanations, no qualifications, no extra words."
    )


def make_likert_question(item_text: str) -> str:
    return (
        f"Please respond to the following statement as honestly as possible.\n"
        f"Statement: \"{item_text}\"\n"
        f"Respond with a single number from 1 to 5:\n"
        f"1 = Strongly Disagree\n"
        f"2 = Disagree\n"
        f"3 = Neutral\n"
        f"4 = Agree\n"
        f"5 = Strongly Agree\n"
        f"Your response:"
    )


def make_true_false_question(item_text: str) -> str:
    return (
        f"Please respond to the following statement as honestly as possible.\n"
        f"Statement: \"{item_text}\"\n"
        f"Respond with either True or False.\n"
        f"Your response:"
    )


def make_yes_no_question(item_text: str) -> str:
    return (
        f"Please respond to the following statement as honestly as possible.\n"
        f"Statement: \"{item_text}\"\n"
        f"Respond with either Yes or No.\n"
        f"Your response:"
    )


def build_user_question(item: dict) -> str:
    fmt = item["response_format"]
    if fmt == "true_false":
        return make_true_false_question(item["text"])
    if fmt == "yes_no":
        return make_yes_no_question(item["text"])
    return make_likert_question(item["text"])

# ============================================================================
#               PARSERS  (robust against thinking-model CoT)
#
# Thinking models emit long chain-of-thought before the final answer. The
# chain often contains stray digits ("1. Analyze the request...") and the
# words "yes"/"no"/"true"/"false" that are NOT the actual answer. The robust
# strategy is:
#   * For likert_5: find ALL [1-5] digits, return the LAST one.
#   * For yes/no, true/false: find ALL word-boundary matches, return the
#     LAST one (i.e. whichever appears nearest to the end of the response).
# Non-thinking models emit a single token, so first == last and the answer
# is unchanged. ``parse_*`` returns ``None`` only when no candidate is found.
# ============================================================================

def parse_rating(response: str) -> int | None:
    matches = re.findall(r"[1-5]", response or "")
    return int(matches[-1]) if matches else None


def parse_true_false(response: str) -> int | None:
    r = (response or "").lower()
    tf = list(re.finditer(r"\b(true|false)\b", r))
    if tf:
        return 1 if tf[-1].group(1) == "true" else 0
    if "true" in r and "false" not in r:
        return 1
    if "false" in r and "true" not in r:
        return 0
    if "true" in r and "false" in r:
        return 1 if r.rfind("true") > r.rfind("false") else 0
    return None


def parse_yes_no(response: str) -> int | None:
    r = (response or "").lower()
    yn = list(re.finditer(r"\b(yes|no)\b", r))
    if yn:
        return 1 if yn[-1].group(1) == "yes" else 0
    if "yes" in r and "no" not in r:
        return 1
    if "no" in r and "yes" not in r:
        return 0
    if "yes" in r and "no" in r:
        return 1 if r.rfind("yes") > r.rfind("no") else 0
    return None


def parse_response(response: str, response_format: str) -> tuple[int | None, int]:
    """Return (parsed_value, default_used_value). On parse failure we fall back
    to a neutral default (3 for Likert, 0 for binary) but flag the failure via
    the None component."""
    if response_format == "true_false":
        v = parse_true_false(response)
        return v, (v if v is not None else 0)
    if response_format == "yes_no":
        v = parse_yes_no(response)
        return v, (v if v is not None else 0)
    v = parse_rating(response)
    return v, (v if v is not None else 3)


def apply_reverse_scoring(raw: int, keyed: str, response_format: str) -> int:
    if keyed != "-":
        return raw
    if response_format == "likert_5":
        return 6 - raw
    return 1 - raw

# ============================================================================
#                      OPENAI-COMPATIBLE CHAT API
# ============================================================================

def call_chat(model: str, system: str, user: str) -> tuple[str, dict, Any]:
    """Single OpenAI-compatible Chat Completions request.

    Returns ``(text, telemetry, full_message_blob)`` where ``full_message_blob``
    is the entire assistant message returned by the API. This preserves any
    auxiliary channels such as ``reasoning_content`` (DeepSeek-R-style),
    ``tool_calls``, etc., for downstream provenance / post-processing.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running, e.g.:\n"
            "    export OPENAI_API_KEY=sk-..."
        )
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(CHAT_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    text = msg.get("content", "") or ""
    return text.strip(), {"http_status": r.status_code, "raw_keys": list(data.keys())}, msg


def call_with_retry(model: str, system: str, user: str,
                     logger: logging.Logger) -> tuple[str | None, str | None, dict, Any]:
    """Call the API, retry up to MAX_RETRIES with RETRY_WAIT_SECONDS gap.
    Returns (text, error_message_if_any, telemetry_dict, full_response_blob).
    text is None iff every attempt failed; full_response_blob is None iff failed.
    """
    last_err = ""
    t0 = time.monotonic()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text, meta, full = call_chat(model, system, user)
            elapsed = time.monotonic() - t0
            return text, None, {
                "attempts": attempt, "elapsed_s": round(elapsed, 3), **meta,
            }, full
        except requests.HTTPError as e:
            body = ""
            try:
                body = e.response.text[:500] if e.response is not None else ""
            except Exception:
                pass
            last_err = f"HTTP {getattr(e.response, 'status_code', '?')}: {body or str(e)}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        logger.warning("attempt %d/%d failed: %s", attempt, MAX_RETRIES, last_err[:300])
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_WAIT_SECONDS)
    elapsed = time.monotonic() - t0
    return None, last_err, {"attempts": MAX_RETRIES, "elapsed_s": round(elapsed, 3)}, None

# ============================================================================
#                          PER-MODEL EXECUTION
# ============================================================================

class ModelAbort(Exception):
    """Raised to abort a model after too many fatal failures."""


def _run_one_request(model: str, persona: str, persona_prompt: str,
                     item: dict, logger: logging.Logger) -> dict:
    user_q = build_user_question(item)
    text, err, telemetry, full = call_with_retry(model, persona_prompt, user_q, logger)
    parsed, used = (None, None)
    if text is not None:
        parsed, used = parse_response(text, item["response_format"])
        scored = apply_reverse_scoring(used, item["keyed"], item["response_format"])
    else:
        scored = None
    return {
        "persona": persona,
        "item_id": item["id"],
        "scale": item["scale"],
        "domain": item["domain"],
        "facet": item.get("facet"),
        "item_text": item["text"],
        "keyed": item["keyed"],
        "response_format": item["response_format"],
        "user_prompt": user_q,
        "raw_response": text,
        "full_response": full,  # complete API message blob (incl. any reasoning channels)
        "parsed_value": parsed,
        "scored_value": scored,
        "parse_failed": parsed is None and text is not None,
        "request_error": err,
        "telemetry": telemetry,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _aggregate_domain_scores(records: list[dict]) -> dict:
    """For one persona: scale::domain → {n_items, mean, std, min, max}."""
    agg: dict[tuple, list] = {}
    for rec in records:
        if rec["scored_value"] is None:
            continue
        key = (rec["scale"], rec["domain"])
        agg.setdefault(key, []).append(rec["scored_value"])
    out: dict[str, dict] = {}
    for (scale, domain), values in agg.items():
        if not values:
            continue
        n = len(values)
        m = sum(values) / n
        var = sum((v - m) ** 2 for v in values) / n if n > 1 else 0.0
        out[f"{scale}::{domain}"] = {
            "scale": scale,
            "domain": domain,
            "n_items": n,
            "mean_score": round(m, 4),
            "std_score": round(var ** 0.5, 4),
            "min_score": min(values),
            "max_score": max(values),
        }
    return out


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def run_model_full(model: str, items: list[dict], mbti_map: dict[str, str],
                    output_json: Path, logger: logging.Logger,
                    battery_meta: dict) -> dict:
    """Run all personas (Default + 16 MBTI) × all items for one model.
    Streams partial results into output_json after every persona block.
    """
    started_at = datetime.now().isoformat(timespec="seconds")
    state = {
        "model_name": model,
        "api_base": OPENAI_BASE_URL,
        "experiment_start": started_at,
        "experiment_end": None,
        "completion_status": "in_progress",
        "battery": {
            "total_items": len(items),
            "scales": battery_meta.get("scales", {}),
            "scoring_notes": (
                "scored_value applies reverse-key transformation: Likert → 6-raw, "
                "binary → 1-raw. parsed_value is the literal LLM answer BEFORE "
                "reverse-scoring."
            ),
        },
        "personas": PERSONA_ORDER,
        "n_personas": len(PERSONA_ORDER),
        "n_items_per_persona": len(items),
        "config": {
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "batch_size": BATCH_SIZE, "max_retries": MAX_RETRIES,
            "retry_wait_seconds": RETRY_WAIT_SECONDS,
            "model_fatal_threshold": MODEL_FATAL_THRESHOLD,
            "default_persona_uses_no_system_prompt": True,
        },
        "results_by_persona": {},
        "errors": [],
    }
    _atomic_write_json(output_json, state)

    consecutive_fatal = 0

    for persona in PERSONA_ORDER:
        persona_desc = mbti_map.get(persona)  # None for "Default"
        persona_prompt = make_persona_prompt(persona, persona_desc)
        logger.info("=== persona %s start (%d items) ===", persona, len(items))
        records: list[dict] = []
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
            futures = {
                pool.submit(_run_one_request, model, persona, persona_prompt,
                             item, logger): item["id"]
                for item in items
            }
            done = 0
            for fut in as_completed(futures):
                rec = fut.result()
                records.append(rec)
                done += 1
                if rec["request_error"]:
                    consecutive_fatal += 1
                    logger.error("[%s/%s] FAILED after retries: %s",
                                  persona, rec["item_id"], rec["request_error"][:200])
                    if consecutive_fatal >= MODEL_FATAL_THRESHOLD:
                        for f in futures:
                            f.cancel()
                        msg = (
                            f"Model `{model}` aborted after "
                            f"{consecutive_fatal} consecutive permanent failures. "
                            f"Last error: {rec['request_error']}"
                        )
                        append_error_log(model, msg)
                        logger.error(msg)
                        state["completion_status"] = "aborted_api_failure"
                        state["errors"].append(msg)
                        records.sort(key=lambda r: r["item_id"])
                        state["results_by_persona"][persona] = {
                            "persona": persona,
                            "persona_prompt": persona_prompt,
                            "responses": records,
                            "domain_scores": _aggregate_domain_scores(records),
                        }
                        state["experiment_end"] = datetime.now().isoformat(timespec="seconds")
                        _atomic_write_json(output_json, state)
                        raise ModelAbort(msg)
                else:
                    consecutive_fatal = 0
                if done % 25 == 0:
                    logger.info("  progress %s: %d/%d", persona, done, len(items))

        records.sort(key=lambda r: r["item_id"])
        state["results_by_persona"][persona] = {
            "persona": persona,
            "persona_prompt": persona_prompt,
            "responses": records,
            "domain_scores": _aggregate_domain_scores(records),
        }
        _atomic_write_json(output_json, state)
        logger.info("=== persona %s done, written ===", persona)

    state["completion_status"] = "complete"
    state["experiment_end"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write_json(output_json, state)
    logger.info("model %s complete → %s", model, output_json.name)
    return state

# ============================================================================
#                              CSV SUMMARY
# ============================================================================

CSV_FIELDS = [
    "model_name", "persona", "scale", "domain",
    "n_items", "mean_score", "std_score", "min_score", "max_score",
    "completion_status",
]


def append_summary_rows(csv_path: Path, model_state: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for persona, block in model_state["results_by_persona"].items():
            for _, dom in block["domain_scores"].items():
                w.writerow({
                    "model_name": model_state["model_name"],
                    "persona": persona,
                    "scale": dom["scale"],
                    "domain": dom["domain"],
                    "n_items": dom["n_items"],
                    "mean_score": dom["mean_score"],
                    "std_score": dom["std_score"],
                    "min_score": dom["min_score"],
                    "max_score": dom["max_score"],
                    "completion_status": model_state["completion_status"],
                })

# ============================================================================
#                                 MAIN
# ============================================================================

def driver_run_model(model: str, items: list[dict], mbti_map: dict[str, str],
                      output_json: Path, summary_csv: Path,
                      battery_meta: dict) -> str:
    logger = setup_logger(model)
    try:
        state = run_model_full(model, items, mbti_map, output_json, logger, battery_meta)
    except ModelAbort as e:
        logger.error("model aborted: %s", e)
        try:
            with open(output_json, encoding="utf-8") as f:
                state = json.load(f)
            append_summary_rows(summary_csv, state)
        except Exception as ex:
            logger.error("could not flush CSV after abort: %s", ex)
        return f"{model}: ABORTED — {e}"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        append_error_log(model, f"Unexpected crash: {msg}")
        logger.exception("unexpected crash")
        return f"{model}: CRASH — {msg}"
    append_summary_rows(summary_csv, state)
    return f"{model}: OK ({len(state['results_by_persona'])} personas)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true",
                    help="Smoke test: first model × (Default + 16 MBTI) × 10 items.")
    g.add_argument("--all", action="store_true",
                    help="Run every model in MODELS × 17 personas × full 221 items.")
    g.add_argument("--model", help="Run only the given model (full 17 × 221).")
    ap.add_argument("--n-items", type=int, default=None,
                    help="Optional override: cap items per persona (debugging).")
    args = ap.parse_args(argv)

    battery = load_battery()
    mbti_map = load_mbti()
    items_all: list[dict] = battery["items"]
    battery_meta = {"scales": battery.get("scales", {})}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.test:
        if not MODELS:
            print("MODELS list is empty — populate it first.", file=sys.stderr)
            return 2
        target_models = [MODELS[0]]
        items = items_all[:10]
        json_path = RESULTS_DIR / "test.json"
        csv_path = RESULTS_DIR / "test.csv"
        for p in (json_path, csv_path):
            if p.exists():
                p.unlink()
        max_workers = 1
    elif args.model:
        target_models = [args.model]
        items = items_all if args.n_items is None else items_all[: args.n_items]
        json_path = RESULTS_DIR / f"exp_mbti_{sanitize(args.model)}.json"
        csv_path = RESULTS_DIR / "summary.csv"
        max_workers = 1
    else:  # --all
        target_models = list(MODELS)
        items = items_all if args.n_items is None else items_all[: args.n_items]
        json_path = None
        csv_path = RESULTS_DIR / "summary.csv"
        max_workers = PARALLEL_MODELS

    print(f"API base: {OPENAI_BASE_URL}")
    print(f"Models: {target_models}")
    print(f"MBTI types: {len(mbti_map)} (+ Default → {len(PERSONA_ORDER)} personas)")
    print(f"Items per persona: {len(items)}")
    print(f"Total requests: {len(target_models) * len(PERSONA_ORDER) * len(items)}")
    print(f"Per-model batch: {BATCH_SIZE}; parallel models: {max_workers}")

    def submit_one(model: str) -> str:
        out = (json_path if json_path is not None
                else RESULTS_DIR / f"exp_mbti_{sanitize(model)}.json")
        return driver_run_model(model, items, mbti_map, out, csv_path, battery_meta)

    if max_workers == 1:
        for model in target_models:
            print(submit_one(model))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as outer:
            futs = {outer.submit(submit_one, m): m for m in target_models}
            for fut in as_completed(futs):
                print(fut.result())

    print(f"\nSummary CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
