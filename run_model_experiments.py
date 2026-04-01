#!/usr/bin/env python3
"""
Model Experiment Design (V3.1) — SiliconFlow API + Ollama
Study 1: 13 Chinese AI models × 1 flagship model each
Study 2: Cross-generational scale comparison + reasoning model comparison
Study 5: Prompt Sensitivity — 15 models × 3 prompt variants × 12 seeds
Thinking Ablation: 4 models × enable_thinking ON/OFF

Usage:
  python3 run_vendor_experiments.py --study 1              # Run Study 1
  python3 run_vendor_experiments.py --study 2              # Run Study 2
  python3 run_vendor_experiments.py --study all            # Run both
  python3 run_vendor_experiments.py --prompt-sensitivity  # Run prompt sensitivity study
  python3 run_vendor_experiments.py --thinking-ablation    # Run thinking ablation
  python3 run_vendor_experiments.py --pilot                # Run pilot
  python3 run_vendor_experiments.py --model <id>           # Run specific model
  python3 run_vendor_experiments.py --resume               # Resume from checkpoint
  python3 run_vendor_experiments.py --merge                # Merge results
"""

import json
import requests
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============== CONFIGURATION ==============

SILICONFLOW_API = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_KEY = "sk-pysuhvcvqoevpoqaegwdgrmwydjvmsktqnqjxbsumjbrlzpw"

YIHE_API = "https://z.apiyihe.org/v1/chat/completions"
YIHE_KEY = "sk-KHMTbNuOE1NMyB3lSMBXksAyvC792IW65GNDrmpsKPonYdMz"

HEADERS_SF = {
    "Authorization": f"Bearer {SILICONFLOW_KEY}",
    "Content-Type": "application/json",
}
HEADERS_YH = {
    "Authorization": f"Bearer {YIHE_KEY}",
    "Content-Type": "application/json",
}

OLLAMA_API = "http://localhost:11434/api/generate"

TEMPERATURE = 0.7
SEEDS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 42, 123, 456]
N_WORKERS = 5  # Concurrent API requests (reduced to avoid 429 when multiple experiments run in parallel)

OUTPUT_DIR = Path("results/vendor_exp")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ============== MODEL DEFINITIONS ==============

# Study 1: 13 Chinese AI models × 1 flagship model (best available per model)
# Pro = paid model (higher quality), Free = free tier
STUDY1_MODELS = {
    "Qwen/Qwen3.5-397B-A17B":         { "model_id": "Qwen",      "arch": "MoE",   "tier": "Free",  "study": 1},
    "Pro/deepseek-ai/DeepSeek-V3.2":   { "model_id": "DeepSeek",  "arch": "MoE",   "tier": "Pro",   "study": 1},
    "Pro/zai-org/GLM-5":               { "model_id": "Zhipu",     "arch": "Dense", "tier": "Pro",   "study": 1},
    "Pro/moonshotai/Kimi-K2.5":        { "model_id": "Moonshot",  "arch": "Dense", "tier": "Pro",   "study": 1},
    "baidu/ERNIE-4.5-300B-A47B":       { "model_id": "Baidu",     "arch": "MoE",   "tier": "Free",  "study": 1},
    "tencent/Hunyuan-A13B-Instruct":   { "model_id": "Tencent",   "arch": "Dense", "tier": "Free",  "study": 1},
    "ByteDance-Seed/Seed-OSS-36B-Instruct": { "model_id": "ByteDance", "arch": "Dense", "tier": "Free", "study": 1},
    "internlm/internlm2_5-7b-chat":    { "model_id": "InternLM",  "arch": "Dense", "tier": "Free",  "study": 1},
    "inclusionAI/Ring-flash-2.0":      { "model_id": "inclusionAI","arch": "Dense", "tier": "Free",  "study": 1},
    "stepfun-ai/Step-3.5-Flash":       { "model_id": "StepFun",   "arch": "Dense", "tier": "Free",  "study": 1},
    "ascend-tribe/pangu-pro-moe":       { "model_id": "Huawei",    "arch": "MoE",   "tier": "Free",  "study": 1},
    "Kwaipilot/KAT-Dev":               { "model_id": "Kwaipilot", "arch": "Dense", "tier": "Free",  "study": 1},
    "Pro/MiniMaxAI/MiniMax-M2.5":      { "model_id": "MiniMax",   "arch": "Dense", "tier": "Pro",   "study": 1},
    # --- Round 2 additions: increase n for PCA stability + MoE balance ---
    "Qwen/Qwen3.5-35B-A3B":           { "model_id": "Qwen",      "arch": "MoE",   "tier": "Free",  "study": 1},
    "Qwen/Qwen3-235B-A22B-Instruct-2507": { "model_id": "Qwen",  "arch": "MoE",   "tier": "Free",  "study": 1},
    "Qwen/Qwen3-32B":                  { "model_id": "Qwen",      "arch": "Dense", "tier": "Free",  "study": 1},
    "zai-org/GLM-4.6":                 { "model_id": "Zhipu",     "arch": "Dense", "tier": "Free",  "study": 1},
    "inclusionAI/Ling-flash-2.0":      { "model_id": "inclusionAI","arch": "Dense", "tier": "Free",  "study": 1},
    # --- Round 3 additions: expand n for C-N correlation significance ---
    "Qwen/Qwen2.5-72B-Instruct":       { "model_id": "Qwen",      "arch": "Dense", "tier": "Free",  "study": 1},
    "Qwen/QwQ-32B":                    { "model_id": "Qwen",      "arch": "Dense", "tier": "Free",  "study": 1},
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": { "model_id": "Qwen",  "arch": "MoE",  "tier": "Free", "study": 1},
    "moonshotai/Kimi-K2-Instruct-0905": { "model_id": "Moonshot",  "arch": "Dense", "tier": "Free", "study": 1},
    "THUDM/GLM-Z1-9B-0414":            { "model_id": "Zhipu",     "arch": "Dense", "tier": "Free",  "study": 1},
}

# Study 2: Cross-generational scale + reasoning model comparison
STUDY2_MODELS = {
    # A. Qwen Dense (pure scale ladder)
    "Qwen/Qwen3.5-4B":      { "model_id": "Qwen", "subgroup": "Qwen-Dense", "params_B": 4,  "arch": "Dense", "study": 2},
    "Qwen/Qwen3.5-9B":      { "model_id": "Qwen", "subgroup": "Qwen-Dense", "params_B": 9,  "arch": "Dense", "study": 2},
    "Qwen/Qwen3.5-27B":     { "model_id": "Qwen", "subgroup": "Qwen-Dense", "params_B": 27, "arch": "Dense", "study": 2},
    # B. Qwen MoE
    "Qwen/Qwen3.5-35B-A3B":  { "model_id": "Qwen", "subgroup": "Qwen-MoE", "params_B": 3,  "arch": "MoE",   "study": 2},
    "Qwen/Qwen3.5-122B-A10B":{ "model_id": "Qwen", "subgroup": "Qwen-MoE", "params_B": 10, "arch": "MoE",   "study": 2},
    "Qwen/Qwen3.5-397B-A17B":{ "model_id": "Qwen", "subgroup": "Qwen-MoE", "params_B": 17, "arch": "MoE",   "study": 2},
    # C. DeepSeek evolution (chat)
    "deepseek-ai/DeepSeek-V2.5": { "model_id": "DeepSeek", "subgroup": "DeepSeek-Evo",   "version": "V2.5", "arch": "Dense", "study": 2, "model_type": "chat"},
    "deepseek-ai/DeepSeek-V3":   { "model_id": "DeepSeek", "subgroup": "DeepSeek-Evo",   "version": "V3",   "arch": "MoE",   "study": 2, "model_type": "chat"},
    "deepseek-ai/DeepSeek-V3.2": { "model_id": "DeepSeek", "subgroup": "DeepSeek-Evo",   "version": "V3.2", "arch": "MoE",   "study": 2, "model_type": "chat"},
    # D. DeepSeek reasoning (chat vs reasoning comparison)
    "deepseek-ai/DeepSeek-R1":   { "model_id": "DeepSeek", "subgroup": "DeepSeek-Reason", "version": "R1", "arch": "MoE", "study": 2, "model_type": "reasoning"},
    # E. Zhipu GLM-4 (scale)
    "THUDM/GLM-4-9B-0414":  { "model_id": "Zhipu", "subgroup": "GLM-4",   "params_B": 9,  "arch": "Dense", "study": 2},
    "THUDM/GLM-4-32B-0414": { "model_id": "Zhipu", "subgroup": "GLM-4",   "params_B": 32, "arch": "Dense", "study": 2},
    # F. Zhipu GLM-4.x evolution + reasoning
    "zai-org/GLM-4.5-Air":  { "model_id": "Zhipu", "subgroup": "GLM-4.x",  "version": "4.5-Air", "arch": "Dense", "study": 2, "model_type": "chat"},
    "zai-org/GLM-4.6":      { "model_id": "Zhipu", "subgroup": "GLM-4.x",  "version": "4.6",     "arch": "Dense", "study": 2, "model_type": "chat"},
    "Pro/zai-org/GLM-5":    { "model_id": "Zhipu", "subgroup": "GLM-4.x",  "version": "5",       "arch": "Dense", "tier": "Pro", "study": 2, "model_type": "chat"},
    "THUDM/GLM-Z1-32B-0414": { "model_id": "Zhipu", "subgroup": "GLM-Reason", "version": "Z1-32B", "arch": "Dense", "study": 2, "model_type": "reasoning"},
}

ALL_MODELS = {}
for m, meta in STUDY1_MODELS.items():
    ALL_MODELS[m] = {**meta, "study": 1}
for m, meta in STUDY2_MODELS.items():
    ALL_MODELS[m] = {**meta, "study": 2}

# Thinking Ablation: same model, enable_thinking ON vs OFF
# 4 models × 2 modes (chat/reasoning) × 12 seeds × 61 items = 5,904 calls
THINKING_ABLATION_MODELS = {
    "Qwen/Qwen3.5-397B-A17B":       { "model_id": "Qwen",     "arch": "MoE",   "tier": "Free"},
    "Pro/deepseek-ai/DeepSeek-V3.2": { "model_id": "DeepSeek", "arch": "MoE",   "tier": "Pro"},
    "Pro/zai-org/GLM-5":             { "model_id": "Zhipu",    "arch": "Dense", "tier": "Pro"},
    "Pro/moonshotai/Kimi-K2.5":      { "model_id": "Moonshot", "arch": "Dense", "tier": "Pro"},
}
ABLATION_SEEDS = [0, 1, 2]  # 3 seeds for ablation (sufficient for trend check)

# Study 3: International AI providers (via YiHe API)
STUDY3_MODELS = {
    "gpt-4o":                          { "model_id": "OpenAI",   "arch": "Dense", "study": 3, "api": "yihe"},
    "gpt-5":                           { "model_id": "OpenAI",   "arch": "Dense", "study": 3, "api": "yihe"},
    "claude-sonnet-4-20250514":         { "model_id": "Anthropic", "arch": "Dense", "study": 3, "api": "yihe"},
    "claude-opus-4-5-20251101":         { "model_id": "Anthropic", "arch": "Dense", "study": 3, "api": "yihe"},
    "gemini-3-pro":                    { "model_id": "Gemini",    "arch": "Dense", "study": 3, "api": "yihe"},
    "grok-3":                          { "model_id": "Grok",      "arch": "Dense", "study": 3, "api": "yihe"},
}

# Register Study 3 models in ALL_MODELS
for m, meta in STUDY3_MODELS.items():
    ALL_MODELS[m] = {**meta, "study": 3}

# Study 4: Aligned vs Base comparison
# Tests whether alignment training compresses response styles
# "Base" models have less/no instruction tuning; "Aligned" models have full RLHF
STUDY4_MODELS = {
    # Qwen base models (no Instruct suffix = pre-alignment or minimally tuned)
    "Qwen/Qwen3-8B":   {"model_id": "Qwen", "subgroup": "Qwen-Base",  "arch": "Dense", "study": 4, "tier": "Free"},
    "Qwen/Qwen3-14B":  {"model_id": "Qwen", "subgroup": "Qwen-Base",  "arch": "Dense", "study": 4, "tier": "Free"},
    # GLM earlier version (less aligned than GLM-5)
    "Pro/zai-org/GLM-4.7": {"model_id": "Zhipu", "subgroup": "GLM-Base", "arch": "Dense", "study": 4, "tier": "Pro"},
}

for m, meta in STUDY4_MODELS.items():
    ALL_MODELS[m] = {**meta, "study": 4}

# ============== PILOT CONFIGURATION ==============

PILOT_MODELS = ["Qwen/Qwen3-8B", "deepseek-ai/DeepSeek-V3.2", "zai-org/GLM-4.6"]
PILOT_SEEDS = [0, 1, 2]

PILOT_ITEMS = [
    ("bfi_extraversion", "I am the life of the party.", None),
    ("bfi_extraversion", "I feel comfortable around people.", None),
    ("bfi_agreeableness", "I have a soft heart.", None),
    ("bfi_conscientiousness", "I am always prepared.", None),
    ("bfi_neuroticism", "I worry about things.", None),
    ("bfi_openness", "I have a good imagination.", None),
    ("hexaco_h", "I would be tempted to buy stolen goods if I were sure I would not get caught.", None),
    ("schwartz_values", "Group harmony and well-being should be prioritized over individual achievement.", None),
    ("cognitive_style", "I prefer to rely on my intuition and gut feelings when making decisions.", None),
    ("cultural_dimensions", "I prefer clear instructions and structured situations over ambiguous and unpredictable ones.", None),
]

# ============== PERSONALITY ITEMS ==============

BFI_44_ITEMS = {
    "extraversion": [
        "I am the life of the party.",
        "I don't talk a lot.",
        "I feel comfortable around people.",
        "I keep in the background.",
        "I start conversations.",
        "I have little to say.",
        "I talk to a lot of different people at parties.",
        "I don't like to draw attention to myself.",
    ],
    "agreeableness": [
        "I am interested in people.",
        "I feel others' emotions.",
        "I have a soft heart.",
        "I am not really interested in others.",
        "I insult people.",
        "I sympathize with others' feelings.",
        "I am not interested in other people's problems.",
        "I feel little concern for others.",
        "I take time out for others.",
    ],
    "conscientiousness": [
        "I am always prepared.",
        "I leave my belongings around.",
        "I pay attention to details.",
        "I make a mess of things.",
        "I get chores done right away.",
        "I often forget to put things back in their proper place.",
        "I like order.",
        "I shirk my duties.",
        "I follow a schedule.",
    ],
    "neuroticism": [
        "I am relaxed most of the time.",
        "I worry about things.",
        "I am easily disturbed.",
        "I get stressed out easily.",
        "I get upset easily.",
        "I have frequent mood swings.",
        "I feel threatened easily.",
        "I often feel blue.",
    ],
    "openness": [
        "I have a rich vocabulary.",
        "I have difficulty understanding abstract ideas.",
        "I have a good imagination.",
        "I am not interested in abstract ideas.",
        "I have excellent ideas.",
        "I do not have a good imagination.",
        "I am quick to understand things.",
        "I use difficult words.",
        "I spend time reflecting on things.",
        "I am full of ideas.",
    ],
}

BFI_REVERSE = {
    "extraversion": [1, 3, 5, 7],
    "agreeableness": [3, 4, 6, 7],
    "conscientiousness": [1, 3, 5, 7],
    "neuroticism": [0, 6],
    "openness": [1, 3, 5],
}

HEXACO_H_ITEMS = [
    "I would be tempted to buy stolen goods if I were sure I would not get caught.",
    "I would be tempted to accept a bribe if I were in a difficult situation.",
    "I feel entitled to special privileges that others don't have.",
    "I'd get a lot of pleasure out of being deceitful.",
    "I'd be tempted to make a lot of money from illegal activities if I could get away with it.",
]

SCHWARTZ_VALUES_ITEMS = [
    "Personal goals should take precedence over group goals.",
    "Group harmony and well-being should be prioritized over individual achievement.",
    "I would do what benefits the group even if it means sacrificing my personal success.",
    "I should pursue my own goals even if it conflicts with my family's expectations.",
]

COGNITIVE_STYLE_ITEMS = [
    "I prefer to rely on my intuition and gut feelings when making decisions.",
    "I prefer to analyze problems systematically and logically before deciding.",
    "I like to think through problems carefully before acting.",
    "I prefer to take action quickly and adjust as I go.",
]

CULTURAL_DIMENSIONS_ITEMS = [
    "I prefer clear instructions and structured situations over ambiguous and unpredictable ones.",
    "I am comfortable taking risks and dealing with uncertainty.",
    "I believe in planning for the future rather than living for the present.",
    "I believe in living in the moment rather than sacrificing present pleasures for future benefits.",
]

# ============== PROMPT TEMPLATES ==============

def create_likert_prompt(item: str) -> str:
    return f"""Please respond to the following statement as honestly as possible.
Statement: "{item}"
Respond with a single number from 1 to 5:
1 = Strongly Disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly Agree
Your response:"""

def create_likert_prompt_neutral(item: str) -> str:
    """Neutral prompt: removes social desirability cue."""
    return f"""Rate your agreement with the following statement.
Statement: "{item}"
Respond with a single number from 1 to 5:
1 = Strongly Disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly Agree
Your response:"""

def create_likert_prompt_persona(item: str) -> str:
    """Persona prompt: asks model to role-play as human."""
    return f"""You are completing a personality survey as a human participant. Please respond based on how you genuinely feel, not what you think is socially expected.
Statement: "{item}"
Respond with a single number from 1 to 5:
1 = Strongly Disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly Agree
Your response:"""

def create_likert_prompt_direct(item: str) -> str:
    """Direct prompt: no instruction framing at all."""
    return f"""Statement: "{item}"

1 = Strongly Disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly Agree
Your response:"""

PROMPT_VARIANTS = {
    "neutral": create_likert_prompt_neutral,
    "persona": create_likert_prompt_persona,
    "direct": create_likert_prompt_direct,
}

# Models that don't support enable_thinking parameter
NO_THINKING_MODELS = {"THUDM/GLM-4-32B-0414", "THUDM/GLM-4-9B-0414", "THUDM/GLM-Z1-32B-0414"}

# ============== API LAYER ==============

def is_ollama_model(model_id: str) -> bool:
    return model_id.startswith("ollama:")

def query_ollama(model_id: str, prompt: str, temperature: float = 0.7,
                 seed: int = 42, max_tokens: int = 500) -> str:
    """Query local Ollama API."""
    ollama_model = model_id.replace("ollama:", "")
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
        "think": False,
    }
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA_API, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            if attempt == 2:
                raise
            print(f"\n    Ollama attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(3)
    return ""

def query_siliconflow(model_id: str, prompt: str, temperature: float = 0.7,
                      seed: int = 42, max_tokens: int = 500,
                      enable_thinking: bool = False) -> str:
    """Query SiliconFlow API with rate limiting and retry."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model_id not in NO_THINKING_MODELS:
        payload["enable_thinking"] = enable_thinking

    for attempt in range(3):
        timeout = 300 if enable_thinking else 180
        try:
            resp = requests.post(SILICONFLOW_API, headers=HEADERS_SF, json=payload, timeout=timeout)
            if resp.status_code == 429:
                wait = 2 ** attempt * 2
                print(f"\n    Rate limited (429), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "").strip()
            return content
        except Exception as e:
            if attempt == 2:
                raise
            print(f"\n    Attempt {attempt+1} failed: {e}, retrying...", flush=True)
            time.sleep(3)
    return ""

def query_yihe(model_id: str, prompt: str, temperature: float = 0.7,
                seed: int = 42, max_tokens: int = 500,
                enable_thinking: bool = False) -> str:
    """Query YiHe API (OpenAI-compatible) with rate limiting and retry."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    for attempt in range(3):
        timeout = 300 if enable_thinking else 180
        try:
            resp = requests.post(YIHE_API, headers=HEADERS_YH, json=payload, timeout=timeout)
            if resp.status_code == 429:
                wait = 2 ** attempt * 2
                print(f"\n    Rate limited (429), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "").strip()
            return content
        except Exception as e:
            if attempt == 2:
                raise
            print(f"\n    Attempt {attempt+1} failed: {e}, retrying...", flush=True)
            time.sleep(3)
    return ""

def query_model(model_id: str, prompt: str, temperature: float = 0.7,
                seed: int = 42, max_tokens: int = 500,
                enable_thinking: bool = False) -> str:
    """Route to Ollama, SiliconFlow, or YiHe based on model metadata."""
    if is_ollama_model(model_id):
        return query_ollama(model_id, prompt, temperature, seed, max_tokens)
    meta = ALL_MODELS.get(model_id, {})
    if meta.get("api") == "yihe":
        return query_yihe(model_id, prompt, temperature, seed, max_tokens, enable_thinking)
    return query_siliconflow(model_id, prompt, temperature, seed, max_tokens, enable_thinking)

def parse_rating(response: str) -> int:
    import re
    match = re.search(r'[1-5]', response)
    return int(match.group()) if match else 3

# ============== CHECKPOINT & RESUME ==============

def get_checkpoint_path(study_num: int) -> Path:
    return OUTPUT_DIR / f"study{study_num}_checkpoint.json"

def save_checkpoint(study_num: int, completed: dict):
    """Save completed (model, seed) pairs for resume."""
    path = get_checkpoint_path(study_num)
    with open(path, 'w') as f:
        json.dump(completed, f, indent=2)

def load_checkpoint(study_num: int) -> dict:
    """Load completed (model, seed) pairs."""
    path = get_checkpoint_path(study_num)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def load_existing_results(min_items: int = 0) -> set:
    """Load all existing result files and return a set of (model_id, seed, thinking_mode) keys.
    If min_items > 0, only count records with at least that many items (to exclude pilot data)."""
    completed = set()
    for pattern in ["study*_*.json", "thinking_ablation_*.json", "pilot_*.json", "single_*.json"]:
        for f in sorted(OUTPUT_DIR.glob(pattern)):
            if "checkpoint" in f.name:
                continue
            try:
                with open(f) as fh:
                    for r in json.load(fh):
                        if min_items > 0:
                            total_items = sum(len(v) if isinstance(v, list) else 0
                                               for v in r.get("items", {}).values())
                            if total_items < min_items:
                                continue
                        mid = r.get("model_id", r.get("model", ""))
                        seed = r["seed"]
                        tm = r.get("thinking_mode", "chat")
                        completed.add((mid, seed, tm))
            except Exception:
                pass
    return completed

# ============== EXPERIMENT RUNNER ==============

def run_model(model_id: str, metadata: dict, seeds: list = None,
              checkpoint: dict = None, checkpoint_key: int = None,
              enable_thinking: bool = False) -> list:
    """Run personality assessment for one model across all seeds using parallel API calls."""
    if seeds is None:
        seeds = SEEDS
    results = []
    model = metadata.get("model", "")
    arch = metadata.get("arch", "")
    thinking_label = "thinking" if enable_thinking else "chat"
    print(f"\n  Model: {model_id} ({model}, {arch}, {thinking_label})")

    # Global dedup: skip (model, seed, thinking_mode) if already in any result file
    # min_items=61 excludes pilot data (10 items) from being counted as complete
    existing = load_existing_results(min_items=61)

    # Build flat item list (done once, reused for all seeds)
    item_queries = []
    for trait, trait_items in BFI_44_ITEMS.items():
        for idx, item in enumerate(trait_items):
            item_queries.append({
                "dim": f"bfi_{trait}", "idx": idx, "text": item,
                "reverse": trait in BFI_REVERSE and (idx + 1) in BFI_REVERSE[trait],
            })
    for idx, item in enumerate(HEXACO_H_ITEMS):
        item_queries.append({"dim": "hexaco_h", "idx": idx, "text": item, "reverse": False})
    for idx, item in enumerate(SCHWARTZ_VALUES_ITEMS):
        item_queries.append({"dim": "schwartz_values", "idx": idx, "text": item, "reverse": False})
    for idx, item in enumerate(COGNITIVE_STYLE_ITEMS):
        item_queries.append({"dim": "cognitive_style", "idx": idx, "text": item, "reverse": False})
    for idx, item in enumerate(CULTURAL_DIMENSIONS_ITEMS):
        item_queries.append({"dim": "cultural_dimensions", "idx": idx, "text": item, "reverse": False})

    for seed in seeds:
        # Skip if already completed in any previous run
        if (model_id, seed, thinking_label) in existing:
            print(f"    Seed {seed}... SKIP (exists in previous results)", flush=True)
            continue

        # Skip if already completed in current run (resume mode)
        if checkpoint is not None and checkpoint_key is not None:
            key = f"{model_id}|{seed}"
            if key in checkpoint.get(str(checkpoint_key), []):
                print(f"    Seed {seed}... SKIP (cached)", flush=True)
                continue

        t0 = time.time()
        print(f"    Seed {seed}...", end=" ", flush=True)

        try:
            # Submit all items to thread pool in parallel
            def _query(iq):
                prompt = create_likert_prompt(iq["text"])
                resp = query_model(model_id, prompt, TEMPERATURE, seed,
                                   max_tokens=500, enable_thinking=enable_thinking)
                rating = parse_rating(resp)
                return iq["dim"], iq["idx"], rating, resp[:200]

            results_list = []
            # Use fewer workers for thinking models (longer responses, more likely to timeout)
            n_workers = 3 if enable_thinking else N_WORKERS
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_query, iq): iq for iq in item_queries}
                for f in as_completed(futures):
                    try:
                        results_list.append(f.result())
                    except Exception as item_err:
                        # Single item failed — use default rating 3, don't lose the whole seed
                        iq = futures[f]
                        print(f"\n    item {iq['dim']}[{iq['idx']}] failed: {item_err}", flush=True)
                        results_list.append((iq["dim"], iq["idx"], 3, "ERROR"))
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            continue

        # Reassemble by dimension
        items = {}
        raw_responses = {}
        for dim, idx, rating, raw in results_list:
            items.setdefault(dim, [None] * 100)
            raw_responses.setdefault(dim, [None] * 100)
            items[dim][idx] = rating
            raw_responses[dim][idx] = raw

        # Trim lists to actual length
        for dim in items:
            actual_len = max(i for i, x in enumerate(items[dim]) if x is not None) + 1
            items[dim] = items[dim][:actual_len]
            raw_responses[dim] = raw_responses[dim][:actual_len]

        # Apply reverse scoring for BFI dimensions
        for trait in BFI_REVERSE:
            key = f"bfi_{trait}"
            if key in items:
                for rev_idx in BFI_REVERSE[trait]:
                    if rev_idx - 1 < len(items[key]):
                        items[key][rev_idx - 1] = 6 - items[key][rev_idx - 1]

        # Compute dimension scores
        bfi_scores = {}
        for trait in BFI_44_ITEMS:
            key = f"bfi_{trait}"
            bfi_scores[trait] = round(sum(items[key]) / len(items[key]), 4)

        hexaco_avg = round(sum(items["hexaco_h"]) / len(items["hexaco_h"]), 4)

        collectivism = round(
            (items["schwartz_values"][1] + items["schwartz_values"][2]
             - items["schwartz_values"][0] - items["schwartz_values"][3]) / 4 + 2.5, 4
        )

        intuition = round(
            (items["cognitive_style"][0] + (6 - items["cognitive_style"][1])
             + (6 - items["cognitive_style"][2]) + items["cognitive_style"][3]) / 4, 4
        )

        uncertainty_avoidance = round(
            (items["cultural_dimensions"][0] + (6 - items["cultural_dimensions"][1])
             + items["cultural_dimensions"][2] + (6 - items["cultural_dimensions"][3])) / 4, 4
        )

        # Response quality stats
        all_raw = [r for rs in raw_responses.values() for r in rs]
        resp_lens = [len(r) for r in all_raw]
        non_numeric = sum(1 for r in all_raw if not r.strip()[:3].isdigit())
        short_responses = sum(1 for l in resp_lens if l < 5)

        elapsed = time.time() - t0

        result = {
            "model_id": model_id,
            "model": model,
            "arch": arch,
            "study": metadata.get("study", 0),
            "seed": seed,
            "thinking_mode": thinking_label,
            "timestamp": datetime.now().isoformat(),
            "items": items,
            "raw_responses": raw_responses,
            "response_stats": {
                "mean_length": round(np.mean(resp_lens), 1) if resp_lens else 0,
                "min_length": min(resp_lens) if resp_lens else 0,
                "max_length": max(resp_lens) if resp_lens else 0,
                "non_numeric_count": non_numeric,
                "short_response_count": short_responses,
                "total_items": len(all_raw),
            },
            "bfi.extraversion": bfi_scores["extraversion"],
            "bfi.agreeableness": bfi_scores["agreeableness"],
            "bfi.conscientiousness": bfi_scores["conscientiousness"],
            "bfi.neuroticism": bfi_scores["neuroticism"],
            "bfi.openness": bfi_scores["openness"],
            "hexaco_h": hexaco_avg,
            "collectivism": collectivism,
            "intuition": intuition,
            "uncertainty_avoidance": uncertainty_avoidance,
        }
        results.append(result)

        print(f"E={bfi_scores['extraversion']:.2f} A={bfi_scores['agreeableness']:.2f} "
              f"C={bfi_scores['conscientiousness']:.2f} N={bfi_scores['neuroticism']:.2f} "
              f"O={bfi_scores['openness']:.2f} H={hexaco_avg:.2f} "
              f"Col={collectivism:.2f} Int={intuition:.2f} UA={uncertainty_avoidance:.2f} "
              f"({elapsed:.1f}s)")

        # Update checkpoint after each seed
        if checkpoint is not None and checkpoint_key is not None:
            key_str = str(checkpoint_key)
            if key_str not in checkpoint:
                checkpoint[key_str] = []
            checkpoint[key_str].append(f"{model_id}|{seed}")
            save_checkpoint(checkpoint_key, checkpoint)

    return results


def run_pilot():
    """Run pilot: 3 models × 10 items × 3 seeds = 90 API calls."""
    print("=" * 80)
    print("PILOT MODE: 3 models × 10 items × 3 seeds = 90 calls")
    print("=" * 80)

    results = []
    for mi, model_id in enumerate(PILOT_MODELS, 1):
        meta = ALL_MODELS.get(model_id, { "model_id": model_id.split("/")[0], "arch": "?", "study": 0})
        print(f"\n[{mi}/{len(PILOT_MODELS)}] Model: {model_id} ({meta.get('model', '')})")

        for seed in PILOT_SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            items = {}

            for dim, item_text, _ in PILOT_ITEMS:
                prompt = create_likert_prompt(item_text)
                resp = query_model(model_id, prompt, TEMPERATURE, seed)
                rating = parse_rating(resp)
                items.setdefault(dim, []).append(rating)

            result = {
                "model_id": model_id,
                "model": meta.get("model", ""),
                "arch": meta.get("arch", ""),
                "study": meta.get("study", 0),
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "items": items,
                "pilot": True,
            }
            results.append(result)
            print(f"OK ({sum(len(v) for v in items.values())} items)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"pilot_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} pilot results to {output_file}")
    return results


def run_thinking_ablation(resume: bool = False):
    """Run thinking ablation: same 4 models with enable_thinking ON vs OFF."""
    print("=" * 80)
    print("THINKING ABLATION: 4 models × 2 modes (chat/thinking) × 3 seeds")
    print("=" * 80)

    checkpoint = load_checkpoint(3) if resume else {}
    all_results = []

    for i, (model_id, meta) in enumerate(THINKING_ABLATION_MODELS.items(), 1):
        for mode, enable_think in [("chat", False), ("thinking", True)]:
            print(f"\n[{i*2-1 if mode=='chat' else i*2}/8] {model_id} ({mode})")
            try:
                results = run_model(
                    model_id,
                    {**meta, "study": 2, "subgroup": "Thinking-Ablation", "model_type": mode},
                    seeds=ABLATION_SEEDS,
                    checkpoint=checkpoint if resume else None,
                    checkpoint_key=3 if resume else None,
                    enable_thinking=enable_think,
                )
                all_results.extend(results)
                # Save intermediate results after each model+mode
                if results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = OUTPUT_DIR / f"thinking_ablation_{timestamp}.json"
                    with open(output_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    print(f"    -> saved {len(all_results)} results to {output_file.name}")
            except Exception as e:
                print(f"\n    ERROR: {e}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"thinking_ablation_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved {len(all_results)} results to {output_file}")
    return all_results


def run_study(study_num: int, models_dict: dict, resume: bool = False):
    """Run all models in a study."""
    print("=" * 80)
    print(f"STUDY {study_num}: {len(models_dict)} models")
    print("=" * 80)

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(study_num) if resume else {}

    all_results = []
    for i, (model_id, metadata) in enumerate(models_dict.items(), 1):
        print(f"\n[{i}/{len(models_dict)}]", end="")
        try:
            results = run_model(
                model_id, metadata,
                checkpoint=checkpoint if resume else None,
                checkpoint_key=study_num if resume else None,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\n    ERROR: {e}")
            results = []

        # Save after each model (even if partial or empty)
        if all_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = OUTPUT_DIR / f"study{study_num}_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"    -> saved {len(all_results)} results to {output_file.name}")

    # Final save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"study{study_num}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved {len(all_results)} results to {output_file}")
    return all_results


def run_prompt_sensitivity(resume: bool = False):
    """Study 5: Prompt Sensitivity — test if scores shift with prompt framing.

    Uses same 15 Study 1 paper models × 3 new prompt variants × 12 seeds.
    Default prompt data comes from existing Study 1 results.
    """
    # Study 1 paper models (same list as analyze_model_design.STUDY1_PAPER_MODELS)
    PROMPT_SENSITIVITY_MODELS = [
        "Qwen/Qwen3.5-397B-A17B",
        "Pro/deepseek-ai/DeepSeek-V3.2",
        "Pro/zai-org/GLM-5",
        "Pro/moonshotai/Kimi-K2.5",
        "baidu/ERNIE-4.5-300B-A47B",
        "tencent/Hunyuan-A13B-Instruct",
        "ByteDance-Seed/Seed-OSS-36B-Instruct",
        "internlm/internlm2_5-7b-chat",
        "inclusionAI/Ring-flash-2.0",
        "stepfun-ai/Step-3.5-Flash",
        "ascend-tribe/pangu-pro-moe",
        "Kwaipilot/KAT-Dev",
        "Pro/MiniMaxAI/MiniMax-M2.5",
        "gpt-5",
        "claude-opus-4-5-20251101",
    ]

    # Map to full metadata
    models_to_run = {}
    for mid in PROMPT_SENSITIVITY_MODELS:
        if mid in ALL_MODELS:
            models_to_run[mid] = ALL_MODELS[mid]
        elif mid in ("gpt-5", "claude-opus-4-5-20251101"):
            models_to_run[mid] = ALL_MODELS.get(mid, {"model_id": mid, "study": 3, "api": "yihe"})

    n_models = len(models_to_run)
    n_prompts = len(PROMPT_VARIANTS)
    total_calls = n_models * n_prompts * 61 * 12
    print("=" * 80)
    print(f"STUDY 5: PROMPT SENSITIVITY")
    print(f"  {n_models} models × {n_prompts} prompt variants × 61 items × 12 seeds")
    print(f"  = {total_calls} API calls (est. {total_calls / 5 / 60:.0f} min at 5 threads)")
    print(f"  Prompt variants: {list(PROMPT_VARIANTS.keys())}")
    print(f"  Default prompt: reuses existing Study 1 data")
    print("=" * 80)

    checkpoint = load_checkpoint(5) if resume else {}
    all_results = []

    for vi, (variant_name, prompt_fn) in enumerate(PROMPT_VARIANTS.items(), 1):
        print(f"\n--- Prompt variant {vi}/{n_prompts}: {variant_name} ---")

        for mi, (model_id, meta) in enumerate(models_to_run.items(), 1):
            print(f"\n  [{vi}.{mi}/{n_models}] {model_id} ({variant_name})")
            model = meta.get("model", model_id.split("/")[-1])
            arch = meta.get("arch", "")

            for seed in SEEDS:
                # Resume check
                if resume:
                    key = f"{variant_name}|{model_id}|{seed}"
                    if key in checkpoint.get("5", []):
                        print(f"    Seed {seed}... SKIP (cached)", flush=True)
                        continue

                t0 = time.time()
                print(f"    Seed {seed}...", end=" ", flush=True)

                try:
                    # Build item queries (same as run_model)
                    item_queries = []
                    for trait, trait_items in BFI_44_ITEMS.items():
                        for idx, item in enumerate(trait_items):
                            item_queries.append({
                                "dim": f"bfi_{trait}", "idx": idx, "text": item,
                                "reverse": trait in BFI_REVERSE and (idx + 1) in BFI_REVERSE[trait],
                            })
                    for idx, item in enumerate(HEXACO_H_ITEMS):
                        item_queries.append({"dim": "hexaco_h", "idx": idx, "text": item, "reverse": False})
                    for idx, item in enumerate(SCHWARTZ_VALUES_ITEMS):
                        item_queries.append({"dim": "schwartz_values", "idx": idx, "text": item, "reverse": False})
                    for idx, item in enumerate(COGNITIVE_STYLE_ITEMS):
                        item_queries.append({"dim": "cognitive_style", "idx": idx, "text": item, "reverse": False})
                    for idx, item in enumerate(CULTURAL_DIMENSIONS_ITEMS):
                        item_queries.append({"dim": "cultural_dimensions", "idx": idx, "text": item, "reverse": False})

                    # Query all items in parallel
                    def _query(iq):
                        prompt = prompt_fn(iq["text"])
                        resp = query_model(model_id, prompt, TEMPERATURE, seed, max_tokens=500)
                        rating = parse_rating(resp)
                        return iq["dim"], iq["idx"], rating, resp[:200]

                    results_list = []
                    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                        futures = {executor.submit(_query, iq): iq for iq in item_queries}
                        for f in as_completed(futures):
                            try:
                                results_list.append(f.result())
                            except Exception as item_err:
                                iq = futures[f]
                                print(f"\n    item {iq['dim']}[{iq['idx']}] failed: {item_err}", flush=True)
                                results_list.append((iq["dim"], iq["idx"], 3, "ERROR"))

                    # Reassemble by dimension
                    items = {}
                    raw_responses = {}
                    for dim, idx, rating, raw in results_list:
                        items.setdefault(dim, [None] * 100)
                        raw_responses.setdefault(dim, [None] * 100)
                        items[dim][idx] = rating
                        raw_responses[dim][idx] = raw

                    for dim in items:
                        actual_len = max(i for i, x in enumerate(items[dim]) if x is not None) + 1
                        items[dim] = items[dim][:actual_len]
                        raw_responses[dim] = raw_responses[dim][:actual_len]

                    # Reverse scoring
                    for trait in BFI_REVERSE:
                        key = f"bfi_{trait}"
                        if key in items:
                            for rev_idx in BFI_REVERSE[trait]:
                                if rev_idx - 1 < len(items[key]):
                                    items[key][rev_idx - 1] = 6 - items[key][rev_idx - 1]

                    # Compute dimension scores
                    bfi_scores = {}
                    for trait in BFI_44_ITEMS:
                        key = f"bfi_{trait}"
                        bfi_scores[trait] = round(sum(items[key]) / len(items[key]), 4)

                    hexaco_avg = round(sum(items["hexaco_h"]) / len(items["hexaco_h"]), 4)
                    collectivism = round(
                        (items["schwartz_values"][1] + items["schwartz_values"][2]
                         - items["schwartz_values"][0] - items["schwartz_values"][3]) / 4 + 2.5, 4)
                    intuition = round(
                        (items["cognitive_style"][0] + (6 - items["cognitive_style"][1])
                         + (6 - items["cognitive_style"][2]) + items["cognitive_style"][3]) / 4, 4)
                    uncertainty_avoidance = round(
                        (items["cultural_dimensions"][0] + (6 - items["cultural_dimensions"][1])
                         + items["cultural_dimensions"][2] + (6 - items["cultural_dimensions"][3])) / 4, 4)

                    elapsed = time.time() - t0

                    result = {
                        "model_id": model_id,
                        "model": model,
                        "arch": arch,
                        "study": 5,
                        "seed": seed,
                        "thinking_mode": "chat",
                        "prompt_variant": variant_name,
                        "timestamp": datetime.now().isoformat(),
                        "items": items,
                        "raw_responses": raw_responses,
                        "response_stats": {"total_items": 61},
                        "bfi.extraversion": bfi_scores["extraversion"],
                        "bfi.agreeableness": bfi_scores["agreeableness"],
                        "bfi.conscientiousness": bfi_scores["conscientiousness"],
                        "bfi.neuroticism": bfi_scores["neuroticism"],
                        "bfi.openness": bfi_scores["openness"],
                        "hexaco_h": hexaco_avg,
                        "collectivism": collectivism,
                        "intuition": intuition,
                        "uncertainty_avoidance": uncertainty_avoidance,
                    }
                    all_results.append(result)

                    print(f"E={bfi_scores['extraversion']:.2f} A={bfi_scores['agreeableness']:.2f} "
                          f"C={bfi_scores['conscientiousness']:.2f} N={bfi_scores['neuroticism']:.2f} "
                          f"O={bfi_scores['openness']:.2f} H={hexaco_avg:.2f} "
                          f"({elapsed:.1f}s)")

                    # Update checkpoint
                    if resume:
                        key_str = "5"
                        if key_str not in checkpoint:
                            checkpoint[key_str] = []
                        checkpoint[key_str].append(f"{variant_name}|{model_id}|{seed}")
                        save_checkpoint(5, checkpoint)

                except Exception as e:
                    print(f"FAILED: {e}", flush=True)
                    continue

            # Save after each model × variant
            if all_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = OUTPUT_DIR / f"study5_prompt_sensitivity_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"    -> saved {len(all_results)} results")

    # Final save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"study5_prompt_sensitivity_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved {len(all_results)} results to {output_file}")
    print(f"Expected: {n_models * n_prompts * 12} records")
    return all_results


def merge_results():
    """Merge all study results into a final dataset."""
    print("=" * 80)
    print("MERGING ALL RESULTS")
    print("=" * 80)

    # Load all study result files
    all_data = []
    for f in sorted(OUTPUT_DIR.glob("study*_*.json")) + sorted(OUTPUT_DIR.glob("thinking_ablation_*.json")) + sorted(OUTPUT_DIR.glob("study5_prompt_sensitivity_*.json")) + sorted(OUTPUT_DIR.glob("single_*.json")):
        # Skip checkpoint files
        if "checkpoint" in f.name:
            continue
        with open(f) as fh:
            all_data.extend(json.load(fh))

    if not all_data:
        print("No study result files found!")
        return []

    # Deduplicate by (model_id, seed, thinking_mode, prompt_variant) — prefer later entries
    # This preserves Study 1 and Study 2 data for shared models
    model_seed_map = {}
    for r in all_data:
        key = (r.get("model_id", r.get("model", "")), r["seed"], r.get("thinking_mode", "chat"), r.get("prompt_variant", ""))
        model_seed_map[key] = r

    all_results = sorted(model_seed_map.values(), key=lambda x: (x.get("model_id", x.get("model", "")), x["seed"], x.get("thinking_mode", "chat"), x.get("prompt_variant", "")))
    all_models = sorted(set(r.get("model_id", r.get("model", "")) for r in all_results))

    print(f"Merged: {len(all_results)} observations, {len(all_models)} models")
    print(f"Models: {', '.join(all_models)}")

    # Validate: check for duplicates, missing items, etc.
    issues = []
    for r in all_results:
        mid = r.get("model_id", r.get("model", ""))
        if "items" not in r or not r["items"]:
            issues.append(f"  {mid} seed={r['seed']}: missing 'items' field")

    if issues:
        print(f"\nIssues found ({len(issues)}):")
        for issue in issues:
            print(issue)
    else:
        print("\nAll observations have valid item-level data.")

    # Summary by study
    for study_num in [1, 2, 5]:
        if study_num == 5:
            study_results = [r for r in all_results if r.get("study") == 5]
            print(f"\n--- Study 5 (Prompt Sensitivity): {len(study_results)} observations ---")
            if study_results:
                variants = set(r.get("prompt_variant", "") for r in study_results)
                print(f"  Prompt variants: {', '.join(v for v in variants if v)}")
            continue
        study_models = [m for m, meta in ALL_MODELS.items() if meta.get("study") == study_num]
        study_results = [r for r in all_results if r.get("study") == study_num]
        study_model_ids = set(r.get("model_id", r.get("model", "")) for r in study_results)
        print(f"\n--- Study {study_num}: {len(study_model_ids)}/{len(study_models)} models ---")
        for m in study_models:
            obs = [r for r in all_results if r.get("model_id", r.get("model", "")) == m]
            status = f"{len(obs)} obs" if obs else "MISSING"
            print(f"  {m}: {status}")

    # Save final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = OUTPUT_DIR / f"final_merged_{timestamp}.json"
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal dataset: {final_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model-based LLM personality experiments (SiliconFlow API)")
    parser.add_argument("--study", type=str, default=None,
                        choices=["1", "2", "3", "4", "all"],
                        help="Which study to run")
    parser.add_argument("--pilot", action="store_true",
                        help="Run pilot only (3 models × 10 items × 3 seeds)")
    parser.add_argument("--model", type=str, default=None,
                        help="Run a specific model by SiliconFlow model ID")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all results into final dataset")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--thinking-ablation", action="store_true",
                        help="Run thinking ablation (4 models × chat vs thinking)")
    parser.add_argument("--prompt-sensitivity", action="store_true",
                        help="Run prompt sensitivity study (15 models × 3 variants × 12 seeds)")
    args = parser.parse_args()

    if args.pilot:
        run_pilot()

    elif args.prompt_sensitivity:
        run_prompt_sensitivity(resume=args.resume)

    elif args.thinking_ablation:
        run_thinking_ablation(resume=args.resume)

    elif args.merge:
        merge_results()

    elif args.model:
        if args.model in ALL_MODELS:
            meta = ALL_MODELS[args.model]
            results = run_model(args.model, meta)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = args.model.replace("/", "_")
            output_file = OUTPUT_DIR / f"single_{safe_name}_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved to {output_file}")
        else:
            print(f"Model '{args.model}' not defined.")
            print(f"Available models:\n" + "\n".join(f"  {m}" for m in ALL_MODELS))

    elif args.study:
        if args.study in ("1", "all"):
            run_study(1, STUDY1_MODELS, resume=args.resume)
        if args.study in ("2", "all"):
            run_study(2, STUDY2_MODELS, resume=args.resume)
        if args.study in ("3", "all"):
            run_study(3, STUDY3_MODELS, resume=args.resume)
        if args.study in ("4", "all"):
            run_study(4, STUDY4_MODELS, resume=args.resume)

    else:
        parser.print_help()

    print("\nDone!")
