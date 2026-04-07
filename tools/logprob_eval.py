"""
Logprob-based fast accuracy evaluation tool for SOAR 2026.

Compares a quantized model's output distribution against a BF16 baseline
using KL divergence, cosine similarity, and top-1 agreement on logprobs.
No GPU needed for comparison — only sends HTTP requests to a running server.

Based on the approach described by Week 3 champion Cao Yi.

Usage:
    # Phase 1: collect baseline from BF16 server
    python logprob_eval.py collect --server http://localhost:30000 --out baselines/bf16.npz

    # Phase 2: evaluate quantized model server
    python logprob_eval.py eval --server http://localhost:30000 --baseline baselines/bf16.npz

    # Both phases in one go (two servers running on different ports)
    python logprob_eval.py compare \
        --baseline-server http://localhost:30000 \
        --test-server    http://localhost:30001 \
        --out baselines/bf16.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Fixed prompts — diverse mix matching competition dataset types:
# reasoning, math, long-context, code, factual
# ---------------------------------------------------------------------------
EVAL_PROMPTS = [
    # Reasoning
    "Alice has 3 times as many apples as Bob. Bob has 12 apples. If Alice gives Bob 10 apples, how many does each person have?",
    "A train leaves Chicago at 8am traveling 60 mph toward New York. Another train leaves New York at 9am traveling 80 mph toward Chicago. The cities are 800 miles apart. When do they meet?",
    "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops definitely Lazzles? Explain step by step.",
    "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?",
    # Math
    "Find the derivative of f(x) = x^3 * sin(x) + e^(2x). Show your work.",
    "What is the sum of the first 100 prime numbers?",
    "Solve: 2x^2 - 7x + 3 = 0. Show all steps.",
    "A circle has radius 5. What is the area of the largest rectangle that fits inside it?",
    # Long-context / coherence
    "Summarize the key differences between transformer and state space model architectures for sequence modeling. Cover attention complexity, memory usage, training stability, and inference speed.",
    "Explain how quantization works in neural networks. Cover: (1) what gets quantized, (2) static vs dynamic quantization, (3) FP8 vs INT8 vs FP4, (4) accuracy vs speed tradeoffs.",
    "Describe the history of the Byzantine Empire from its founding to its fall. Include key emperors, territorial changes, and cultural contributions.",
    "What are the main differences between Python's multiprocessing, threading, and asyncio? When should you use each? Give concrete examples.",
    # Code
    "Write a Python function that checks if a string is a valid IPv4 address without using regex.",
    "Implement a binary search tree in Python with insert, search, and in-order traversal methods.",
    "Write a Python decorator that retries a function up to N times with exponential backoff.",
    "Given a list of integers, find the longest increasing subsequence. Implement in Python with O(n log n) complexity.",
    # Factual / knowledge
    "What is the difference between supervised, unsupervised, and reinforcement learning? Give one real-world example of each.",
    "Explain the CAP theorem in distributed systems. What are the tradeoffs between consistency, availability, and partition tolerance?",
    "What is CUDA and how does it enable GPU parallelism? Explain warps, blocks, and grids.",
    "Describe how transformer attention works mathematically. Include the QKV formulation and scaled dot-product attention.",
]

# Sampling params: greedy, return logprobs for last 128 tokens
TOP_LOGPROBS = 256   # SGLang max
MAX_NEW_TOKENS = 256
LOGPROB_TOKENS = 128  # how many output tokens' logprobs to compare


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_model_name(server: str) -> str:
    resp = requests.get(f"{server}/v1/models", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def fetch_logprobs(server: str, prompt: str, model: str) -> dict | None:
    """
    Send one prompt with logprobs enabled.
    Returns the raw API response dict, or None on failure.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
    }
    try:
        resp = requests.post(f"{server}/v1/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [warn] request failed: {e}", file=sys.stderr)
        return None


def extract_token_logprob_matrix(response: dict, vocab_size_approx: int = 32000) -> np.ndarray | None:
    """
    Extract a [T, V] sparse logprob matrix from an API response.
    T = number of output tokens (up to LOGPROB_TOKENS from the end)
    V = vocab_size_approx (we'll use a dict representation instead)

    Returns: list of dicts [{token_id: logprob, ...}] length T
    """
    try:
        choices = response["choices"]
        if not choices:
            return None
        logprobs_data = choices[0].get("logprobs")
        if not logprobs_data:
            return None
        content = logprobs_data.get("content", [])
        if not content:
            return None
        # Take last LOGPROB_TOKENS tokens
        content = content[-LOGPROB_TOKENS:]
        result = []
        for token_info in content:
            top = token_info.get("top_logprobs", [])
            d = {}
            for entry in top:
                tid = entry.get("token")  # token string
                lp = entry.get("logprob", -100.0)
                if tid is not None:
                    d[tid] = lp
            result.append(d)
        return result
    except Exception as e:
        print(f"  [warn] logprob extraction failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Phase 1: Collect baseline
# ---------------------------------------------------------------------------

def collect_baseline(server: str, out_path: Path) -> None:
    print(f"\n[collect] Server: {server}")
    print(f"[collect] Output: {out_path}")
    print(f"[collect] Prompts: {len(EVAL_PROMPTS)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = get_model_name(server)
    print(f"[collect] Model: {model}")

    all_logprobs = []  # list of list-of-dicts

    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"[collect] {i+1}/{len(EVAL_PROMPTS)}: {prompt[:60]}...")
        resp = fetch_logprobs(server, prompt, model)
        if resp is None:
            print(f"  [warn] skipping prompt {i}")
            all_logprobs.append(None)
            continue
        token_logprobs = extract_token_logprob_matrix(resp)
        all_logprobs.append(token_logprobs)
        if token_logprobs:
            print(f"  got {len(token_logprobs)} token positions")

    # Serialize as JSON inside npz (numpy has no native dict storage)
    serialized = json.dumps(all_logprobs)
    np.savez_compressed(
        out_path,
        logprobs_json=np.array([serialized]),
        prompts=np.array(EVAL_PROMPTS),
        server=np.array([server]),
        model=np.array([model]),
    )
    valid = sum(1 for x in all_logprobs if x is not None)
    print(f"\n[collect] Done. {valid}/{len(EVAL_PROMPTS)} prompts collected.")
    print(f"[collect] Saved to {out_path}")


# ---------------------------------------------------------------------------
# Phase 2: Evaluate against baseline
# ---------------------------------------------------------------------------

def compute_metrics(
    baseline_dicts: list[dict],
    test_dicts: list[dict],
) -> dict:
    """
    Compare two sequences of {token: logprob} dicts.
    Returns KL divergence, cosine similarity, top-1 agreement.
    """
    kl_vals = []
    cos_vals = []
    top1_matches = 0
    total = 0

    for b_dict, t_dict in zip(baseline_dicts, test_dicts):
        if not b_dict or not t_dict:
            continue

        # Union of tokens seen in either distribution
        all_tokens = set(b_dict.keys()) | set(t_dict.keys())
        # Small smoothing mass for unseen tokens
        smooth = -20.0  # logprob ~= 2e-9 for unseen

        b_lps = np.array([b_dict.get(tok, smooth) for tok in all_tokens], dtype=np.float64)
        t_lps = np.array([t_dict.get(tok, smooth) for tok in all_tokens], dtype=np.float64)

        # Convert to probabilities (numerically stable)
        b_lps -= np.logaddexp.reduce(b_lps)  # renormalize
        t_lps -= np.logaddexp.reduce(t_lps)

        b_probs = np.exp(b_lps)
        t_probs = np.exp(t_lps)

        # KL(baseline || test)  — how much test diverges from baseline
        # Only where baseline > 0
        mask = b_probs > 1e-10
        kl = float(np.sum(b_probs[mask] * (b_lps[mask] - t_lps[mask])))
        kl_vals.append(kl)

        # Cosine similarity between logprob vectors
        dot = float(np.dot(b_lps, t_lps))
        norm_b = float(np.linalg.norm(b_lps))
        norm_t = float(np.linalg.norm(t_lps))
        cos = dot / (norm_b * norm_t + 1e-10)
        cos_vals.append(cos)

        # Top-1 agreement
        b_top1 = max(b_dict, key=b_dict.get)
        t_top1 = max(t_dict, key=t_dict.get)
        top1_matches += int(b_top1 == t_top1)
        total += 1

    return {
        "kl_mean":    float(np.mean(kl_vals))  if kl_vals  else float("nan"),
        "kl_p95":     float(np.percentile(kl_vals, 95)) if kl_vals else float("nan"),
        "cos_mean":   float(np.mean(cos_vals)) if cos_vals else float("nan"),
        "cos_min":    float(np.min(cos_vals))  if cos_vals else float("nan"),
        "top1_agree": top1_matches / total     if total > 0 else float("nan"),
        "n_tokens":   total,
    }


def grade_metrics(m: dict) -> str:
    kl   = m["kl_mean"]
    cos  = m["cos_mean"]
    top1 = m["top1_agree"]
    if kl < 0.005 and cos > 0.9995 and top1 > 0.97:
        return "EXCELLENT — negligible quantization loss"
    if kl < 0.02 and cos > 0.999 and top1 > 0.95:
        return "GOOD — safe to submit"
    if kl < 0.1 and cos > 0.99 and top1 > 0.90:
        return "MARGINAL — accuracy penalty possible"
    return "POOR — likely to fail accuracy threshold"


def evaluate(server: str, baseline_path: Path) -> dict:
    print(f"\n[eval] Server:   {server}")
    print(f"[eval] Baseline: {baseline_path}")

    data = np.load(baseline_path, allow_pickle=True)
    baseline_logprobs = json.loads(str(data["logprobs_json"][0]))
    baseline_prompts  = list(data["prompts"])

    model = get_model_name(server)
    print(f"[eval] Model:    {model}")

    all_metrics = []

    for i, (prompt, baseline_toks) in enumerate(zip(baseline_prompts, baseline_logprobs)):
        print(f"[eval] {i+1}/{len(baseline_prompts)}: {prompt[:60]}...")
        if baseline_toks is None:
            print("  [skip] no baseline for this prompt")
            continue

        resp = fetch_logprobs(server, prompt, model)
        if resp is None:
            continue
        test_toks = extract_token_logprob_matrix(resp)
        if test_toks is None:
            continue

        m = compute_metrics(baseline_toks, test_toks)
        all_metrics.append(m)
        print(f"  KL={m['kl_mean']:.4f}  cos={m['cos_mean']:.4f}  top1={m['top1_agree']:.3f}  n={m['n_tokens']}")

    if not all_metrics:
        print("[eval] No metrics collected.")
        return {}

    # Aggregate across all prompts
    aggregate = {
        "kl_mean":    float(np.mean([m["kl_mean"]    for m in all_metrics])),
        "kl_p95":     float(np.percentile([m["kl_mean"] for m in all_metrics], 95)),
        "cos_mean":   float(np.mean([m["cos_mean"]   for m in all_metrics])),
        "cos_min":    float(np.min( [m["cos_min"]    for m in all_metrics])),
        "top1_agree": float(np.mean([m["top1_agree"] for m in all_metrics])),
        "n_prompts":  len(all_metrics),
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  KL divergence (mean):  {aggregate['kl_mean']:.5f}  (p95: {aggregate['kl_p95']:.5f})")
    print(f"  Cosine similarity:     {aggregate['cos_mean']:.5f}  (min: {aggregate['cos_min']:.5f})")
    print(f"  Top-1 agreement:       {aggregate['top1_agree']:.4f}  ({aggregate['top1_agree']*100:.1f}%)")
    print(f"  Prompts evaluated:     {aggregate['n_prompts']}")
    print()
    verdict = grade_metrics(aggregate)
    print(f"  Verdict: {verdict}")
    print("=" * 60)
    return aggregate


# ---------------------------------------------------------------------------
# Phase 1+2 combined: two servers side by side
# ---------------------------------------------------------------------------

def compare(baseline_server: str, test_server: str, out_path: Path) -> None:
    collect_baseline(baseline_server, out_path)
    evaluate(test_server, out_path)


# ---------------------------------------------------------------------------
# Modal wrapper — run against servers launched inside Modal
# ---------------------------------------------------------------------------

def run_in_modal(server: str, baseline_path: str) -> dict:
    """Import and call from modal_app.py test functions."""
    return evaluate(server, Path(baseline_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Logprob-based quantization accuracy evaluator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # collect
    p_collect = sub.add_parser("collect", help="Collect BF16 baseline logprobs")
    p_collect.add_argument("--server", default="http://localhost:30000")
    p_collect.add_argument("--out",    default="baselines/bf16.npz")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate quantized model against baseline")
    p_eval.add_argument("--server",   default="http://localhost:30000")
    p_eval.add_argument("--baseline", default="baselines/bf16.npz")

    # compare (both phases)
    p_cmp = sub.add_parser("compare", help="Collect baseline + evaluate (two servers)")
    p_cmp.add_argument("--baseline-server", required=True)
    p_cmp.add_argument("--test-server",     required=True)
    p_cmp.add_argument("--out", default="baselines/bf16.npz")

    args = parser.parse_args()

    if args.cmd == "collect":
        collect_baseline(args.server, Path(args.out))
    elif args.cmd == "eval":
        evaluate(args.server, Path(args.baseline))
    elif args.cmd == "compare":
        compare(args.baseline_server, args.test_server, Path(args.out))


if __name__ == "__main__":
    main()
