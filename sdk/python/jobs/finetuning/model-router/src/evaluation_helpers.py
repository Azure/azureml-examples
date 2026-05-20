"""Helper functions for evaluating fine-tuned vs base model router deployments."""

import glob
import hashlib
import json
import os
import re
import requests
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# RPM limits per deployment type
BASE_RPM = 250
FT_RPM = 100

# Pricing per 1M tokens (USD)
MODEL_PRICING = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
}


def load_test_files(data_folder: str, pattern: str = "*_test_*.jsonl") -> Dict[str, List[dict]]:
    """Load test JSONL files matching a glob pattern.

    Args:
        data_folder: Path to the data folder.
        pattern: Glob pattern to match test files.

    Returns:
        dict mapping filename -> list of records.
    """
    files = glob.glob(os.path.join(data_folder, pattern))
    datasets = {}
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        datasets[fname] = records
    return datasets


def run_inference(
    endpoint: str,
    api_key: str,
    deployment_name: str,
    messages: List[dict],
) -> dict:
    """Run chat completions inference on a deployment.

    Args:
        endpoint: Azure AI Foundry project endpoint.
        api_key: API key.
        deployment_name: Deployment name.
        messages: Chat messages.

    Returns:
        dict with "model", "content", "prompt_tokens", "completion_tokens".
    """
    url = (
        f"{endpoint}/openai/deployments/{deployment_name}"
        f"/chat/completions?api-version=2025-01-01-preview"
    )
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {"messages": messages}

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    result = resp.json()

    usage = result.get("usage", {})
    model = result.get("model", deployment_name)

    # Normalize model name: API returns "gpt-5-mini-2025-08-07" but metrics
    # use "gpt-5-mini_2025-08-07" (underscore before date). Convert last "-"
    # before a date pattern to "_".
    import re
    model = re.sub(r"-(\d{4}-\d{2}-\d{2})", r"_\1", model)

    return {
        "model": model,
        "content": result["choices"][0]["message"]["content"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def _cache_path(cache_dir: str, deployment_name: str, filename: str) -> str:
    """Return path to the cache file for a deployment + test file combination."""
    safe_name = f"{deployment_name}_{filename}".replace("/", "_").replace("\\", "_")
    return os.path.join(cache_dir, f"{safe_name}.cache.json")


def _load_cache(cache_file: str) -> List[Optional[dict]]:
    """Load cached results. Returns list with None for uncached entries."""
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_cache(cache_file: str, results: List[Optional[dict]]) -> None:
    """Save results to cache file."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(results, f)


class _RateLimiter:
    """Simple token-bucket rate limiter for RPM."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


def evaluate_deployment(
    endpoint: str,
    api_key: str,
    deployment_name: str,
    records: List[dict],
    label: str = "",
    cache_dir: str = "",
    filename: str = "",
    max_workers: int = 10,
    rpm: int = 250,
) -> List[dict]:
    """Run inference for all records using thread pool, with rate limiting and caching.

    Args:
        endpoint: Azure AI Foundry project endpoint.
        api_key: API key.
        deployment_name: Deployment name.
        records: List of test records with "messages" and "metrics" keys.
        label: Label for progress printing.
        cache_dir: Directory to store cache files. Empty string disables caching.
        filename: Test filename (used as cache key).
        max_workers: Number of concurrent threads.
        rpm: Requests per minute limit.

    Returns:
        List of dicts with "routed_model", "prompt_tokens", "completion_tokens",
        "correct" (whether the routed model has metric==1 in ground truth).
    """
    total = len(records)

    # Load cache
    cached = [None] * total
    cache_file = ""
    if cache_dir and filename:
        cache_file = _cache_path(cache_dir, deployment_name, filename)
        prev = _load_cache(cache_file)
        for i in range(min(len(prev), total)):
            cached[i] = prev[i]

    cached_count = sum(1 for c in cached if c is not None)

    # Identify records that need inference
    pending = [(i, records[i]) for i in range(total) if cached[i] is None]
    if not pending:
        return cached

    rate_limiter = _RateLimiter(rpm)
    cache_lock = threading.Lock()
    completed = [0]

    def _infer(idx: int, record: dict) -> None:
        messages = record["messages"]
        metrics = record.get("metrics", {})

        rate_limiter.acquire()
        try:
            resp = run_inference(endpoint, api_key, deployment_name, messages)
            routed_model = resp["model"]
            correct = metrics.get(routed_model, 0) == 1
            result = {
                "routed_model": routed_model,
                "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"],
                "correct": correct,
            }
        except Exception as e:
            result = {
                "routed_model": "error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "correct": False,
            }

        with cache_lock:
            cached[idx] = result
            completed[0] += 1
            if cache_file:
                _save_cache(cache_file, cached)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_infer, idx, rec) for idx, rec in pending]
        for f in as_completed(futures):
            f.result()  # propagate any unexpected exceptions

    return cached


def compute_model_distribution(results: List[dict]) -> Dict[str, dict]:
    """Compute model routing distribution from evaluation results.

    Args:
        results: List of per-record evaluation results.

    Returns:
        dict mapping model_name -> {"count": int, "percentage": float}.
    """
    counts = defaultdict(int)
    total = len(results)
    for r in results:
        counts[r["routed_model"]] += 1

    distribution = {}
    for model, count in sorted(counts.items(), key=lambda x: -x[1]):
        distribution[model] = {
            "count": count,
            "percentage": round(100.0 * count / total, 1) if total > 0 else 0,
        }
    return distribution


def _get_model_base_name(routed_model: str) -> str:
    """Extract base model name (without version) for pricing lookup.

    e.g. 'gpt-5-mini_2025-08-07' -> 'gpt-5-mini', 'gpt-5_2025-08-07' -> 'gpt-5'
    """
    # Remove version suffix (underscore + date)
    parts = routed_model.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else routed_model


def _compute_record_cost(routed_model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute USD cost for a single record based on model pricing."""
    base_name = _get_model_base_name(routed_model)
    pricing = MODEL_PRICING.get(base_name)
    if not pricing:
        # Fall back to GPT-5 pricing for unknown models
        pricing = MODEL_PRICING["gpt-5"]
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def compute_accuracy_and_cost(
    results: List[dict],
    records: List[dict],
) -> dict:
    """Compute accuracy and dollar cost from evaluation results.

    Uses token counts from the test file's usage field (per-model ground truth)
    rather than inference response tokens for cost calculation.

    Args:
        results: List of per-record evaluation results.
        records: Original test records (with ground-truth usage).

    Returns:
        dict with accuracy, token counts, and actual dollar cost.
    """
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    # Compute actual cost using test file usage tokens for the routed model
    actual_cost = 0.0
    gpt5_cost = 0.0
    for r, rec in zip(results, records):
        routed_model = r["routed_model"]
        usage = rec.get("usage", {})

        # Get tokens from test file for the routed model
        model_usage = usage.get(routed_model, {})
        prompt_tokens = model_usage.get("prompt_tokens", r["prompt_tokens"])
        completion_tokens = model_usage.get("completion_tokens", r["completion_tokens"])
        actual_cost += _compute_record_cost(routed_model, prompt_tokens, completion_tokens)

        # GPT-5 cost: use GPT-5 tokens from test file usage
        gpt5_key = [k for k in usage if k.startswith("gpt-5_")]
        if gpt5_key:
            gpt5_usage = usage[gpt5_key[0]]
            gpt5_cost += _compute_record_cost(
                "gpt-5", gpt5_usage["prompt_tokens"], gpt5_usage["completion_tokens"]
            )
        else:
            # Fallback: use routed model tokens with GPT-5 pricing
            gpt5_cost += _compute_record_cost("gpt-5", prompt_tokens, completion_tokens)

    return {
        "accuracy": round(100.0 * correct / total, 1) if total > 0 else 0,
        "correct": correct,
        "total": total,
        "actual_cost": actual_cost,
        "gpt5_cost": gpt5_cost,
    }


def compute_cost_savings(base_stats: dict, ft_stats: dict) -> dict:
    """Compute cost savings of fine-tuned model vs base model.

    Args:
        base_stats: Stats dict from compute_accuracy_and_cost for base model.
        ft_stats: Stats dict from compute_accuracy_and_cost for fine-tuned model.

    Returns:
        dict with savings metrics.
    """
    base_tokens = base_stats["total_tokens"]
    ft_tokens = ft_stats["total_tokens"]
    token_savings = base_tokens - ft_tokens
    token_savings_pct = round(100.0 * token_savings / base_tokens, 1) if base_tokens > 0 else 0

    return {
        "base_total_tokens": base_tokens,
        "finetuned_total_tokens": ft_tokens,
        "token_savings": token_savings,
        "token_savings_pct": token_savings_pct,
        "base_accuracy": base_stats["accuracy"],
        "finetuned_accuracy": ft_stats["accuracy"],
        "accuracy_diff": round(ft_stats["accuracy"] - base_stats["accuracy"], 1),
    }


def format_distribution_table(
    base_dist: Dict[str, dict],
    ft_dist: Dict[str, dict],
) -> str:
    """Format model distribution table with deployments as rows and models as columns."""
    all_models = sorted(set(list(base_dist.keys()) + list(ft_dist.keys())))

    col_width = max((len(m) for m in all_models), default=10) + 2
    col_width = max(col_width, 12)

    header = f"{'Deployment':<20}"
    for model in all_models:
        header += f" {model:>{col_width}}"
    sep = "-" * len(header)
    lines = [header, sep]

    for label, dist in [("Base Model", base_dist), ("Fine-Tuned Model", ft_dist)]:
        row = f"{label:<20}"
        for model in all_models:
            d = dist.get(model, {"count": 0, "percentage": 0})
            row += f" {d['percentage']:>{col_width - 1}.1f}%"
        lines.append(row)

    return "\n".join(lines)


def format_accuracy_cost_table(
    base_stats: dict,
    ft_stats: dict,
) -> str:
    """Format accuracy and cost table with deployments as rows.

    Columns: Accuracy, Actual Cost, Cost Saving vs GPT-5.
    Cost saving is based on actual model pricing vs GPT-5-only pricing.
    """
    header = f"{'Deployment':<20} {'Accuracy':>12} {'Actual Cost':>15} {'Cost Saving vs GPT-5':>22}"
    sep = "-" * len(header)
    lines = [header, sep]

    for label, stats in [("Base Model", base_stats), ("Fine-Tuned Model", ft_stats)]:
        gpt5_cost = stats["gpt5_cost"]
        actual_cost = stats["actual_cost"]
        saving = round(100.0 * (gpt5_cost - actual_cost) / gpt5_cost, 1) if gpt5_cost > 0 else 0
        lines.append(f"{label:<20} {stats['accuracy']:>11.1f}% {f'${actual_cost:.4f}':>15} {saving:>21.1f}%")

    return "\n".join(lines)


def evaluate_and_compare(
    endpoint: str,
    api_key: str,
    base_deployment: str,
    ft_deployment: str,
    data_folder: str = "data",
    test_pattern: str = "*_test_*.jsonl",
    cache_dir: str = ".eval_cache",
) -> None:
    """Run end-to-end evaluation comparing base vs fine-tuned model router.

    Loads all test files matching the pattern, runs inference on both deployments,
    and prints per-file tables:
    - First files (admin, medical): distribution table showing which LLMs each router selects.
    - Last file (mix): accuracy & cost table showing accuracy and cost savings.

    Results are cached per deployment+file so reruns skip already-completed records.

    Args:
        endpoint: Model router endpoint (e.g. https://<resource>.openai.azure.com).
        api_key: API key.
        base_deployment: Base model router deployment name.
        ft_deployment: Fine-tuned model router deployment name.
        data_folder: Path to the data folder.
        test_pattern: Glob pattern to match test files.
        cache_dir: Directory to store inference cache. Set to "" to disable.
    """
    # Map filenames to friendly scenario names
    SCENARIO_NAMES = {
        "contoso_clinic_test_admin_clean_upload.jsonl": "General Query Testset",
        "contoso_clinic_test_medical_upload.jsonl": "Medical Query Testset",
        "contoso_clinic_test_mix_upload.jsonl": "Prod Traffic Sample Testset",
    }

    # Ordered list: medical first, general second, prod last
    FILE_ORDER = [
        "contoso_clinic_test_medical_upload.jsonl",
        "contoso_clinic_test_admin_clean_upload.jsonl",
        "contoso_clinic_test_mix_upload.jsonl",
    ]

    if cache_dir and not os.path.isabs(cache_dir):
        cache_dir = os.path.join(os.path.abspath(data_folder), "..", cache_dir)

    test_datasets = load_test_files(data_folder, test_pattern)
    # Sort filenames per desired order; any unknown files go at the end
    filenames = sorted(test_datasets.keys(), key=lambda f: FILE_ORDER.index(f) if f in FILE_ORDER else len(FILE_ORDER))

    for idx, filename in enumerate(filenames):
        records = test_datasets[filename]
        is_last_file = (idx == len(filenames) - 1)
        scenario = SCENARIO_NAMES.get(filename, filename)

        # Evaluate base model (250 RPM, more threads)
        base_results = evaluate_deployment(
            endpoint, api_key, base_deployment, records, "Base",
            cache_dir=cache_dir, filename=filename,
            max_workers=50, rpm=BASE_RPM,
        )

        # Evaluate fine-tuned model (100 RPM, fewer threads)
        ft_results = evaluate_deployment(
            endpoint, api_key, ft_deployment, records, "FT",
            cache_dir=cache_dir, filename=filename,
            max_workers=20, rpm=FT_RPM,
        )

        # Compute metrics
        base_dist = compute_model_distribution(base_results)
        ft_dist = compute_model_distribution(ft_results)
        base_stats = compute_accuracy_and_cost(base_results, records)
        ft_stats = compute_accuracy_and_cost(ft_results, records)

        print(f"\n{scenario}")
        print(f"{'─'*len(scenario)}")

        if is_last_file:
            print(format_accuracy_cost_table(base_stats, ft_stats))
        else:
            print(format_distribution_table(base_dist, ft_dist))

        print()
