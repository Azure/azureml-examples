"""Helper functions for generating Model Router training data.

This module handles parallel LLM calls and judge-based metric generation.
"""

import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional


def call_llm(
    endpoint: str,
    api_key: str,
    deployment_name: str,
    messages: List[Dict[str, str]],
    model_name: str,
    model_version: str,
) -> Dict[str, Any]:
    """Call a specific LLM through the model-router deployment using routing headers.

    Args:
        endpoint: Azure AI Foundry project endpoint URL.
        api_key: API key for the endpoint.
        deployment_name: The model-router deployment name.
        messages: Chat messages to send.
        model_name: Model name to route to via AGENT_X_MODEL_ROUTER_SELECTED_MODEL header.
        model_version: Model version via AGENT_X_MODEL_ROUTER_SELECTED_MODEL_VERSION header.

    Returns:
        dict with keys: "content", "completion_tokens", "prompt_tokens", "model"
    """
    url = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2025-04-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
        "AGENT_X_MODEL_ROUTER_SELECTED_MODEL": model_name,
        "AGENT_X_MODEL_ROUTER_SELECTED_MODEL_VERSION": model_version,
    }
    payload = {"messages": messages}

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    result = resp.json()

    choice = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})

    return {
        "content": choice,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "model": model_name,
    }


def call_all_llms_for_prompt(
    endpoint: str,
    api_key: str,
    deployment_name: str,
    prompt: str,
    llm_list: List[Dict[str, str]],
    max_workers: int = 5,
) -> List[Dict[str, Any]]:
    """Call all LLMs in parallel for a single prompt.

    Args:
        endpoint: Azure AI Foundry project endpoint URL.
        api_key: API key for the endpoint.
        deployment_name: The model-router deployment name.
        prompt: The user prompt to send to each LLM.
        llm_list: List of dicts with "name" and "version" keys.
        max_workers: Max parallel threads. Default 5.

    Returns:
        List of response dicts from each LLM.
    """
    messages = [{"role": "user", "content": prompt}]
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                call_llm, endpoint, api_key, deployment_name, messages, llm["name"], llm["version"]
            ): llm
            for llm in llm_list
        }
        for future in as_completed(futures):
            llm = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error calling {llm['name']}: {e}")
                results.append({
                    "content": None,
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "model": llm["name"],
                    "error": str(e),
                })

    return results


def judge_responses(
    endpoint: str,
    api_key: str,
    deployment_name: str,
    judge_model_name: str,
    judge_model_version: str,
    prompt: str,
    responses: List[Dict[str, Any]],
    judge_prompt_template: str,
) -> Dict[str, int]:
    """Use a judge LLM to score each model's response.

    Args:
        endpoint: Azure AI Foundry project endpoint URL.
        api_key: API key for the endpoint.
        deployment_name: The model-router deployment name.
        judge_model_name: Model name to use as judge.
        judge_model_version: Version of the judge model.
        prompt: The original user prompt.
        responses: List of response dicts from each LLM.
        judge_prompt_template: Template string for the judge prompt.
            Must contain {prompt} and {responses} placeholders.

    Returns:
        dict mapping model_name -> binary score (1 = correct/good, 0 = incorrect/bad).
    """
    # Format responses for the judge
    responses_text = ""
    for i, r in enumerate(responses):
        if r.get("content") is not None:
            responses_text += f"\n--- Model: {r['model']} ---\n{r['content']}\n"

    judge_prompt = judge_prompt_template.format(prompt=prompt, responses=responses_text)
    messages = [{"role": "user", "content": judge_prompt}]

    result = call_llm(
        endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        messages=messages,
        model_name=judge_model_name,
        model_version=judge_model_version,
    )

    # Parse judge response as JSON
    try:
        scores = json.loads(result["content"])
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        content = result["content"]
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            scores = json.loads(content[start:end])
        else:
            print(f"Warning: Could not parse judge response as JSON: {content[:200]}")
            scores = {r["model"]: 0 for r in responses}

    return scores


def generate_training_record(
    prompt: str,
    responses: List[Dict[str, Any]],
    metrics: Dict[str, int],
) -> Dict[str, Any]:
    """Generate a single training record in the format expected by the finetuning notebook.

    Args:
        prompt: The original user prompt.
        responses: List of response dicts from each LLM.
        metrics: dict mapping model_name -> binary score.

    Returns:
        dict with "messages", "metrics", and "usage" fields.
    """
    usage = {}
    for r in responses:
        model_name = r["model"]
        usage[model_name] = {
            "completion_tokens": r["completion_tokens"],
            "prompt_tokens": r["prompt_tokens"],
        }

    return {
        "messages": [{"role": "user", "content": prompt}],
        "metrics": metrics,
        "usage": usage,
    }


def process_prompts(
    endpoint: str,
    api_key: str,
    deployment_name: str,
    prompts: List[str],
    llm_list: List[Dict[str, str]],
    judge_model_name: str,
    judge_model_version: str,
    judge_prompt_template: str,
    max_workers: int = 5,
    output_path: str = "training_data.jsonl",
) -> str:
    """Process all prompts end-to-end: call LLMs, judge, and write training data.

    Args:
        endpoint: Azure AI Foundry project endpoint URL.
        api_key: API key for the endpoint.
        deployment_name: The model-router deployment name.
        prompts: List of user prompts.
        llm_list: List of dicts with "name" and "version" keys.
        judge_model_name: Model name to use as judge.
        judge_model_version: Version of the judge model.
        judge_prompt_template: Template for the judge prompt.
        max_workers: Max parallel threads per prompt. Default 5.
        output_path: Path to write the output JSONL file.

    Returns:
        str: Path to the generated JSONL file.
    """
    records = []
    total = len(prompts)

    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i + 1}/{total}...")

        # Call all LLMs in parallel for this prompt
        responses = call_all_llms_for_prompt(
            endpoint=endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            prompt=prompt,
            llm_list=llm_list,
            max_workers=max_workers,
        )

        # Filter out failed responses for judging
        valid_responses = [r for r in responses if r.get("content") is not None]
        if not valid_responses:
            print(f"  Skipping prompt {i + 1}: all LLM calls failed.")
            continue

        # Judge the responses
        metrics = judge_responses(
            endpoint=endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            judge_model_name=judge_model_name,
            judge_model_version=judge_model_version,
            prompt=prompt,
            responses=valid_responses,
            judge_prompt_template=judge_prompt_template,
        )

        # Build training record
        record = generate_training_record(prompt, responses, metrics)
        records.append(record)

    # Write to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(records)} training records -> {output_path}")
    return output_path
