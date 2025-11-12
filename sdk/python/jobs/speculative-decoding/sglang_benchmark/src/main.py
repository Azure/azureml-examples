import argparse
import os
import sys
import json

from bench_serving import run_benchmark
from helper import get_api_key_from_connection, log_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="SGLang Benchmarking")
    # parser.add_argument(
    #     "--metrics_path",
    #     type=str,
    #     required=True,
    #     help="Output JSON file to store the benchmarking metrics.",
    # )
    parser.add_argument(
        "--connection-name",
        type=str,
        required=True,
        help="The name of the workspace connection used to fetch API key.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials to run the benchmark, result will be averaged over all trials.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sglang",
        help="Must specify a backend, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=[
            "sharegpt",
            "random",
            "generated-shared-prefix",
            "ultrachat",
            "loogle",
            "nextqa",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer. If not set, using the model conf.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--fixed-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the dataset.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        default=1024,
        type=int,
        help="Number of output tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use request rate range rather than single value.",
    )
    parser.add_argument(
        "--request-rate-range",
        type=str,
        default="2,34,2",
        help="Range of request rates in the format start,stop,step. Default is 2,34,2. It also supports a list of request rates, requiring the parameters to not equal three.",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument(
        "--enable-multiturn",
        action="store_true",
        help="Enable multiturn chat for online serving benchmarking. "
        "This option is effective on the following datasets: "
        "sharegpt, ultrachat, loogle, nextqa",
    )
    parser.add_argument(
        "--enable-shared-prefix",
        action="store_true",
        help="Enable shared prefix for online serving benchmarking. "
        "This option is effective on the following datasets: "
        "loogle, nextqa",
    )
    parser.add_argument(
        "--disable-shuffle",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Disable shuffling datasets. Accepts true/false.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--disable-stream",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--return-logprob",
        action="store_true",
        help="Return logprob.",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Disable ignoring EOS.",
    )
    parser.add_argument(
        "--extra-request-body",
        metavar='{"key1": "value1", "key2": "value2"}',
        type=str,
        help="Append given JSON object to the request payload. You can use this to specify"
        "additional generate params like sampling params.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default=None,
        help="The name of LoRA adapter",
    )

    group = parser.add_argument_group("generated-shared-prefix dataset arguments")
    group.add_argument(
        "--gsp-num-groups",
        type=int,
        default=64,
        help="Number of system prompt groups for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-prompts-per-group",
        type=int,
        default=16,
        help="Number of prompts per system prompt group for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-system-prompt-len",
        type=int,
        default=2048,
        help="Target length in tokens for system prompts in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-question-len",
        type=int,
        default=128,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-output-len",
        type=int,
        default=256,
        help="Target length in tokens for outputs in generated-shared-prefix dataset",
    )
    # videos specific
    parser.add_argument(
        "--max-frames",
        type=int,
        default=sys.maxsize,
        help="The maximum number of frames to extract from each video. "
        "This option is specific to the nextqa dataset (video benchmark). ",
    )
    args = parser.parse_args()
    return args


def _generate_avg_metrics(metrics_file: str):
    metrics_list = []
    with open(metrics_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics_list.append(json.loads(line))

    # Compute average metrics
    avg_metrics = {}
    count = len(metrics_list)

    for key in metrics_list[0].keys():
        if isinstance(metrics_list[0][key], (int, float)):
            avg_metrics[key] = sum(result[key] for result in metrics_list) / count
        else:
            avg_metrics[key] = metrics_list[0][key]

    with open(os.path.join(os.path.dirname(metrics_file), "metrics_avg.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)

    log_metrics(avg_metrics)


def main():
    args = parse_args()

    api_key, _ = get_api_key_from_connection(args.connection_name)
    del args.connection_name
    os.environ["OPENAI_API_KEY"] = api_key

    args.output_file = os.path.join(args.output_file, "metrics_each_trial.jsonl")

    trials = args.trials
    del args.trials
    
    for trial in range(trials):
        print(f"Starting trial {trial + 1} of {trials}...")
        run_benchmark(args)
    
    _generate_avg_metrics(args.output_file)


if __name__ == "__main__":
    main()