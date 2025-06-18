# Copyright Sierra

import argparse
from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy

from tau_bench.agents.base import Agent as BaseAgent
import google.generativeai as genai
import os
import json
import multiprocessing
import random
from tau_bench.envs import get_env
from tau_bench.run import agent_factory
from tau_bench.run import run

# DEBUG only
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False
def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        choices=[
            # openai api models
            "gpt-4-turbo",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
            "gpt-4o",
            # anthropic api models
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            # google api models
            "gemini-2.5-pro-preview-06-05",
            'gemini-2.5-flash-preview-05-20',
            "gemini-2.0-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-1.0-pro",
            "gemini-pro",
            # mistral api models,
            "open-mixtral-8x22b",
            "mistral-large-latest",
            # anyscale api models
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ],
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot", "decibel"],
    )
    parser.add_argument("--agent_id", type=str, default="ae1ef45c-4d14-49b0-ab3e-2ab7d89d91bb") # 2.0: b469add4-b544-4dcb-93c4-78245eacac7a
    parser.add_argument("--project_id", type=str, default="df-decibel2-dev-test")
    parser.add_argument("--gcp_sa_file", type=str, default="df-decibel2-dev-test-3ae2afb1e27f.json")
    parser.add_argument(
        "--task_split", type=str, default="test", choices=["train", "test", "dev"]
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, help="Path to a jsonlines file containing few shot displays")
    args = parser.parse_args()
    print(args)
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        agent_id=args.agent_id, 
        project_id=args.project_id,
        gcp_sa_file=args.gcp_sa_file,
    )


def main():
    config = parse_args()
    run(config)


if __name__ == "__main__":
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    main()