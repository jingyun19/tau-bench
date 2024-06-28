
# util class to save all the tools information, so that they can be programmatically added to DF agent.  
# To add tools, use https://colab.corp.google.com/drive/1G-vNeGxGOzVtd4HgGinQ31dWweXkfr6G?resourcekey=0-j5Vw2x5rHYN_gE1AOyVQGw#scrollTo=6TubplLGqzlN

from tau_bench.envs import get_env
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import multiprocessing
import os
import random


tool_output_schema = {
    "properties": {
        "result": {
            "type": "string"
        }
    },
    "type": "object"
}


def run(
    args: argparse.Namespace,
):
  env = get_env(
      args.env,
      user_mode="naive",
      user_model="gemini-1.5-pro",
      task_split=args.task_split,
  )
  requests = []
  for tool in env.tools:
    # All tools are "function" types.
    function_tool = tool.__info__["function"]
    tool_name = function_tool["name"]
    description = function_tool["description"]
    params = function_tool["parameters"]
    tool_request = {"display_name": tool_name, "description": description, "input_schema":params, "output_schema": tool_output_schema}
    requests.append(tool_request)
    with open(args.env+"_tools_info.txt", "w") as f:
        f.write(json.dumps(requests))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--env", type=str, choices=["retail", "airline"], default="retail"
  )
  parser.add_argument(
      "--task_split", type=str, default="test", choices=["train", "test", "dev"]
  )
  args = parser.parse_args()
  print(args)
  run(args=args)
if __name__ == "__main__":
  main()
