# τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains
Paper: https://arxiv.org/abs/2406.12045

# install DF CX SDK

```
gsutil cp gs://agent-evals/v3alpha1_dialogflow-v3alpha1-py.tar /content/v3alpha1_dialogflow-v3alpha1-py.tar
tar -xvf content/v3alpha1_dialogflow-v3alpha1-py.tar
venv/bin/python3 dialogflow-v3alpha1-py/setup.py sdist
venv/bin/pip install content/v3alpha1_dialogflow-v3alpha1-py.tar

venv/bin/pip install google-cloud-dialogflow-cx
```

## Setup

1. Clone this repository:

```bash
git clone https://github.com/sierra-research/tau-bench && cd ./tau-bench
```

2. Install from source (which also installs required packages):

```bash
pip install -e .
```

3. Set up your OpenAI / Anthropic / Google / Mistral / AnyScale API keys as environment variables.

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
ANYSCALE_API_KEY=...
```


## Run
Run a function calling agent on the τ-retail environment:

```bash
python run.py --env retail --model gpt-4o --max_concurrency 10
```

Set max concurrency according to your API limit.

Run a decibel agent with gemini as user model: 

```
venv/bin/python3 tau-bench/run.py --env retail  --agent_strategy decibel --agent_id 429da584-b933-4372-822c-52d124ba5a26 --project_id df-decibel2-dev-test  --start_index 0 --end_index -1  --user_model gemini-1.5-pro
```
