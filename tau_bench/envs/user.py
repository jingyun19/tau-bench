# Copyright Sierra

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from google.generativeai import GenerativeModel, GenerationConfig
from google.generativeai import protos


class BaseUserSimulationEnv:
    metadata = {}

    def reset(self, instruction=None) -> str:
        return ""

    def step(self, content: str) -> str:
        return ""

    def get_total_cost(self):
        return 0


class HumanUserSimulationEnv(BaseUserSimulationEnv):
    def reset(self, instruction=None) -> str:
        return input(f"{instruction}\n")

    def step(self, content: str) -> str:
        return input(f"{content}\n")


# client = OpenAI()

prompt_price_per_million = {"gpt-4o": 5, "gpt-4-turbo": 10, "gpt-4": 30, "gpt-4-32k-0613": 60, "gpt-3.5-turbo": 0.5, "gemini-1.5-pro": 3.5}
completion_price_per_million = {"gpt-4o": 15, "gpt-4-turbo": 30, "gpt-4": 60, "gpt-4-32k-0613": 120, "gpt-3.5-turbo": 1.5, "gemini-1.5-pro": 10.5}
safety = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
]

# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, model="gpt-4", temperature=1.0, max_output_tokens=150):
    if not (model in prompt_price_per_million and  model in completion_price_per_million):
        print(f"Model [{model}] price missing!")
    if "gpt" in model:
        client = OpenAI()
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        content = response.choices[0].message.content
        cost = prompt_price_per_million[model] * response.usage.prompt_tokens / 1e6 + completion_price_per_million[model] * response.usage.completion_tokens / 1e6
    elif "gemini" in model:
        client = GenerativeModel(model_name=model, system_instruction=messages[0]['content'])
        contents = []
        for m in messages[1:]:
            contents.append({"role":{"user": "user", "assistant": "model"}[m["role"]] ,"parts":[{"text": m["content"]}]})
        response = client.generate_content(contents, generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens}, safety_settings=safety)
        content = response.text
        cost = prompt_price_per_million[model] * response.usage_metadata.prompt_token_count / 1e6 + completion_price_per_million[model] * response.usage_metadata.candidates_token_count / 1e6
    else:
        print("Unsupported user model:", model)
    return content, cost


SYSTEM_PROMPT = """You are an user interacting with an agent.

Instruction: {instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.
"""



class NaiveUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model="gpt-4"):
        super().__init__()
        self.messages = None
        self.system_prompt = SYSTEM_PROMPT
        self.model = model
        self.total_cost = 0

    def reset(self, instruction=None) -> str:
        self.total_cost = 0
        self.messages = [
            {"role": "system", "content": self.system_prompt.format(instruction=instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        content, cost = chat_completion_request(self.messages, self.model, temperature=1.0, max_output_tokens=150)
        self.messages.append({"role": "assistant", "content": content})
        self.total_cost += cost
        return content

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        content, cost = chat_completion_request(self.messages, self.model, temperature=1.0, max_output_tokens=150)
        self.messages.append({"role": "assistant", "content": content})
        self.total_cost += cost
        return content

    def get_total_cost(self):
        return self.total_cost


def load_user(user_mode: str, model: str="gpt-4") -> BaseUserSimulationEnv:
    if user_mode == "human":
        return HumanUserSimulationEnv()
    elif user_mode == "naive":
        return NaiveUserSimulationEnv(model=model)
    else:
        raise ValueError(f"Unknown user mode {user_mode}")
