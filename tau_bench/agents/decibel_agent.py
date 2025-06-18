
import json
import time
import uuid
import traceback
from typing import Any, Dict, List

from termcolor import colored

from tau_bench.agents.base import Agent as BaseAgent
from tau_bench.envs.base import Env, retry_helper
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from google.cloud.dialogflowcx_v3beta1 import services, types
from google.oauth2 import service_account
# import google.cloud.dialogflow_v3alpha1 as df_v3alpha1
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict
import proto.marshal.collections.maps
import proto.marshal.collections.repeated


LANG_CODE="en"


def map_composite_to_dict(map_composite):
    return {key: make_json_dumpable(value) for key, value in map_composite.items()}


def repeated_composite_to_list(repeated_composite):
    return [make_json_dumpable(item) for item in repeated_composite]


def make_json_dumpable(data):
    if isinstance(data, dict):
        return {key: make_json_dumpable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_json_dumpable(item) for item in data]
    elif isinstance(data, proto.marshal.collections.maps.MapComposite):
        return map_composite_to_dict(data)
    elif isinstance(data, proto.marshal.collections.repeated.RepeatedComposite):
        return repeated_composite_to_list(data)
    else:
        return data


def pretty_print_conversation(messages: List[Dict[str, Any]]) -> None:
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "yellow",
        "tool": "magenta",
    }

    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" :
            print(
                colored(f"assistant: {message['query_result']}\n", role_to_color[message["role"]])
            )
        elif message["role"] == "tool":
            print(
                colored(
                    f"tool: {message['tool_result']}\n",
                    role_to_color[message["role"]],
                )
            )


class DecibelAgent(BaseAgent):
    def __init__(self, model: str = "gemini-pro", agent_id=None, project_id="df-decibel2-dev-test", service_account_file="df-decibel2-dev-test-934d38d2bb24.json"):
        self.model = model
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        self.session_client = services.sessions.SessionsClient(credentials=credentials)
        tool_client = services.tools.ToolsClient(credentials=credentials)
        self.request_parent = f"projects/{project_id}/locations/global/agents/{agent_id}"
        all_tools = tool_client.list_tools(parent=self.request_parent).tools
        # tool full name to display name map
        self.tools_name_map = {t.name:t.display_name for t in all_tools}
        # stack to keep track of the tool call results we should sent to DF
        self.pending_tool_calls = []
        self.reset()

    def get_action(self, messages: List[Dict[str, Any]]):
        request = types.session.DetectIntentRequest(session=self.session_id)
        if len(self.pending_tool_calls) == 0:
            if isinstance(messages[-1]["content"], str):
                text_input = types.session.TextInput(text=messages[-1]["content"])
            # elif isinstance(messages[-1]["content"], list):
            #     print(messages[-1]["content"])
            #     text_input = types.session.TextInput(text=messages[-1]["content"][0])
            request.query_input = types.session.QueryInput(text=text_input, language_code=LANG_CODE)
        else:
            tool_call = self.pending_tool_calls.pop()
            tool_call_result=types.tool_call.ToolCallResult(tool=tool_call.tool, action=tool_call.action)
            # if "tool_result" not in messages[-1]:
            #     print(messages[-1])
            #     breakpoint()
            if "tool_result" not in messages[-1]:
                print(tool_call_result)
                breakpoint()
                # messages[-1]["tool_result"] = "Tool call result not found"
                # return
                raise Exception("Tool call result not found in the last message, probably a transfer to human tool call")
            if "error" in messages[-1]["tool_result"].lower():
                tool_call_result.error = types.tool_call.ToolCallResult.Error(message=messages[-1]["tool_result"])
            else:
                tool_call_result_output = Struct()
                tool_call_result_output.update({"result": messages[-1]["tool_result"]})
                tool_call_result.output_parameters = tool_call_result_output
            request.query_input = types.session.QueryInput(tool_call_result=tool_call_result, language_code=LANG_CODE)
        def request_func():
            response = self.session_client.detect_intent(request=request)
            result = response.query_result.response_messages
            if len(result) == 0:
                raise Exception(f"Empty response messages, retrying...\nResult: {result}")
            return response
        response = retry_helper(request_func)
        
        result = response.query_result.response_messages
        if len(result)==1:
            if "text" in result[0]:
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": result[0].text.text[0]})
            elif "end_interaction" in result[0]:
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": "END CONVERSATION"})
            else:
                tool_call = result[0].tool_call
                if tool_call.tool not in self.tools_name_map:
                    print("Unknown tool [{%s}] from decibel agent. response: [{%s}]" %(tool_call.tool, response))
                action_args = {k:v for k,v in tool_call.input_parameters.items()}
                # action_args = MessageToDict(tool_call.input_parameters)
                self.pending_tool_calls.append(tool_call)
                action = Action(name=self.tools_name_map[tool_call.tool], kwargs=make_json_dumpable(action_args))
            response_message = {"response_message": make_json_dumpable(MessageToDict(result[0]._pb)), "generative_info": make_json_dumpable(MessageToDict(response.query_result.generative_info._pb))}
        elif len(result) == 2:
            assert "text" in result[0] or "end_interaction" in result[0]
            assert hasattr(result[1], 'tool_call'), "Second message in response should be a tool call"
            tool_call = result[1].tool_call
            if tool_call.tool == '':
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": "END CONVERSATION"})
            else:
                if tool_call.tool not in self.tools_name_map:
                    raise Exception(f"Unknown tool [{tool_call.tool}] from decibel agent")
                action_args = {k:v for k,v in tool_call.input_parameters.items()}
                self.pending_tool_calls.append(tool_call)
                action = Action(name=self.tools_name_map[tool_call.tool], kwargs=make_json_dumpable(action_args))

            merged_dict = {**MessageToDict(result[0]._pb), **MessageToDict(result[1]._pb)}
            response_message = {"response_message": make_json_dumpable(merged_dict), "generative_info": make_json_dumpable(MessageToDict(response.query_result.generative_info._pb))}
        else:
            raise Exception(f"DetectIntentResponse.query_result.response_messages has incorrect length, expect 1 or 2, got [{len(result)}]")
        
        return response_message, action


    def reset(self):
        self.session_id = f"{self.request_parent}/sessions/"+str(uuid.uuid1())

    def solve(self, env: Env, task_index=None, verbose=False, temperature=0.0):
        self.reset()
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        print("INFO: Env reset. Info: ", info)
        reward = 0
        messages = [{"role": "user", "content": obs}]
        for _ in range(30):
            try:
                cur_message, action = self.get_action(messages)
            except Exception as e:
                print(traceback.format_exc())
                info["error"] = str(e)
                breakpoint()
            # print("=======INFO: DF Agent returned action: ", action)
            log_message = {"role": "assistant"}
            response_message = cur_message['response_message']
            if "toolCall" in response_message:
                log_message['tool_calls'] = [response_message['toolCall']]
            else:
                log_message['tool_calls'] = None
            if "text" in response_message:
                log_message['content'] = response_message['text']['text']
            else:
                log_message['content'] = None
            messages.append(log_message)
                
            # print(f"action: {action.name}, arguments: {action.kwargs}")
            # print(f"model: {log_message}")
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            # print("=======INFO: Env returned observation: ", obs)
            obs = env_response.observation
            if action.name != RESPOND_ACTION_NAME:
                messages.append({"role": "tool", "tool_result": obs})
                # print("Tool call result: ", obs)
            else:
                messages.append({"role": "user", "content": obs})
                # print("User: ", obs)
            if verbose:
                self.render(messages, 2)
            if env_response.done:
                break
        self.pending_tool_calls = []
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages
        )

    def render(self, messages, last_n=None):
        if last_n is not None:
            pretty_print_conversation(messages[-last_n:])
        else:
            pretty_print_conversation(messages)

