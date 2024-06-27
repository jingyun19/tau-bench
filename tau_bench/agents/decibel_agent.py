
import json
import time
import uuid
import traceback

from tenacity import retry, stop_after_attempt, wait_random_exponential

from tau_bench.agents.base import BaseAgent
from tau_bench.agents.utils import pretty_print_conversation
from google.cloud.dialogflowcx_v3beta1 import services, types
from google.oauth2 import service_account
import google.cloud.dialogflow_v3alpha1 as df_v3alpha1
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict


LANG_CODE="en"



class DecibelAgent(BaseAgent):
    def __init__(self, model: str = "gemini-pro", agent_id=None, project_id="df-decibel2-dev-test", service_account_file="df-decibel2-dev-test-934d38d2bb24.json"):
        self.model = model
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        self.session_client = services.sessions.SessionsClient(credentials=credentials)
        tool_client = df_v3alpha1.services.tools.ToolsClient(credentials=credentials)
        self.request_parent = f"projects/{project_id}/locations/global/agents/{agent_id}"
        all_tools = tool_client.list_tools(parent=self.request_parent).tools
        # tool full name to display name map
        self.tools_name_map = {t.name:t.display_name for t in all_tools}
        # stack to keep track of the tool call results we should sent to DF
        self.pending_tool_calls = []
        self.reset()

        
    def get_action(self):
        request = types.session.DetectIntentRequest()
        request.session = self.session_id
        if len(self.pending_tool_calls) == 0:
            text_input = types.session.TextInput(text=self.messages[-1]["content"])
            request.query_input = types.session.QueryInput(text=text_input, language_code=LANG_CODE)
        else:
            tool_call = self.pending_tool_calls.pop()
            tool_call_result=types.tool_call.ToolCallResult(tool=tool_call.tool, action=tool_call.action)
            if "error" in self.messages[-1]["tool_result"].lower():
                tool_call_result.error = types.tool_call.ToolCallResult.Error(message=self.messages[-1]["tool_result"])
            else:
                tool_call_result_output = Struct()
                tool_call_result_output.update({"result": self.messages[-1]["tool_result"]})
                tool_call_result.output_parameters = tool_call_result_output
            request.query_input = types.session.QueryInput(tool_call_result=tool_call_result, language_code=LANG_CODE)
        response = self.session_client.detect_intent(request=request)
        result = response.query_result.response_messages
        if len(result)!=1:
            raise Exception("DetectIntentResponse.query_result.response_messages has incorrect length!")
        
        action_name = ""
        action_args = {}
        if "text" in result[0]:
            action_name = "respond"
            action_args = {"content": result[0].text.text[0]}
        else:
            tool_call = result[0].tool_call
            if tool_call.tool not in self.tools_name_map:
                raise Exception(f"Unknown tool from decibel agent. response: [{result}]")
            action_name = self.tools_name_map[tool_call.tool]
            action_args = {k:v for k,v in tool_call.input_parameters.items()}
            self.pending_tool_calls.append(tool_call)
        return response.query_result, {"name": action_name, "arguments": action_args}


    def reset(self):
        self.messages= []
        self.session_id = f"{self.request_parent}/sessions/"+str(uuid.uuid1())

    def act(self, env, index=None, verbose=False, temperature=0.0):
        self.reset()
        obs, info = env.reset(index=index)
        reward = 0
        self.messages.append({"role": "user", "content": obs})
        print("Agent messages: ",self.messages)
        for _ in range(30):
            try:
                message, action = self.get_action()
            except Exception as e:
                print(traceback.format_exc())
                info["error"] = str(e)
                break
            print("=======INFO: DF Agent returned action: ", action)
            self.messages.append({"role": "assistant", "detect_intent_result": MessageToDict(message._pb)})

            obs, reward, done, info = env.step(action)
            print("=======INFO: Env returned observation: ", obs)
            if action["name"] != "respond":
                self.messages.append({"role": "user", "tool_result": obs})
            else:
                self.messages.append({"role": "user", "content": obs})
            if verbose:
                self.render(2)
            if done:
                break
        return reward, info

    def render(self, last_n=None):
        if last_n is not None:
            pretty_print_conversation(self.messages[-last_n:])
        else:
            pretty_print_conversation(self.messages)

    def get_messages(self):
        return self.messages

