from typing import Union

import torch
import json
import os

from .tools import TOOL_END, TOOL_START, Tool, extract_tool
from .logger import get_logger

# Get logger for this module
logger = get_logger(__name__)




class Prompt:
    def __init__(self, content=None, role=None, pre_tokenizer_format: dict=None, image_path: str = None,
                 attempted_tool_call:bool = None, successful_tool_call:bool = None, image_size:(int, int)=None,
                 num_tokens: int=None):


        self.content: list[dict[str, str]]
        self.role: str

        self.image_path = image_path
        self.image_size = image_size # role = user
        self.num_tokens = num_tokens # role = assistant

        self.attempted_tool_call = attempted_tool_call # this only makes sense for role=assistant
        self.successful_tool_call = successful_tool_call # this only makes sense for role=user


        if pre_tokenizer_format is not None:
            self.content = pre_tokenizer_format["content"]
            self.role = pre_tokenizer_format["role"]
        else:
            assert role is not None
            assert content is not None

            self.content = content
            self.role = role

    def get_text(self) -> str:
        return_candidates = []

        for content in self.content:
            if content["type"] == "text":
                return_candidates.append(content["text"])
        if len(return_candidates) == 0:
            raise ValueError(f"Prompt {self.content} has no text")
        if len(return_candidates) > 1:
            raise ValueError(f"Prompt {self.content} has more than one text")

        return return_candidates[0]

    def to_dict(self) -> dict:
        return {"content": self.content, "role": self.role}

    def to_list_dict(self) -> list[dict]:
        return [self.to_dict()]

    def has_image_path(self) -> bool:
        if self.image_path is None:
            return False
        else:
            return True

    def get_image_path(self) -> str:
        return self.image_path

    def get_image_size(self) -> (int, int):
        return self.image_size

    def get_num_tokens(self) -> int:
        return self.num_tokens



class Conversations:
    def __init__(self, num_conversations: int):
        self.user_prompts:list[list[Prompt]] = [[] for _ in range(num_conversations)]
        self.model_replies:list[list[Prompt]] = [[] for _ in range(num_conversations)]

        self.is_finished:list[bool] = [False for _ in range(num_conversations)]

        #self.image_paths:list[str] = ["" for _ in range(num_conversations)]

    def get_conv_partner(self, source: str):
        if source == "user":
            messages = self.user_prompts
        elif source == "model" or source == "assistant":
            messages = self.model_replies
        else:
            raise ValueError(f"Invalid source: {source}")
        return messages

    def add_messages(self, messages: list[Prompt]):
        increased_messages = self.get_conv_partner(messages[0].role)
        assert len(increased_messages) == len(messages)
        for idx, message in enumerate(messages):
            increased_messages[idx].append(message)

    def get_model_generations(self) -> list[str]:
        """
        concatenates the model generations. useful for reward calculation
        :return:
        """
        all_texts = []
        for conv in self.model_replies:
            text = ""
            for prompt in conv:
                text += prompt.get_text()
            all_texts.append(text)
        return all_texts


    def get_messages(self, source: str, pos_in_conv:int) -> list[Prompt]:
        """

        :param source:
        :param pos_in_conv:
        :return: slices the column at position pos_in_conv
        """
        target_messages = self.get_conv_partner(source)
        return [message[pos_in_conv] for message in target_messages]

    def get_messages_as_dicts(self, source: str, pos_in_conv:int) -> list[dict[str, str]]:
        target_messages = self.get_conv_partner(source)
        return [message[pos_in_conv].to_dict() for message in target_messages]


    def add_message(self, message: Prompt, idx: int):

        increased_messages = self.get_conv_partner(message.role)
        increased_messages[idx].append(message)

    def show(self, source: str, only_text=False):
        messages = self.get_conv_partner(source)
        descriptive_strings = []
        for idx, conv in enumerate(messages):
            if only_text:
                descriptive_strings.append(f"{idx}: {[prompt.get_text() for prompt in conv]}")
            else:
                descriptive_strings.append(f"{idx}: {conv}")
        return "\n".join(descriptive_strings)

    #def add_image_path(self, path:str, ):
     #   self.image_paths.append(path)

    def interleave(self) -> list[list[Prompt]]:
        """
        This function interleaves self.user_prompts and self.model_replies
        :return: [[self.user_prompts[0][0], self.model_replies[0][1], ...],
                  [self.user_prompts[1][0], self.model_replies[1][1], ...],
                  ...
                  ]
        """
        interleaved = []
        for i in range(len(self.user_prompts)):
            conversation = []
            # Determine the length of the combined conversation
            conversation_length = len(self.user_prompts[i]) + len(self.model_replies[i])

            # Alternate adding user prompts and model replies
            for j in range(conversation_length):
                if j % 2 == 0:  # Even indices for user prompts
                    user_index = j // 2
                    if user_index < len(self.user_prompts[i]):
                        conversation.append(self.user_prompts[i][user_index])
                else:  # Odd indices for model replies
                    model_index = j // 2
                    if model_index < len(self.model_replies[i]):
                        conversation.append(self.model_replies[i][model_index])

            interleaved.append(conversation)

        return interleaved

    def update_image_size(self, image_sizes:list[(int,int)], conv_idx:int):
        conversation = self.user_prompts[conv_idx]
        for prompt_idx, prompt in enumerate(conversation):
            prompt.image_size = image_sizes[prompt_idx]

    def get_image_sizes(self):
        image_sizes = []
        for conv_idx, conv in enumerate(self.user_prompts):
            image_sizes.append([prompt.get_image_size() for prompt in conv if prompt.has_image_path()])
        return image_sizes

    def get_image_paths(self, ignore_finished:bool=False) -> list[list[str]]:
        image_paths = []
        for conv_idx, conv in enumerate(self.user_prompts):
            if ignore_finished and self.is_finished[conv_idx]:
                continue
            image_paths.append([prompt.get_image_path()  for prompt in conv if prompt.has_image_path()])
        return image_paths

    def get_num_tokens(self) -> list[list[int]]:
        num_tokens = []
        for conv_idx, conv in enumerate(self.model_replies):
            num_tokens.append([prompt.get_num_tokens() for prompt in conv ])
        return num_tokens

    def get_attempted_tool_calls(self, idx: int):
        """
        Only counts the no. of tool calls that were successful.
        Args:
            idx:

        Returns:

        """
        conv = self.model_replies[idx]
        no_tool_calls = 0
        for prompt in conv:
            if prompt.attempted_tool_call:
                no_tool_calls += 1
        return no_tool_calls


    def get_full_for_hf_prep(self, ignore_finished:bool=False) -> list[dict[str, dict]]:
        """

        :return: [
                    {"prompt": [{"content": [{"text": str, "type": str}, {"text": str, "type": str}], "role": "user"},
                                {"content": [{"text": str, "type": str}, {"text": str, "type": str}], "role": "assistant"}]
                    }
        ]
        """
        interleaved = self.interleave()

        interleaved_dicts = []

        for conv_idx, conv in enumerate(interleaved):
            if ignore_finished and self.is_finished[conv_idx]:
                continue
            interleaved_dicts.append([prompt.to_dict() for prompt in conv])


        hf_prompt_format = self.wrap_with_prompt_for_hf(interleaved_dicts)

        image_paths = self.get_image_paths(ignore_finished)

        for conv, image_path_list in zip(hf_prompt_format, image_paths):
            if len(image_path_list) > 0:
                conv["image_path"] = image_path_list

        return hf_prompt_format

    def wrap_with_prompt_for_hf(self, sub_conversations: list):
        """
        turns [a,b, [c]] into [{"prompt": [a]}, {"prompt": [b]}, {"prompt": [c]}]
        :param sub_conversations:
        :return:
        """
        hf_format = []
        for conv in sub_conversations:
            if not isinstance(conv, list):
                conv = [conv]
            hf_format.append({"prompt": conv})
        return hf_format

    def set_finished(self, idxs: list[int]):
        for idx in idxs:
            self.is_finished[idx] = True

    def get_no_tool_calls(self, idx: int):
        """
        Only counts the no. of tool calls that were successful.
        Args:
            idx:

        Returns:

        """
        conv = self.user_prompts[idx]
        no_tool_calls = 0
        for prompt in conv:
            if prompt.successful_tool_call:
                no_tool_calls += 1
        return no_tool_calls


    def handle_tool_call(self, save_path, step, tools:Union[Tool, list[Tool]]):
        if isinstance(tools, Tool):
            tools = [tools]
        image_paths = self.get_image_paths()
        for conv_id, reply in enumerate(self.model_replies):
            if self.is_finished[conv_id]:
                continue
            prompt = reply[-1]

            if prompt.content[0]["type"] == "text":
                if TOOL_START in prompt.content[0]["text"] or TOOL_END in prompt.content[0]["text"]:
                    prompt.attempted_tool_call = True

                    try:
                        tool_params = extract_tool(prompt.content[0]["text"])
                        logger.info(f"tool params: {tool_params}")
                        tool_params["image_paths"] = image_paths[conv_id]
                        correct_tool_name = False
                        for tool in tools:
                            if (tool.name == tool_params["name"] or
                               (tool_params["name"] == "crop_image" and tool.name == "crop_image_normalized")):
                                correct_tool_name = True
                                tool_call_result = tool.call_tool(tool_params, save_path)



                                #tool_call_result = call_tool(tool_params, save_path)

                                json.dump(
                                    {"original_image_path": tool_call_result["original_image_path"],
                                    "tool_input_image_path": tool_call_result["tool_input_image_path"],
                                     "tool_name": tool.name,
                                    "tool_input": tool_call_result["tool_input"],
                                    "step": step},
                                    open(os.path.join(save_path, f"{tool_call_result["new_image_id"]}.json"), "w"))

                                self.add_message(Prompt(content=tool_call_result["output_message"],
                                                             role="user", successful_tool_call=True,
                                                             image_path=tool_call_result["new_image_path"]),
                                                 idx=conv_id)
                        if not correct_tool_name:
                            raise ValueError(f"Invalid tool name: {tool_params['name']}")

                    except Exception as e:
                        logger.info(f"Error in tool call: {e}")
                        #self.add_message(Prompt(
                        #    content=[{"type": "text", "text": "Error in tool call: " + str(e)}],
                        #    role="user", successful_tool_call=False
                        #), conv_id)
                        self.is_finished[conv_id] = True
                        prompt.failed_tool_call = True
                else:
                    self.is_finished[conv_id] = True








