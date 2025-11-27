import traceback
from typing import Any, Iterable, List, Optional, Sequence, Union, Tuple

import numpy as np
from numpy import array
import torch
import json
import os
import copy

from .tools import TOOL_END, TOOL_START, Tool, extract_tool
from .logger import get_logger
from transformers import AutoProcessor
from trl.data_utils import maybe_apply_chat_template
from PIL import Image

from .utils import get_resized_image_scales

# Get logger for this module
logger = get_logger(__name__)

SPECIAL_TOKENS = {
    "turn_start":   {"id": 151644, "text": "<|im_start|>"},
    "turn_end":     {"id": 151645, "text": "<|im_end|>"},
    "image_start":  {"id": 151652, "text": "<|vision_start|>"},
    "image_end":    {"id": 151653, "text": "<|vision_end|>"},
    "image_pad":    {"id": 151655, "text": "<|image_pad|>"},

    "system":       {"id": 8948, "text": "system"},
    "user":         {"id": 872, "text": "user"},
    "assistant":    {"id": 77091, "text": "assistant"},

    "newline":      {"id": 198, "text": "\n"},
    "turn_separator": {"id": 198, "text": "\n"},

    "tool_start": {"id": 151657, "text": "<tool_call>"},
    "tool_end": {"id": 151658, "text": "</tool_call>"},

    # from here on, we only use the ID, no guarantee that the text is correct (escaping etc)
    "backslash": {"id": 1124, "text": "\\"},
    "double_backslash": {"id": 59, "text": "\\\\"},
    # C is newline
    "dot_CC": {"id": 382, "text": ".ĊĊ"},
    # G is whitespace
    "CC": {"id": 4710, "text": "ĠĊĊ"},
    "C": {"id": 715, "text": "ĠĊ"},

    "box": {"id": 79075, "text": "boxed"},
    "open_curly_bracket": {"id": 90, "text": "{"},

    "the": {"id": 785, "text": "The"},
    "answer": {"id": 4226, "text": " answer"},
    "is": {"id": 374, "text": " is"},
}

class Turn:
    def __init__(self,
                 text: str = None,
                 text_wrapped: str = None,
                 text_tokenized: str = None,
                 text_wrapped_tokenized: str = None,
                 token_ids: list[int] = None,
                 token_ids_shorten_image: list[int] = None,
                 image_token_lens: list[int] = None,
                 role: str = None,
                 image_paths: list[str] = None,
                 image_grid_thw_list: List[np.array] = None,
                 pixel_values_list: List[np.array] = None,
                 image_sizes:list[Tuple[int, int]]=None,
                 attempted_tool_call:bool = None,
                 successful_tool_call:bool = None,
                 is_dummy:bool = False):
        self.text = text
        self.role = role

        self.text_wrapped = text_wrapped
        self.text_tokenized = text_tokenized
        self.text_wrapped_tokenized = text_wrapped_tokenized

        self.token_ids = token_ids
        self.token_ids_shorten_image = token_ids_shorten_image

        self.image_token_lens = image_token_lens

        self.image_paths = image_paths
        self.image_sizes = image_sizes

        self.image_grid_thw_list = image_grid_thw_list
        self.pixel_values_list = pixel_values_list


        self.attempted_tool_call = attempted_tool_call
        self.successful_tool_call = successful_tool_call

        self.is_dummy = is_dummy

    def wrap(self, type: str, full_image_pad: bool) -> Union[str, list[int]]:
        """ valid types are 'text' and 'id' """
        if type == "text":
            seq = (f"{SPECIAL_TOKENS['turn_start'][type]}"
                    f"{SPECIAL_TOKENS[self.role][type]}"
                    f"{SPECIAL_TOKENS['newline'][type]}"
                   f"{self.text}")
            if self.role != "assistant" or not seq.endswith(SPECIAL_TOKENS['turn_end'][type]):
                if not self.is_dummy:
                    seq += f"{SPECIAL_TOKENS['turn_end'][type]}"
            #if self.role != "assistant":
            #    seq += f"{SPECIAL_TOKENS['turn_end'][type]}"
        elif type == "id":
            seq = []
            seq.append(SPECIAL_TOKENS['turn_start'][type])
            seq.append(SPECIAL_TOKENS[self.role][type])
            seq.append(SPECIAL_TOKENS['newline'][type])
            if full_image_pad:
                seq += self.token_ids
            else:
                seq += self.token_ids_shorten_image

            if self.role != "assistant" or seq[-1] != SPECIAL_TOKENS['turn_end'][type]:
                if not self.is_dummy:
                    seq.append(SPECIAL_TOKENS['turn_end'][type])
        else:
            raise ValueError(f"Invalid turn type: {type}, choose from 'id', 'text'")

        return seq

    def get_mask(self, type:str) -> list[int]:
        if type == "everything_except_model_generation":
            seq = []
            #seq.append(SPECIAL_TOKENS['turn_start'][type])
            seq.append(0)
            #seq.append(SPECIAL_TOKENS[self.role][type])
            seq.append(0)
            #seq.append(SPECIAL_TOKENS['newline'][type])
            seq.append(0)
            if self.role == "assistant":
                seq += [1 for _ in self.token_ids]
                if seq[-1] != SPECIAL_TOKENS['turn_end']["id"]:
                    seq.append(0)
            else:
                seq += [0 for _ in self.token_ids]
                #seq.append(SPECIAL_TOKENS['turn_end'][type])
                seq.append(0)
            return seq
        else:
            raise ValueError(f"Invalid type {type}, choose from 'everything_except_model_generation'")




    def set_token_ids(self, input_ids: list[int]):
        reduced_input_ids, image_lens = remove_image_pad(input_ids,
                                                         vision_start=SPECIAL_TOKENS["image_start"]["id"],
                                                         vision_end=SPECIAL_TOKENS["image_end"]["id"],
                                                         vision_pad=SPECIAL_TOKENS["image_pad"]["id"]
                                                         )
        restored_input_ids = restore_image_pad(reduced_input_ids, image_lens, vision_start=SPECIAL_TOKENS["image_start"]["id"],
                                                         vision_end=SPECIAL_TOKENS["image_end"]["id"],
                                                         vision_pad=SPECIAL_TOKENS["image_pad"]["id"])

        if restored_input_ids != input_ids:
            raise RuntimeError(f"Image is malformed and could not be reduced: {input_ids}")
        else:
            self.token_ids = input_ids
            self.token_ids_shorten_image = reduced_input_ids
            self.image_token_lens = image_lens

    def __repr__(self):
        full_vars = vars(self)
        short_vars = {}
        for k,v in full_vars.items():
            if k == "token_ids":
                continue
            elif k == "image_grid_thw_list":
                short_vars["image_grid_thw_shapes"] = [x.shape for x in v] if v is not None else None
            elif k == "pixel_values_list":
                short_vars["pixel_values_shapes"] = [x.shape for x in v] if v is not None else None
            else:
                short_vars[k] = v

        return json.dumps(short_vars, indent=4)

    def __str__(self):
        return vars(self)


class MultiTurn:
    def __init__(self, batch_size: int, processor, tools: Optional[list[Tool]]):
        self.batch_size = batch_size
        self.processor = processor



        self.tools = tools

        self.all_multi_turn:list[list[Turn]] = [[] for _ in range(batch_size)]

        self.is_finished:list[bool] = [False for _ in range(batch_size)]

    def add_initial_user_prompt(self, prompts:list[dict], image_paths:list[str]):
        """

        Args:
            prompts: {
        'role': 'user',
        'content': [
            {'type': 'image', 'text': None},
            {'type': 'text', 'text': "Hello World"}
        ]
        }

        Returns:

        """

        system_user_texts = []
        for idx in range(self.batch_size):
            prompt = prompts[idx]

            #logger.info(f"prompt: {repr(prompt)}")

            wrapped_prompt = maybe_apply_chat_template({"prompt": [prompt]},
                                      tokenizer=self.processor,
                                      add_generation_prompt=None,
                                      return_assistant_tokens_mask=False,
                                      tools=[tool.get_tool_dict() for tool in self.tools] if self.tools is not None else None)["prompt"]

            #logger.info(f"wrapped prompt: {repr(wrapped_prompt)}")
            split_prompt = wrapped_prompt.split(SPECIAL_TOKENS["turn_start"]["text"])
            #logger.info(f"split prompt: {repr(split_prompt)}")
            system_text = split_prompt[1].removesuffix(SPECIAL_TOKENS["turn_end"]["text"]+
                                                       SPECIAL_TOKENS["turn_separator"]["text"]).removeprefix(SPECIAL_TOKENS["system"]["text"]+
                                                                                                              SPECIAL_TOKENS["newline"]["text"])
            #logger.info(f"system_text: {repr(system_text)}")
            system_turn = Turn(role="system", text=system_text)

            user_text = split_prompt[2].removesuffix(SPECIAL_TOKENS["turn_end"]["text"]+
                                                     SPECIAL_TOKENS["turn_separator"]["text"]).removeprefix(SPECIAL_TOKENS["user"]["text"]+
                                                                                                            SPECIAL_TOKENS["newline"]["text"])
            #logger.info(f"user_text: {repr(user_text)}")
            user_turn = Turn(role="user", text=user_text)

            system_user_texts.append(system_text)
            system_user_texts.append(user_text)
            #for img_path in image_paths[idx]:
            #    system_user_image_paths.append(Image.open(img_path))

            self.all_multi_turn[idx].append(system_turn)
            self.all_multi_turn[idx].append(user_turn)

        open_images = []
        for image_path in image_paths:
            img = Image.open(image_path)
            try:
             # Ensure minimum dimensions of 28 pixels
                w, h = img.size
                if w < 28 or h < 28:
                # Calculate new dimensions maintaining aspect ratio
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            except Exception as e:
                    logger.info(f"Warning: could not process image {image_path}: {e}")
            open_images.append(img)

        processed = self.processor(
            text=system_user_texts,
            images=open_images,
            return_tensors=None,
            padding=False,
            add_special_tokens=False,
            return_offsets_mapping=False
        )

        input_ids = processed["input_ids"]

        image_grid_thw_list, pixel_values_list = split_image(processed["image_grid_thw"], processed["pixel_values"])

        # we need to make the mapping of the images with the turns, as the input list is flattened
        img_count_old = 0
        img_count_new = 0
        for idx in range(self.batch_size):
            self.all_multi_turn[idx][0].set_token_ids(input_ids[2*idx])
            self.all_multi_turn[idx][1].set_token_ids(input_ids[2*idx+1])
            img_count_new = img_count_old + len(self.all_multi_turn[idx][1].image_token_lens)
            self.all_multi_turn[idx][1].image_paths = image_paths[img_count_old:img_count_new]
            img_sizes = []
            for img in open_images[img_count_old:img_count_new]:
                w,h = img.size
                h,w = get_resized_image_scales(h,w,
                                               self.processor.image_processor.min_pixels,
                                               self.processor.image_processor.max_pixels)
                img_sizes.append((w,h))
            self.all_multi_turn[idx][1].image_sizes = img_sizes.copy()
            self.all_multi_turn[idx][1].image_grid_thw_list = image_grid_thw_list[img_count_old:img_count_new]
            self.all_multi_turn[idx][1].pixel_values_list = pixel_values_list[img_count_old:img_count_new]
            img_count_old = img_count_new

    def add_model_reply(self, token_ids: list[list[int]], mapping: list[int] = None):
        """ the mapping is between the token_id list and the global position in the manager and they should have the same length.
        Example: [1,2,4] indicates 0->1, 1->2 and 2->4"""
        if mapping is None:
            mapping = range(len(token_ids))
        #logger.info(f"in add model reply: mapping: {mapping}")
        #logger.info(f"in add model reply: len token_ids: {len(token_ids)}")
        text_completions = self.processor.batch_decode(token_ids, skip_special_tokens=True)
        for idx in range(len(token_ids)):

            model_turn = Turn(role="assistant", text=text_completions[idx])
            model_turn.set_token_ids(token_ids[idx])

            self.all_multi_turn[mapping[idx]].append(model_turn)

    def get_ids(self, is_finished:bool):
        filtered_ids = []
        for idx in range(self.batch_size):
            if self.is_finished[idx] == is_finished:
                filtered_ids.append(idx)

        return filtered_ids

    def add_user_message(self, prompts:list[dict], image_paths:Optional[list[str]], mapping: list[int]=None):
        #logger.info(f"in add user message: prompts: {prompts}, image_paths: {image_paths}")

        if mapping is None:
            mapping = range(len(prompts))

        text_image_prompts = []
        for idx in range(len(mapping)):
            prompt = prompts[idx]

            text_image_prompt = ""
            for prompt_part in prompt["content"]:
                if prompt_part["type"] == "image":
                    text_image_prompt += SPECIAL_TOKENS["image_start"]["text"]
                    text_image_prompt += SPECIAL_TOKENS["image_pad"]["text"]
                    text_image_prompt += SPECIAL_TOKENS["image_end"]["text"]
                if prompt_part["type"] == "text":
                    text_image_prompt += prompt_part["text"]

            text_image_prompts.append(text_image_prompt)

            user_turn = Turn(role="user", text=text_image_prompt)
            self.all_multi_turn[mapping[idx]].append(user_turn)

        open_images = [Image.open(image_path) for image_path in image_paths] if image_paths is not None else None

        processed = self.processor(
            text=text_image_prompts,
            images= open_images,
            return_tensors=None,
            padding=False,
            add_special_tokens=False,
            return_offsets_mapping=False
        )

        input_ids = processed["input_ids"]

        if image_paths is not None:
            image_grid_thw_list, pixel_values_list = split_image(processed["image_grid_thw"], processed["pixel_values"])

        # we need to make the mapping of the images with the turns, as the input list is flattened
        img_count_old = 0
        img_count_new = 0
        # TODO: this fails if the user message does not contain an image path. This should only happen in the think_again text mode
        for idx in range(len(mapping)):
            self.all_multi_turn[mapping[idx]][-1].set_token_ids(input_ids[idx])
            if image_paths is not None:
                img_count_new = img_count_old + len(self.all_multi_turn[mapping[idx]][-1].image_token_lens)
                self.all_multi_turn[mapping[idx]][-1].image_paths = image_paths[img_count_old:img_count_new]
                img_sizes = []
                for img in open_images[img_count_old:img_count_new]:
                    w, h = img.size
                    h, w = get_resized_image_scales(h, w,
                                                    self.processor.image_processor.min_pixels,
                                                    self.processor.image_processor.max_pixels)
                    img_sizes.append((w, h))
                self.all_multi_turn[mapping[idx]][-1].image_sizes = img_sizes.copy()
                self.all_multi_turn[mapping[idx]][-1].image_grid_thw_list = image_grid_thw_list[img_count_old:img_count_new]
                self.all_multi_turn[mapping[idx]][-1].pixel_values_list = pixel_values_list[img_count_old:img_count_new]
                img_count_old = img_count_new



    def get_sequences(self, type:str, add_assistant_start:bool, full_image_pad:bool, ignore_finished:bool=False) -> Union[list[str], list[list[int]]]:
        """

        Args:
            type: 'text' or 'id'
            add_assistant_start: if the last turn was from the user, should we start the assistant turn (i.e. append |im_start|assistant\n )
            full_image_pad: only makes a difference when type= 'id'. Then we return all 1000+ image_pad token ids, instead of only a single placeholder
            ignore_finished: whether to ignore finished sequences

        Returns:

        """
        seqs = []
        for idx in range(self.batch_size):
            if ignore_finished and self.is_finished[idx]:
                continue
            conv = self.all_multi_turn[idx].copy()

            seqs.append(get_sequence(conv, type, add_assistant_start, full_image_pad))

        return seqs

    def get_flattened_image_paths(self) -> list[list[str]]:
        all_img_paths = []
        for idx in range(self.batch_size):
            img_paths = []
            for turn in self.all_multi_turn[idx]:
                if turn.image_paths is not None:
                    img_paths += turn.image_paths
        return all_img_paths

    def get_image_paths(self, ignore_finished:bool=False, flatten=True) -> Union[list[str], list[list[str]]]:
        image_paths = []
        for idx in range(self.batch_size):
            if ignore_finished and self.is_finished[idx]:
                continue
            seq_image_paths = []
            for turn in self.all_multi_turn[idx]:
                if turn.image_paths is not None:
                    seq_image_paths += turn.image_paths
            if flatten:
                image_paths += seq_image_paths
            else:
                image_paths.append(seq_image_paths)


        return image_paths

    def get_image_sizes(self, flatten=False) -> list[list[tuple[int, int]]]:
        all_image_sizes = []
        for idx in range(self.batch_size):
            image_sizes = []
            for turn in self.all_multi_turn[idx]:
                if turn.image_sizes is not None:
                    image_sizes += turn.image_sizes

            if flatten:
                all_image_sizes += image_sizes
            else:
                all_image_sizes.append(image_sizes)

        return all_image_sizes

    def handle_tool_call(self, save_path, step, strict_extraction=False):
        image_paths = self.get_image_paths(flatten=False)
        for conv_id, turns in enumerate(self.all_multi_turn):
            if self.is_finished[conv_id]:
                continue
            turn = turns[-1]
            assert turn.role == "assistant", f"expected turn.role to be 'assistant' but got {turn.role}"


            if TOOL_START in turn.text or TOOL_END in turn.text:
                turn.attempted_tool_call = True

                try:
                    tool_params = extract_tool(turn.text, strict=strict_extraction)
                    logger.info(f"tool params: {tool_params}")
                    tool_params["image_paths"] = image_paths[conv_id]
                    correct_tool_name = False
                    for tool in self.tools:
                        if (tool.name == tool_params["name"] or
                           (tool_params["name"] == "crop_image" and tool.name == "crop_image_normalized")):
                            correct_tool_name = True
                            tool_call_result = tool.call_tool(tool_params, save_path)

                            json.dump(
                                {"original_image_path": tool_call_result["original_image_path"],
                                "tool_input_image_path": tool_call_result["tool_input_image_path"],
                                 "tool_name": tool.name,
                                "tool_input": tool_call_result["tool_input"],
                                "step": step},
                                open(os.path.join(save_path, f"{tool_call_result["new_image_id"]}.json"), "w"))

                            self.add_user_message(prompts=[{"content": tool_call_result["output_message"],
                                                            "role": "user"}],
                                                  image_paths=[tool_call_result["new_image_path"]],
                                                  mapping=[conv_id])
                            turn.successful_tool_call = True
                            #self.add_message(Prompt(content=tool_call_result["output_message"],
                            #                             role="user", successful_tool_call=True,
                            #                             image_path=tool_call_result["new_image_path"]),
                            #                 idx=conv_id)
                    if not correct_tool_name:
                        raise ValueError(f"Invalid tool name: {tool_params['name']}")

                except Exception as e:
                    logger.info(f"Error in tool call: {e}")

                    self.add_user_message(prompts=[{"content": [{"text": f"Error in tool call: {e}", "type": "text"}],
                                                    "role": "user"}],
                                          image_paths=None,
                                          mapping=[conv_id])

                    #self.is_finished[conv_id] = True
                    turn.successful_tool_call = False
            else:
                self.is_finished[conv_id] = True

    def get_no_tool_calls(self, idx, type="success") -> int:
        no_tool_calls = 0
        for turn in self.all_multi_turn[idx]:
            if type == "success":
                if turn.successful_tool_call:
                    no_tool_calls += 1
            elif type == "attempt":
                if turn.attempted_tool_call:
                    no_tool_calls += 1
            else:
                raise ValueError(f"Invalid type: {type}. Choose from 'success' or 'attempt'")
        return no_tool_calls

    def get_mask(self, type:str) -> list[list[int]]:
        seqs = []
        for idx in range(self.batch_size):
            conv = self.all_multi_turn[idx]

            if type == "everything_except_model_generation":
                seq = []
                for conv_idx, turn in enumerate(conv):
                    seq += turn.get_mask(type)
                    #seq += turn.wrap(type, full_image_pad=full_image_pad)
                    if conv_idx + 2 <= len(conv):
                        #seq.append(SPECIAL_TOKENS["turn_separator"][type])
                        seq.append(0)
                seqs.append(seq)
            else:
                raise ValueError(f"type {type} not supported, choose from 'everything_except_model_generation'")

        return seqs

    def get_model_generations(self, type:str) -> list[str]:
        """
        type is 'text' or 'id'
        """
        full_model_generations = []
        for idx in range(self.batch_size):
            conv = self.all_multi_turn[idx]
            if type == "text":
                model_generations = ""
            elif type == "id":
                model_generations = []
            else:
                raise ValueError(f"type {type} not supported, choose from 'text', 'id'")
            for conv_idx, turn in enumerate(conv):
                if turn.role == "assistant":
                    if type == "text":
                        model_generations += turn.text
                    if type == "id":
                        model_generations += turn.token_ids_shorten_image
            full_model_generations.append(model_generations)
        return full_model_generations

    def get_model_generation_ids(self, lengths: bool, flatten: bool) -> Union[list[int], list[list[int]]]:
        """

        Returns: if lengths: list of len(token_ids)
                 else: list of list of token_ids

        """
        full_model_generations = []
        for idx in range(self.batch_size):
            conv = self.all_multi_turn[idx]
            model_generations = []
            for conv_idx, turn in enumerate(conv):
                if turn.role == "assistant":
                    if flatten:
                        model_generations += turn.token_ids
                    else:
                        model_generations.append(turn.token_ids)
            if lengths and flatten: # list(int) -> int
                full_model_generations.append(len(model_generations))
            if lengths and not flatten: # list(list(int)) -> list(int)
                full_model_generations.append([len(m) for m in model_generations])
            if not lengths and flatten: # list(int) -> list(int)
                full_model_generations.append(model_generations)
            if not lengths and not flatten: # list(list(int)) -> list(list(int))
                full_model_generations.append(model_generations)

        return full_model_generations

    def get_multimodal(self, type:str, positions:dict[int, list[int]] = None) -> np.array:
        """
        type is either "image_grid_thw" or "pixel_values"
        positions is a dict of the form {0: [1, 3]} and indicates that only the images
        from the first and third turn (i.e. zeroth and first user turn) of the zeroth conversation should be used
        """
        multimodal_list = []
        for idx in range(self.batch_size):
            if positions is not None and idx not in positions.keys():
                continue
            conv = self.all_multi_turn[idx]
            for turn_idx, turn in enumerate(conv):
                if positions is not None and turn_idx not in positions[idx]:
                    continue
                if type == "image_grid_thw":
                    if turn.image_grid_thw_list is not None:
                        multimodal_list += turn.image_grid_thw_list
                elif type == "pixel_values":
                    if turn.pixel_values_list is not None:
                        multimodal_list += turn.pixel_values_list
                else:
                    raise ValueError("got type {type}. Choose 'pixel_values' or 'image_grid_thw'")

        return np.concatenate(multimodal_list)

    def get_shortened_sequences(self, remove:str, contrasted_area:str, bridge:str):
        """
        returns
        inputs: list[str] ,
        the image positions: dict[int, list[int]} ,
        considered seqs: dict[int, dict[contrasted_area: tuple, contrasted_area_short: tuple]
        """
        considered_seqs = {}
        for idx in range(self.batch_size):
            conv = self.all_multi_turn[idx]
            validity = self.get_shortened_sequence(conv, remove=remove, contrasted_area_name=contrasted_area, bridge=bridge)
            if validity is not None:
                considered_seqs[idx] = validity
            else:
                considered_seqs[idx] = {"contrasted_area": [0,1],
                                                    "contrasted_area_short": [0,1],
                                                    "short_sequence": get_sequence(conv, type="id",
                                                                                   add_assistant_start=False,
                                                                                   full_image_pad=True),
                                                    "image_turns": range(len(conv)),
                                                    "dummy": True}




        input_ids = [v["short_sequence"] for v in considered_seqs.values()]

        positions = {k:v["image_turns"] for k,v in considered_seqs.items()}

        images_per_sample = []
        for conv_id, v in considered_seqs.items():
            images_per_conv = 0
            for turn_id in v["image_turns"]:
                if self.all_multi_turn[conv_id][turn_id].image_paths is not None:
                    images_per_conv += len(self.all_multi_turn[conv_id][turn_id].image_paths)
            images_per_sample.append(images_per_conv)

        return input_ids, positions, images_per_sample, considered_seqs



    def get_shortened_sequence(self, conv: list[Turn], remove:str, contrasted_area_name: str, bridge: str) -> Optional[dict]:
        if remove == "tool_to_box":
            # system, user, model, tool_exec, model

            # the end of the reasoning chain usually looks like this:
            # end_text = [". ", "\\", "boxed", "{", ..., "}", "<|im_end|>"]
            # end_token = [382, 59, 79075, 90, ..., 92, 151645]
            #
            # the tool use looks like this:
            # tool_text = [".",  "<tool_call>", ..., "</tool_call>", "<|im_end|>"]
            # tool_token = [13, 151657,     ..., 151658,      151645]
            #
            # the idea is to replace tool_token with end_token and
            # calculate the logp diff on [..., 92, 151645], i.e. everything after boxed{


            if len(conv) != 5:
                return None
            if len(conv[2].token_ids) == 0:
                return None

            # searching from behind to get the last tool call
            tool_start_idx = rindex_any(conv[2].token_ids, SPECIAL_TOKENS["tool_start"]["id"])

            if len(conv[4].token_ids) == 0:
                return None

            box_idx = rindex_any(conv[4].token_ids,
                                 [SPECIAL_TOKENS["box"]["id"],
                                               SPECIAL_TOKENS["open_curly_bracket"]["id"]])

            if box_idx is None:
                return None

            new_conv = copy.deepcopy(conv)

            # only keep (system, user, model)
            new_conv = new_conv[:3]
            # get rid of tool call and the "." before (if applicable)
            # cutoff_idx = max(tool_start_idx - 1, 0)

            # get rid of tool call
            cutoff_idx = max(tool_start_idx, 0)
            new_conv[2].token_ids = conv[2].token_ids[:cutoff_idx]

            # between M_1^R and boxed{
            if bridge is None:
                new_conv[2].token_ids += [SPECIAL_TOKENS["backslash"]["id"]]
            elif bridge == "double_newline":
                new_conv[2].token_ids += [SPECIAL_TOKENS["CC"]["id"],
                                          SPECIAL_TOKENS["backslash"]["id"],
                                          ]
            elif bridge == "double_newline,the,answer,is":
                new_conv[2].token_ids += [SPECIAL_TOKENS["CC"]["id"],
                                          SPECIAL_TOKENS["the"]["id"],
                                          SPECIAL_TOKENS["answer"]["id"],
                                          SPECIAL_TOKENS["is"]["id"],
                                          SPECIAL_TOKENS["backslash"]["id"],
                                          ]
            else:
                raise ValueError(f"bridge: {bridge} is unsupported. Choose from 'double_newline', 'double_newline,the,answer,is' or None")

            # add answer
            new_conv[2].token_ids += conv[4].token_ids[box_idx:]

            # indices are relative and from the right
            if contrasted_area_name == "first_box_entry_to_end":
                contrasted_area_begin = len(conv[4].token_ids[box_idx+2:])
                contrasted_area_end = 0
            elif contrasted_area_name == "first_box_entry":
                contrasted_area_begin = len(conv[4].token_ids[box_idx + 2:])
                contrasted_area_end = contrasted_area_begin - 1
            else:
                raise ValueError(f"contrasted_area_name: {contrasted_area_name} is unsupported. Choose from 'first_box_entry_to_end' or 'first_box_entry'")

            logger.info(f"contrasted_area_begin: {contrasted_area_begin}")
            logger.info(f"contrasted_area_end: {contrasted_area_end}")

            # can happen if generation ends with "\boxed{"
            if contrasted_area_begin == 0:
                return None

            logger.info(f"short model generation: {new_conv[2].token_ids}")
            logger.info(f"contrasted area token ids: {conv[4].token_ids[-contrasted_area_begin:len(conv[4].token_ids)-contrasted_area_end]}")

            short_sequence = get_sequence(new_conv, type="id", add_assistant_start=False, full_image_pad=True)
            sequence = get_sequence(conv, type="id", add_assistant_start=False, full_image_pad=True)

            contrasted_area_short = (len(short_sequence) - contrasted_area_begin,
                                     len(short_sequence) - contrasted_area_end)
            contrasted_area = (len(sequence) - contrasted_area_begin,
                               len(sequence) - contrasted_area_end)

            logger.info(f"contrasted area: {contrasted_area}")
            logger.info(f"contrasted area short: {contrasted_area_short}")

            assert sequence[contrasted_area[0]:contrasted_area[1]] == short_sequence[contrasted_area_short[0]:contrasted_area_short[1]], f"the contrasted area should contain the same token ids but got full: {sequence[contrasted_area]} and short: {short_sequence[contrasted_area_short]} "

        else:
            raise ValueError(f"the given remove value {remove} is not supported. Choose from 'tool_and_box'")

        return {"contrasted_area": contrasted_area,
                "contrasted_area_short": contrasted_area_short,
                "short_sequence": short_sequence,
                "image_turns": [0,1,2],
                "dummy": False}

    def check(self, all_multimodal_inputs: list[dict], all_multimodal_token_inputs: list[dict],
              input_text: list[str]):
        # Sanity check 1: manager_text == handler_text
        if input_text != [i["prompt"] for i in all_multimodal_inputs]:
            logger.debug(
                f"input_text: {input_text} is not equal to ground truth: {[i["prompt"] for i in all_multimodal_inputs]}")
        else:
            logger.info(f"manager text is equal to ground truth!")

        full_token_seq = [x["prompt_token_ids"] for x in all_multimodal_token_inputs]
        # Sanity check 2: tokenized(manager_text) == manager_tokens. Together with 1 this implies manager_tokens == tokenized(handler text)
        if full_token_seq == [] and input_text == []:
            logger.info(f"tokenized manager text is equal to ground truth!")
        else:
            logger.info(f"in check: input text: {input_text}")
            tokenized_input_text = self.processor(text=input_text, images=None)["input_ids"]
            if tokenized_input_text != full_token_seq:
                logger.debug(f"tokenized input_text: {tokenized_input_text} is not equal to ground truth: {full_token_seq}")
            else:
                logger.info(f"tokenized manager text is equal to ground truth!")

        # SAnity check 3: the same image paths are used
        assert len(all_multimodal_inputs) == len(all_multimodal_token_inputs), f"all_multimodal_inputs={all_multimodal_inputs} != {all_multimodal_token_inputs}=all_multimodal_token_inputs"
        for idx in range(len(all_multimodal_inputs)):

            if all_multimodal_inputs[idx]["image_path"] != all_multimodal_token_inputs[idx]["image_path"]:
                logger.debug(f"image paths are different. got: {all_multimodal_token_inputs[idx]["image_path"]} but ground truth is {all_multimodal_inputs[idx]["image_path"]}")


def remove_image_pad(input_ids: Union[list[list[int]], list[int]], vision_start:int, vision_end:int, vision_pad:int) -> (list[list[int]], list[list[int]]):
    """
    This function gets [vision_start, vision_pad * 10, vision_end] and returns ([vision_start, vision_pad * 1, vision_end] , [10])
    """
    input_was_flat_list = False
    if len(input_ids) == 0:
        return [], []
    else:
        if not isinstance(input_ids[0], list):
            input_ids = [input_ids]
            input_was_flat_list = True

    reduced_seqs = []
    all_image_lens = []
    for seq in input_ids:
        in_image = None
        image_lens = []
        reduced_seq = []
        for token_id in seq:
            if token_id == vision_start:
                in_image = True
                image_len = 0
                reduced_seq.append(token_id)
            elif token_id == vision_pad:
                image_len += 1
            elif token_id == vision_end:
                in_image = False
                image_lens.append(image_len)
                reduced_seq.append(vision_pad)
                reduced_seq.append(vision_end)
            else:
                reduced_seq.append(token_id)
        reduced_seqs.append(reduced_seq)
        all_image_lens.append(image_lens)

    if input_was_flat_list:
        reduced_seqs = reduced_seqs[0]
        all_image_lens = all_image_lens[0]
    return reduced_seqs, all_image_lens

# Python
def restore_image_pad(
    reduced_ids: list[list[int]],
    all_image_lens: list[list[int]],
    vision_start: int,
    vision_end: int ,
    vision_pad: int ,
) -> list[list[int]]:
    """
    Inverse of remove_image_pad.
    Given sequences where each image region is reduced to [vision_start, vision_pad, vision_end]
    and the corresponding lengths per image region (e.g., [10]), restore the original
    [vision_start, vision_pad * n, vision_end].

    Args:
        reduced_ids: token id sequences with single vision_pad per image region.
        all_image_lens: per-sequence list of image lengths to expand.
        vision_start, vision_end, vision_pad: special token ids.

    Returns:
        A list of restored token id sequences.
    """
    input_was_flat_list = False
    if len(reduced_ids) == 0:
        return []
    else:
        if not isinstance(reduced_ids[0], list):
            reduced_ids = [reduced_ids]
            all_image_lens = [all_image_lens]
            input_was_flat_list = True

    restored = []
    for seq, lens in zip(reduced_ids, all_image_lens):
        lens_idx = 0
        out = []
        i = 0
        while i < len(seq):
            tok = seq[i]
            if tok == vision_start:
                out.append(tok)
                # Expect a single vision_pad then a vision_end; expand using lens[lens_idx]
                # Copy the next token if it's the single pad
                j = i + 1
                if j < len(seq) and seq[j] == vision_pad and (j + 1) < len(seq) and seq[j + 1] == vision_end:
                    # Expand pads according to stored length
                    n = lens[lens_idx] if lens_idx < len(lens) else 0
                    out.extend([vision_pad] * n)
                    out.append(vision_end)
                    lens_idx += 1
                    i = j + 2
                    continue
                else:
                    # Fallback: if structure is unexpected, just move forward
                    i += 1
                    continue
            else:
                out.append(tok)
                i += 1
        restored.append(out)

    if input_was_flat_list:
        restored = restored[0]
    return restored

def split_image(image_grid_thw: np.array, pixel_values: np.array) -> Tuple[List[np.array], List[np.array]]:
    # Get the grid information to determine split points
    # image_grid_thw  # Shape: (total_images, 3)
    # pixel_values   # Shape: (total_patches, channels, patch_height, patch_width)

    # Calculate number of patches per image
    patches_per_image = (image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()

    # Split image_grid_thw (assuming one grid per image in batch)
    image_grid_thw_split = []
    start_idx = 0
    pixel_values_split = []
    pixel_start_idx = 0
    for i in range(len(image_grid_thw)):
        end_idx = start_idx + 1 #int(self.num_images[0, i].item())
        image_grid_thw_split.append(image_grid_thw[start_idx:end_idx])

        num_patches = sum(patches_per_image[start_idx:end_idx])
        pixel_end_idx = pixel_start_idx + num_patches
        pixel_values_split.append(pixel_values[pixel_start_idx:pixel_end_idx])
        start_idx = end_idx
        pixel_start_idx = pixel_end_idx
    return image_grid_thw_split, pixel_values_split

def pad(unpadded: list[list[int]], padding_side: str, padding_value: int) -> list[list[int]]:
    padded = []
    max_len = max([len(seq) for seq in unpadded])
    for seq in unpadded:
        if len(seq) < max_len:
            if padding_side == "right":
                seq += [padding_value for _ in range(len(seq), max_len)]
            elif padding_side == "left":
                seq = [padding_value for _ in range(len(seq), max_len)] + seq
            else:
                raise ValueError(f"padding_side={padding_side} is not supported. Choose 'left' or 'right'")
        padded.append(seq)
    return padded

def _is_subseq_at(seq: Sequence[Any], sub: Sequence[Any], start: int) -> bool:
    """Return True if sub matches seq[start:start+len(sub)]."""
    end = start + len(sub)
    if end > len(seq):
        return False
    # Fast path single element
    if len(sub) == 1:
        return seq[start] == sub[0]
    return list(seq[start:end]) == list(sub)

def rindex_any(seq: Sequence[Any], value_or_subseq: Union[Any, Sequence[Any]]) -> Optional[int]:
    """
    Return the starting index of the last occurrence of value_or_subseq in seq.
    - Works for single elements and contiguous sublists/sequences.
    - Returns None if not found.
    """
    # Normalize to a sequence for unified handling
    if isinstance(value_or_subseq, (list, tuple)):
        sub = list(value_or_subseq)
    else:
        sub = [value_or_subseq]

    n = len(seq)
    m = len(sub)
    if m == 0:
        return None  # define empty pattern as not found

    # Search from the end
    # Last possible start index is n - m
    for i in range(n - m, -1, -1):
        if _is_subseq_at(seq, sub, i):
            return i
    return None

def get_sequence(conv: list[Turn], type: str, add_assistant_start: bool, full_image_pad: bool) -> Union[list[int], str]:

    if add_assistant_start and conv[-1].role == "user":
        conv.append(Turn(role="assistant", text="", token_ids=[], token_ids_shorten_image=[], is_dummy=True))

    if type == "text":
        seq = SPECIAL_TOKENS["turn_separator"][type].join(
            [turn.wrap(type, full_image_pad=full_image_pad) for turn in conv])

    elif type == "id":
        seq = []
        for conv_idx, turn in enumerate(conv):
            seq += turn.wrap(type, full_image_pad=full_image_pad)
            if conv_idx + 2 <= len(conv):
                seq.append(SPECIAL_TOKENS["turn_separator"][type])


    else:
        raise ValueError(f"type {type} not supported, choose from 'id' and 'text'")

    return seq











