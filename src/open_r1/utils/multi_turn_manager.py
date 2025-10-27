from typing import Union, Tuple, List, Optional

import torch
import json
import os

from .tools import TOOL_END, TOOL_START, Tool, extract_tool
from .logger import get_logger
from transformers import AutoProcessor
from trl.data_utils import maybe_apply_chat_template
from PIL import Image

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
    "turn_separator": {"id": 198, "text": "\n"}
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
                 image_path: str = None,
                 image_size:Tuple[int, int]=None,
                 attempted_tool_call:bool = None,
                 successful_tool_call:bool = None):
        self.text = text
        self.role = role

        self.text_wrapped = text_wrapped
        self.text_tokenized = text_tokenized
        self.text_wrapped_tokenized = text_wrapped_tokenized

        self.token_ids = token_ids
        self.token_ids_shorten_image = token_ids_shorten_image

        self.image_token_lens = image_token_lens

        self.image_path = image_path
        self.image_size = image_size
        self.attempted_tool_call = attempted_tool_call
        self.successful_tool_call = successful_tool_call

    def wrap(self, type: str) -> Union[str, list[int]]:
        """ valid types are 'text' and 'id' """
        if type == "text":
            seq = (f"{SPECIAL_TOKENS['turn_start'][type]}"
                    f"{SPECIAL_TOKENS[self.role][type]}"
                    f"{SPECIAL_TOKENS['newline'][type]}"
                   f"{self.text}")
            if self.role != "assistant":
                seq += f"{SPECIAL_TOKENS['turn_end'][type]}"
        elif type == "id":
            seq = []
            seq.append(SPECIAL_TOKENS['turn_start'][type])
            seq.append(SPECIAL_TOKENS[self.role][type])
            seq.append(SPECIAL_TOKENS['newline'][type])
            seq += self.token_ids

            if self.role != "assistant":
                seq.append(SPECIAL_TOKENS['turn_end'][type])
        else:
            raise ValueError(f"Invalid turn type: {type}, choose from 'id', 'text'")

        return seq


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
        full_vars = {k:v for k,v in full_vars.items() if k != "token_ids"}
        return json.dumps(full_vars, indent=4)

    def __str__(self):
        return vars(self)


class MultiTurn:
    def __init__(self, batch_size: int, processor, tools: Optional[Tool]):
        self.batch_size = batch_size
        self.processor = processor
        self.tools = tools

        self.all_multi_turn:list[list[Turn]] = [[] for _ in range(batch_size)]

    def add_initial_user_prompt(self, prompts:list[dict], image_paths:list[list[str]]):
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
        system_user_image_paths = []
        for idx in range(self.batch_size):
            prompt = prompts[idx]
            text = None
            for content in prompt['content']:
                if content['type'] == 'text':
                    text = content['text']
            if text is None:
                raise ValueError(f'Text content is missing: {prompt}')

            logger.info(f"prompt: {repr(prompt)}")

            wrapped_prompt = maybe_apply_chat_template({"prompt": [prompt]},
                                      tokenizer=self.processor,
                                      add_generation_prompt=None,
                                      return_assistant_tokens_mask=False,
                                      tools=self.tools.get_tool_dict() if self.tools is not None else None)["prompt"]

            logger.info(f"wrapped prompt: {repr(wrapped_prompt)}")
            split_prompt = wrapped_prompt.split(SPECIAL_TOKENS["turn_start"]["text"])
            logger.info(f"split prompt: {repr(split_prompt)}")
            system_text = split_prompt[1].removesuffix(SPECIAL_TOKENS["turn_end"]["text"]+
                                                       SPECIAL_TOKENS["turn_separator"]["text"]).removeprefix(SPECIAL_TOKENS["system"]["text"]+
                                                                                                              SPECIAL_TOKENS["newline"]["text"])
            logger.info(f"system_text: {repr(system_text)}")
            system_turn = Turn(role="system", text=system_text)

            user_text = split_prompt[2].removesuffix(SPECIAL_TOKENS["turn_end"]["text"]+
                                                     SPECIAL_TOKENS["turn_separator"]["text"]).removeprefix(SPECIAL_TOKENS["user"]["text"]+
                                                                                                            SPECIAL_TOKENS["newline"]["text"])
            logger.info(f"user_text: {repr(user_text)}")
            user_turn = Turn(role=prompt["role"], text=user_text)

            system_user_texts.append(system_text)
            system_user_texts.append(user_text)
            for img_path in image_paths[idx]:
                system_user_image_paths.append(Image.open(img_path))

            self.all_multi_turn[idx].append(system_turn)
            self.all_multi_turn[idx].append(user_turn)

        input_ids = self.processor(
            text=system_user_texts,
            images=system_user_image_paths,
            return_tensors=None,
            padding=False,
            add_special_tokens=False,
            return_offsets_mapping=False
        )["input_ids"]

        for idx, prompt in enumerate(prompts):
            self.all_multi_turn[idx][0].set_token_ids(input_ids[2*idx])
            self.all_multi_turn[idx][1].set_token_ids(input_ids[2*idx+1])

    def add_model_reply(self, token_ids: list[list[int]]):
        text_completions = self.processor.batch_decode(token_ids, skip_special_tokens=True)
        for idx in range(self.batch_size):

            model_turn = Turn(role="assistant", text=text_completions[idx])
            model_turn.set_token_ids(token_ids[idx])

            self.all_multi_turn[idx].append(model_turn)



    def add_tool_execution(self, prompts:list[dict], image_paths:list[list[str]]):
        pass

    def get_sequences(self, type:str) -> Union[list[str], list[int]]:
        """ valid types are 'text' and 'id' """
        seqs = []
        for idx in range(self.batch_size):
            conv = self.all_multi_turn[idx]
            if type == "text":
                seqs.append(SPECIAL_TOKENS["turn_separator"][type].join([turn.wrap(type) for turn in conv]))
            elif type == "id":
                seq = []
                for conv_idx, turn in enumerate(conv):
                    seq += turn.wrap(type)
                    if conv_idx + 2 >= self.batch_size:
                        seq.append(SPECIAL_TOKENS["turn_separator"][type])
                seqs.append(seq)
            else:
                raise ValueError(f"type {type} not supported, choose from 'id' and 'text'")


        return seqs


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





