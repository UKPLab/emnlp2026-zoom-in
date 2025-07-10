import torch
import json
from PIL import Image
import uuid
import os

SHOW_IMAGE = {
        "type": "function",
        "function": {
            "name": "show_image",
            "description": "The input image gets repeated. Specify which image position to repeat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_image": {
                        "type": "integer",
                        "description": "The position of the image to repeat (e.g., 1 for first image, 2 for second image, etc.)",
                        "minimum": 1
                    }
                },
                "required": ["target_image"]
            },
        },
    }

ZOOM_IN = {
        "type": "function",
        "function": {
            "name": "zoom_in",
            "description": "Zoom in on the image based on the bounding box coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "description":"normalized coordinates for bounding box of the region you want to zoom in. Values should be within [0.0,1.0].",
                        # "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.",
                        "items": {
                        "type": "float",
                        }
                    },
                    "target_image":{
                        "type": "integer",
                        "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."
                    }
                },
                "required": ["bbox_2d", "target_image"]
            }
        },
    }

TOOLS = [
    #SHOW_IMAGE,
    ZOOM_IN
]



TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"

def extract_tool(text:str, tool_start:str = TOOL_START, tool_end:str = TOOL_END):
    print(f"extract tool: {text.split(tool_start)[-1].split(tool_end)[0]}")
    return json.loads(text.split(tool_start)[-1].split(tool_end)[0])

def call_tool(tool_params:dict, save_path:str=None):
    format_reminder = " Once you are done, output the final answer in <answer> </answer> tags."

    tool_name = tool_params['name']
    tool_args = tool_params['arguments']

    actual_tool_args = tool_args.copy()

    if tool_name == "show_image":
        tool = show_image
        output_description = f"Here is image {tool_args['target_image']} again."

        actual_tool_args.pop("target_image")
        # use this instead: tool_params['image_paths'][int(tool_args['target_image']) - 1] or at least assert it
        actual_tool_args["image_path"] = tool_params['image_paths'][0]


    elif tool_name == "zoom_in":
        tool = zoom_in
        output_description = f"Here is image {tool_args['target_image']} zoomed in at {tool_args['bbox_2d']}."

        actual_tool_args.pop("target_image")

        if not (1 <= int(tool_args['target_image']) <= len(tool_params['image_paths'])):
            raise ValueError(f"target_image position out of bounds: {tool_args['target_image']}")
        actual_tool_args["image_path"] = tool_params['image_paths'][int(tool_args['target_image']) - 1]

    else:
        raise ValueError(f"Invalid tool name: {tool_name}")

    output_description += format_reminder

    new_image = tool(**actual_tool_args)
    
    if isinstance(new_image, str):
        new_image_path = new_image
        new_image_id = str(uuid.uuid4())
    else:
        new_image_id = str(uuid.uuid4())
        new_image_name = f"{new_image_id}.png"
        new_image_path = os.path.join(save_path, "generated_images", new_image_name)
        new_image.save(new_image_path)

    return {
        "original_image_path": tool_params['image_paths'][0],
        "tool_input_image_path": actual_tool_args["image_path"],
        "tool_name": tool_name,
        "tool_input": tool_args,

        "new_image_path": new_image_path,
        "new_image_id": new_image_id,
        "output_message": output_description,
    }


def show_image(image_path: str, **kwargs):
    """
    Returns the input image. It is then fed to the MLLM again so it can spot details that were missed earlier.
    :return: The input image.
    """
    return image_path

def zoom_in(image_path, bbox_2d, padding=(0.1,0.1)):
    """
    Crop the image based on the bounding box coordinates.
    """
    image = Image.open(image_path)
    img_x, img_y = image.size
    padding_tr = (600.0/img_x,600.0/img_y)
    padding = (min(padding[0],padding_tr[0]),min(padding[1],padding_tr[1]))

    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0])-padding[0], float(bbox_2d[1])-padding[1], float(bbox_2d[2])+padding[0], float(bbox_2d[3])+padding[1])
    else:
        normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding[0], float(bbox_2d[1])/img_y-padding[1], float(bbox_2d[2])/img_x+padding[0], float(bbox_2d[3])/img_y+padding[1])
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 =min(max(0, normalized_x1), 1)
    normalized_y1 =min(max(0, normalized_y1), 1)
    normalized_x2 =min(max(0, normalized_x2), 1)
    normalized_y2 =min(max(0, normalized_y2), 1)
    cropped_img = image.crop((int(normalized_x1*img_x), int(normalized_y1*img_y), int(normalized_x2*img_x), int(normalized_y2*img_y)))
    w, h = cropped_img.size
    assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"

    
    return cropped_img

class Prompt:
    def __init__(self, content=None, role=None, pre_tokenizer_format: dict=None, image_path: str = None,
                 attempted_tool_call:bool = None, successful_tool_call:bool = None):


        self.content: list[dict[str, str]]
        self.role: str

        self.image_path = image_path

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

    def get_image_paths(self, ignore_finished:bool=False) -> list[list[str]]:
        image_paths = []
        for conv_idx, conv in enumerate(self.user_prompts):
            if ignore_finished and self.is_finished[conv_idx]:
                continue
            image_paths.append([prompt.get_image_path()  for prompt in conv if prompt.has_image_path()])
        return image_paths

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
        conv = self.user_prompts[idx]
        no_tool_calls = 0
        for prompt in conv:
            if prompt.successful_tool_call:
                no_tool_calls += 1
        return no_tool_calls


    def handle_tool_call(self, save_path, step):
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
                        print(f"tool params: {tool_params}")
                        tool_params["image_paths"] = image_paths[conv_id]
                        tool_call_result = call_tool(tool_params, save_path)

                        json.dump(
                            {"original_image_path": tool_call_result["original_image_path"],
                            "tool_input_image_path": tool_call_result["tool_input_image_path"],
                             "tool_name": tool_call_result["tool_name"],
                            "tool_input": tool_call_result["tool_input"],
                            "step": step},
                            open(os.path.join(save_path, f"{tool_call_result["new_image_id"]}.json"), "w"))

                        self.add_message(Prompt(content=[{"text": None, "type": "image"},
                                                         {'text': tool_call_result["output_message"],
                                                            'type': 'text'}],
                                                     role="user", successful_tool_call=True,
                                                     image_path=tool_call_result["new_image_path"]),
                                         idx=conv_id)
                    except Exception as e:
                        print(f"Error in tool call: {e}")
                        self.add_message(Prompt(
                            content=[{"type": "text", "text": "Error in tool call: " + str(e)}],
                            role="user", successful_tool_call=False
                        ), conv_id)
                        self.is_finished[conv_id] = True
                else:
                    self.is_finished[conv_id] = True





def get_boundaries(content: list[str], left_boundary="<|im_start|>assistant\n", right_boundary="<|im_end|>"):

    # maybe off by 1 error?

    model_generated_boundaries = []
    for generation in content:
        end_of_str_reached = False
        model_generated_boundary = []
        start_idx = 0
        while not end_of_str_reached:
            new_left_idx = generation[start_idx:].find(left_boundary) + len(left_boundary) + start_idx
            start_idx = new_left_idx
            if generation[start_idx:].find(right_boundary) != -1:
                new_right_idx = generation[start_idx:].find(right_boundary) + start_idx
                model_generated_boundary.append([new_left_idx, new_right_idx])
                start_idx = new_right_idx
            else:
                end_of_str_reached = True
                model_generated_boundary.append([new_left_idx, None])
        model_generated_boundaries.append(model_generated_boundary)

    return model_generated_boundaries

def get_boundaries_tokenized(content: torch.tensor, left_boundary_tokens:torch.tensor, right_boundary_tokens:torch.tensor)-> list[list[list[int]]]:
    """
    while the left and right boundaries are wrt to the model generation, this function creates start and end of the non-model
    generations
    return model_generated_boundaries:list. outermost list is a text sequence, then we have a list of 2-lists (consisting of start non-model, end non-model)
    as the number of rounds
    """
    # Default boundaries must be provided
    if left_boundary_tokens is None or right_boundary_tokens is None:
        raise ValueError("Both left_boundary_tokens and right_boundary_tokens must be provided")

    left_len = len(left_boundary_tokens)
    right_len = len(right_boundary_tokens)

    model_generated_boundaries = []

    for token_seq in content:
        seq_len = len(token_seq)
        boundaries = []
        j = 0
        i = 0
        left_idx = 0
        right_idx = 0

        while j <= seq_len - left_len:
            if torch.equal(token_seq[j:j + left_len], left_boundary_tokens):
                right_idx = j + left_len
                break
            j += 1
        boundaries.append([0, right_idx])

        i = j
        while i <= seq_len - left_len:
            if torch.equal(token_seq[i:i + right_len], right_boundary_tokens):
                left_idx = i
                j = left_idx
                while j <= seq_len - left_len:
                    if torch.equal(token_seq[j:j + left_len], left_boundary_tokens):
                        right_idx = j + left_len
                        break
                    j += 1
                i = right_idx
                boundaries.append([left_idx, right_idx])
            else:
                i += 1

        model_generated_boundaries.append(boundaries)

    return model_generated_boundaries


