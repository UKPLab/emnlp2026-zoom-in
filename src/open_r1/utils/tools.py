import json

from PIL import Image
import uuid
import os
from functools import partial, reduce
import operator
import copy
from qwen_vl_utils import smart_resize

from .logger import get_logger
from .utils import get_resized_image_scales

# Get logger for this module
logger = get_logger(__name__)

TOOL_CONFIGS = {
            "PR_zoom_in":
            {   "tool_name": "zoom_in",
                "tool_template": "zoom_in",
                "tool_json_customization": {"function,description":"Zoom in on the image based on the bounding box coordinates.",
                                            "function,parameters,properties,bbox_2d,description":"normalized coordinates for bounding box of the region you want to zoom in. Values should be within [0.0,1.0].",
                                            "function,parameters,properties,bbox_2d,items,type": "number",
                                            "function,parameters,properties,target_image,type": "number",},

                "tool_message_image_pos": "last",
                "tool_message_text_message": "\nHere is the cropped image (Image Size: <width>x<height>):",
                "tool_message_text_fillers": ["width", "height"],
                "prompt_type": "pr_adapted"
            },
            "PR_zoom_in_with_hint":
            {"tool_name": "zoom_in",
             "tool_template": "zoom_in",
             "tool_json_customization": {
                 "function,description": "Zoom in on the image based on the bounding box coordinates. It is useful when the object or text in the image is too small to be seen.",
                 "function,parameters,properties,bbox_2d,description": "normalized coordinates for bounding box of the region you want to zoom in. Values should be within [0.0,1.0].",
                 "function,parameters,properties,bbox_2d,items,type": "number",
                 "function,parameters,properties,target_image,type": "number", },

             "tool_message_image_pos": "last",
             "tool_message_text_message": "\nHere is the cropped image (Image Size: <width>x<height>):",
             "tool_message_text_fillers": ["width", "height"],
             "prompt_type": "pr_adapted"
             },
            "zoom_in_absolute":
            {   "tool_name": "zoom_in",
                "tool_template": "zoom_in",
                "tool_json_customization": {"function,description": "Zoom in on the image based on the bounding box coordinates.",
                                            "function,parameters,properties,bbox_2d,description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.",
                                            "function,parameters,properties,bbox_2d,items,type": "integer"},
                "tool_message_image_pos": "last",
                "tool_message_text_message": "\nHere is the cropped image (Image Size: <width>x<height>):",
                "tool_message_text_fillers": ["width", "height"],
                "prompt_type": "pr_adapted"
            },
            "zoom_in_relative":
            {   "tool_name": "zoom_in",
                "tool_template": "zoom_in",
                "tool_json_customization": {"function,description": "Zoom in on the image based on the bounding box coordinates.",
                                            "function,parameters,properties,bbox_2d,description": "normalized coordinates for bounding box of the region you want to zoom in. Values should be within [0.0,1.0].",
                                            "function,parameters,properties,bbox_2d,items,type": "float"},
                "tool_message_image_pos": "last",
                "tool_message_text_message": "\nHere is the cropped image (Image Size: <width>x<height>):",
                "tool_message_text_fillers": ["width", "height"],
                "prompt_type": "pr_adapted"
            },
            "no_tool": {
                "tool_name": "",
                "tool_template": "",
                "tool_json_customization": {},
                "tool_message_image_pos": "",
                "tool_message_text_message": "",
                "tool_message_text_fillers": [],
                "prompt_type": "no_tool"
            },
            "select_frames": {
                "tool_name": "select_frames",
                "tool_template": "select_frames",
                "tool_json_customization": {},
                "tool_message_image_pos": "last",
                "tool_message_text_message": "\nHere are the selected frames (Frame Size: <width>x<height>, Numbered <start> to <end>):",
                "tool_message_text_fillers": ["width", "height", "start", "end"],
                "prompt_type": "pr_original"
            }

        }

def show_image(image_path: str, **kwargs):
    """
    Returns the input image. It is then fed to the MLLM again so it can spot details that were missed earlier.
    :return: The input image.
    """
    return image_path

def zoom_in(image_path, bbox_2d, padding=(0.1,0.1), min_pixels=None, max_pixels=None, bbox_type:str=None,
            adaptive_padding_threshold:int=600.0):
    """
    Crop the image based on the bounding box coordinates.
    """
    image = Image.open(image_path)
    img_x, img_y = image.size

    input_height, input_width = get_resized_image_scales(img_y, img_x, min_pixels, max_pixels)

    logger.info(f"in zoom_in: original size: {img_x}x{img_y}")
    logger.info(f"in zoom_in: small size: {input_width}x{input_height}")

    if adaptive_padding_threshold is not None:
        padding_tr = (adaptive_padding_threshold/input_width,adaptive_padding_threshold/input_height)
        padding = (min(padding[0],padding_tr[0]),min(padding[1],padding_tr[1]))

    if bbox_type == "relative":
        if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
            normalized_bbox_2d = (float(bbox_2d[0])-padding[0],
                                  float(bbox_2d[1])-padding[1],
                                  float(bbox_2d[2])+padding[0],
                                  float(bbox_2d[3])+padding[1])
        else:
            raise ValueError(f"Invalid bounding box coordinates: {bbox_2d}. They should be floating point values in [0,1].")
    if bbox_type == "absolute":
        if isinstance(bbox_2d[0], int) and isinstance(bbox_2d[1], int) and isinstance(bbox_2d[2], int) and isinstance(bbox_2d[3], int):
            normalized_bbox_2d = (float(bbox_2d[0])/input_width-padding[0],
                                  float(bbox_2d[1])/input_height-padding[1],
                                  float(bbox_2d[2])/input_width+padding[0],
                                  float(bbox_2d[3])/input_height+padding[1])
        else:
            raise ValueError(f"Invalid bounding box coordinates: {bbox_2d}. They should be integers >= 0.")

    if bbox_type is None:
        if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
            normalized_bbox_2d = (float(bbox_2d[0])-padding[0],
                                  float(bbox_2d[1])-padding[1],
                                  float(bbox_2d[2])+padding[0],
                                  float(bbox_2d[3])+padding[1])
        else:
            normalized_bbox_2d = (float(bbox_2d[0]) / input_width - padding[0],
                                  float(bbox_2d[1]) / input_height - padding[1],
                                  float(bbox_2d[2]) / input_width + padding[0],
                                  float(bbox_2d[3]) / input_height + padding[1])

    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 =min(max(0, normalized_x1), 1)
    normalized_y1 =min(max(0, normalized_y1), 1)
    normalized_x2 =min(max(0, normalized_x2), 1)
    normalized_y2 =min(max(0, normalized_y2), 1)
    cropped_img = image.crop((int(normalized_x1*img_x),
                              int(normalized_y1*img_y),
                              int(normalized_x2*img_x),
                              int(normalized_y2*img_y)))
    w, h = cropped_img.size
    assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"

    return cropped_img

def select_frames():
    raise NotImplementedError


TOOL_TEMPLATES = {
    "zoom_in": {
        "json": {
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
            }
        },
        "callable": zoom_in},

    "select_frames": {
        "json": {
            "type": "function",
            "function": {
                "name": "select_frames",
                "description": "Select frames from a video.",
                "parameters": {
                            "type": "object",
                            "properties": {
                                "target_frames": {
                                    "type": "array",
                                    "description": "List of frame indices to select from the video (no more than 8 frames in total).",
                                    "items": {
                                        "type": "integer",
                                        "description": "Frame index from 1 to 16."
                                    }
                                }
                            },
                            "required": ["target_frames"]
                }
            }
        },
        "callable": select_frames},

    "show_image": {
        "json": {
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
        },
        "callable": show_image},
}

TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"

class Message:
    def __init__(self, image_position:str, text_message:str, text_fillers: list):
        self.image_position = image_position
        self.text_message = text_message
        self.text_fillers = text_fillers

    def fill_message(self, params:dict):
        filled_message = self.text_message
        for filler in self.text_fillers:
            if filler in params:
                filled_message = filled_message.replace(f"<{filler}>", str(params[filler]))
            else:
                raise ValueError(f"Missing filler: {filler}, we are only given {params}")
        return filled_message

    def get_content(self, params):
        content = []
        if self.image_position == "first":
            content.append({"text": None, "type": "image"})
            content.append({'text': self.fill_message(params), 'type': 'text'})
        elif self.image_position == "last":
            content.append({'text': self.fill_message(params), 'type': 'text'})
            content.append({"text": None, "type": "image"})
        else:
            raise ValueError(f"Invalid image position: {self.image_position}. Should be 'first' or 'last'")
        return content



class Tool:
    def __init__(self, name:str, template_name: str, json_customization: dict, message:Message, tool_hparams: dict):
        self.name = name

        self.json_customization = json_customization

        self.tool_dict = copy.deepcopy(TOOL_TEMPLATES[template_name]["json"])
        for path_to_key, value in self.json_customization.items():
            keys = path_to_key.split(",")
            #print(f"in Tool: keys: {keys}")
            #print(f"in Tool: {self.tool_dict}")

            last_container = reduce(operator.getitem, keys[:-1], self.tool_dict)
            last_container[keys[-1]] = value

        self.message = message

        self.tool_hparams = tool_hparams
        self.callable_function = partial(TOOL_TEMPLATES[template_name]["callable"], **self.tool_hparams)

    def get_tool_dict(self) -> dict:
        return self.tool_dict

    def call_tool(self, tool_params: dict, save_path: str = None):

        tool_args = tool_params['arguments']

        actual_tool_args = tool_args.copy()

        actual_tool_args.pop("target_image")

        if not (1 <= int(tool_args['target_image']) <= len(tool_params['image_paths'])):
            raise ValueError(f"target_image position out of bounds: {tool_args['target_image']}")

        actual_tool_args["image_path"] = tool_params['image_paths'][int(tool_args['target_image']) - 1]

        #else:
        #    raise ValueError(f"Invalid tool name: {tool_name}")

        # output_description += format_reminder

        new_image = self.callable_function(**actual_tool_args)

        img_x, img_y = new_image.size

        input_height, input_width = get_resized_image_scales(img_y, img_x, self.tool_hparams["min_pixels"], self.tool_hparams["max_pixels"])


        message_args = tool_args.copy()
        message_args["width"] = input_width
        message_args["height"] = input_height

        if isinstance(new_image, str):
            new_image_path = new_image
            new_image_id = str(uuid.uuid4())
        else:
            new_image_id = str(uuid.uuid4())
            new_image_name = f"{new_image_id}.png"
            new_image_path = os.path.join(save_path, "generated_images", new_image_name)
            new_image.save(new_image_path)

        logger.info(tool_args)
        logger.info(self.message.get_content(message_args))

        return {
            "original_image_path": tool_params['image_paths'][0],
            "tool_input_image_path": actual_tool_args["image_path"],
            "tool_name": self.name,
            "tool_input": tool_args,

            "new_image_path": new_image_path,
            "new_image_id": new_image_id,
            "output_message": self.message.get_content(message_args),
        }

def extract_tool(text:str, tool_start:str = TOOL_START, tool_end:str = TOOL_END, strict:bool = False, tool_start_position:str = "last"):
    logger.info(f"text from which tool should be extracted: {text}")

    no_tool_starts = text.count(tool_start)
    logger.info(f"found {no_tool_starts} tool starts")

    no_tool_ends = text.count(tool_end)
    logger.info(f"found {no_tool_ends} tool ends")

    ends_with_tool_end = text.endswith(tool_end)
    logger.info(f"tool_end is at the end: {ends_with_tool_end}")

    if strict:
        logger.info(f"tool extraction is strict!")
        assert no_tool_starts == 1, f"Tool call failed: there should be a single '{tool_start}', but found {no_tool_starts} instead"
        assert no_tool_ends == 1, f"Tool call failed: there should be a single '{tool_end}', but found {no_tool_ends} instead"
        assert ends_with_tool_end, f"Tool call failed: generated text should stop with '{tool_end}'"

    if tool_start_position == "first":
        after_tool_start = text.split(tool_start)[1]
    elif tool_start_position == "last":
        after_tool_start = text.split(tool_start)[-1]
    else:
        raise ValueError(f"tool_start_position {tool_start_position} not supported. choose from 'first', 'last'")
    tool_content = after_tool_start.split(tool_end)[0]
    logger.info(f"extracted tool: {tool_content}")

    return json.loads(tool_content)




