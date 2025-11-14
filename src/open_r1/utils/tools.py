import json
from PIL import Image
import uuid
import os
from functools import partial
from qwen_vl_utils import smart_resize

from .logger import get_logger
from .utils import get_resized_image_scales

# Get logger for this module
logger = get_logger(__name__)

TOOL_CONFIGS = {
            "PR_crop": {
                "tool_name": "crop_image",
                "tool_description": "Zoom in on the image based on the bounding box coordinates. It is useful when the object or text in the image is too small to be seen.",
                "tool_parameter_descriptions": {"bbox_2d": "coordinates for bounding box of the area you want to zoom in. Values should be within [0.0,1.0]."},
                "tool_message_image_pos": "last",
                "tool_message_text_message": "\nHere is the cropped image (Image Size: <width>x<height>):",
                "tool_message_text_fillers": ["width", "height"],
                "prompt_type": "pr_adapted"
        },
            "PR_crop_original": {
                "tool_name": "crop_image_normalized",
                "tool_description": "Zoom in on the image based on the bounding box coordinates. It is useful when the object or text in the image is too small to be seen.",
                "tool_parameter_descriptions": {"bbox_2d": "coordinates for bounding box of the area you want to zoom in. Values should be within [0.0,1.0]."},
                "tool_message_image_pos": "last",
                "tool_message_text_message": "\nHere is the cropped image (Image Size: <width>x<height>):",
                "tool_message_text_fillers": ["width", "height"],
                "prompt_type": "pr_original"
        },
            "PR_zoom_in_old":
            {
                "tool_name": "zoom_in",
                "tool_description": "Zoom in on the image based on the bounding box coordinates.",
                "tool_parameter_descriptions": {
                    "bbox_2d": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image."},
                "tool_message_image_pos": "last",
                "tool_message_text_message": "\nHere is the cropped image (Image Size: <width>x<height>):",
                "tool_message_text_fillers": ["width", "height"],
                "prompt_type": "pr_adapted"
            },
            "PR_zoom_in_old_exploration":
            {
                "tool_name": "zoom_in",
                "tool_description": "Zoom in on the image based on the bounding box coordinates.",
                "tool_parameter_descriptions": {
                    "bbox_2d": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image."},
                "tool_message_image_pos": "last",
                "tool_message_text_message": "\nHere is the cropped image (Image Size: <width>x<height>):",
                "tool_message_text_fillers": ["width", "height"],
                "prompt_type": "pr_adapted"
            },
            "PR_zoom_in_new":
                {
                    "tool_name": "zoom_in",
                    "tool_description": "Zoom in on the image based on the bounding box coordinates.",
                    "tool_parameter_descriptions": {
                        "bbox_2d": "normalized coordinates for bounding box of the region you want to zoom in. Values should be within [0.0,1.0]."},
                    "tool_message_image_pos": "last",
                    "tool_message_text_message": "\nHere is the cropped image (Image Size: <width>x<height>):",
                    "tool_message_text_fillers": ["width", "height"],
                    "prompt_type": "pr_adapted"
                },
            "first_own_training":
                {
                    "tool_name": "zoom_in",
                    "tool_description": "Zoom in on the image based on the bounding box coordinates.",
                    "tool_parameter_descriptions": {
                        "bbox_2d": "normalized coordinates for bounding box of the region you want to zoom in. Values should be within [0.0,1.0]."},
                    "tool_message_image_pos": "first",
                    "tool_message_text_message": "Here is image <target_image> zoomed in at <bbox_2d>.",
                    "tool_message_text_fillers": ["target_image", "bbox_2d"],
                    "prompt_type": "pr_adapted"
                },
            "no_tool": {
                "tool_name": "",
                "tool_description": "",
                "tool_parameter_descriptions": {},
                "tool_message_image_pos": "",
                "tool_message_text_message": "",
                "tool_message_text_fillers": [],
                "prompt_type": "no_tool"
            },
            "select_frames": {
                "tool_name": "select_frames",
                "tool_description": "Select frames from a video.",
                "tool_parameter_descriptions": {
                        "target_frames": "List of frame indices to select from the video (no more than 8 frames in total)."
                },
                "tool_message_image_pos": "last",
                "tool_message_text_message": "\nHere are the selected frames (Frame Size: <width>x<height>, Numbered <start> to <end>):",
                "tool_message_text_fillers": ["width", "height", "start", "end"],
                "prompt_type": "pr_original"

            }

        }



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

CROP_IMAGE = {
        "type": "function",
        "function": {
            "name": "crop_image",
            "description": "Zoom in on the image based on the bounding box coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "description": "coordinates for bounding box of the area you want to zoom in. Values should be within [0.0,1.0].",
                        "items": {
                            "type": "number",
                        }
                    },
                    "target_image":{
                        "type": "number",
                        "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."
                    }
                },
                "required": ["bbox_2d", "target_image"]
            }
        }
    }

SELECT_FRAMES = {
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
    def __init__(self, name:str, description:str, message:Message, parameter_descriptions:dict, tool_hparams: dict):
        self.name = name
        self.description = description
        self.parameter_descriptions = parameter_descriptions
        self.message = message

        self.tool_hparams = tool_hparams


        if self.name == "show_image":
            self.callable_function = show_image
            self.tool_dict = SHOW_IMAGE
        elif self.name == "zoom_in":
            self.callable_function = partial(zoom_in,
                                             min_pixels=self.tool_hparams["min_pixels"] if "min_pixels" in self.tool_hparams.keys() else None,
                                             max_pixels=self.tool_hparams["max_pixels"] if "max_pixels" in self.tool_hparams.keys() else None)
            self.tool_dict = ZOOM_IN
        elif self.name == "crop_image":
            self.callable_function = partial(crop_image,
                                             min_pixels=self.tool_hparams["min_pixels"] if "min_pixels" in self.tool_hparams.keys() else None,
                                             max_pixels=self.tool_hparams["max_pixels"] if "max_pixels" in self.tool_hparams.keys() else None)
            self.tool_dict = CROP_IMAGE
        elif self.name == "crop_image_normalized":
            self.callable_function = partial(crop_image,
                                             min_pixels=self.tool_hparams["min_pixels"] if "min_pixels" in self.tool_hparams.keys() else None,
                                             max_pixels=self.tool_hparams["max_pixels"] if "max_pixels" in self.tool_hparams.keys() else None)
            self.tool_dict = CROP_IMAGE
            self.tool_dict["function"]["name"] = "crop_image_normalized"
        elif self.name == "select_frames":
            self.callable_function = select_frames
            self.tool_dict = SELECT_FRAMES
        else:
            raise ValueError(f"Invalid tool name: {name}")

        self.tool_dict["function"]["description"] = self.description
        for param_name, param_desc in self.parameter_descriptions.items():
            self.tool_dict["function"]["parameters"]["properties"][param_name]["description"] = param_desc

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

        #if self.tool_hparams["min_pixels"] is None and self.tool_hparams["max_pixels"] is None:
        #    input_height, input_width = img_y, img_x
        #elif self.tool_hparams["min_pixels"] is None and self.tool_hparams["max_pixels"] is not None:
        #    input_height, input_width = smart_resize(height=img_y, width=img_x,
        #                                             max_pixels=self.tool_hparams["max_pixels"])
        #elif self.tool_hparams["min_pixels"] is not None and self.tool_hparams["max_pixels"] is None:
        #    input_height, input_width = smart_resize(height=img_y, width=img_x,
        #                                             min_pixels=self.tool_hparams["min_pixels"])
        #elif self.tool_hparams["min_pixels"] is not None and self.tool_hparams["max_pixels"] is not None:
        #    input_height, input_width = smart_resize(height=img_y, width=img_x,
        #                                             min_pixels=self.tool_hparams["min_pixels"],
        #                                             max_pixels=self.tool_hparams["max_pixels"])
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


def extract_tool(text:str, tool_start:str = TOOL_START, tool_end:str = TOOL_END):

    after_last_tool_start = text.split(tool_start)[-1]
    assert tool_end in after_last_tool_start, "no tool_end present"
    logger.info(f"extract tool: {after_last_tool_start.split(tool_end)[0]}")
    return json.loads(after_last_tool_start.split(tool_end)[0])



def show_image(image_path: str, **kwargs):
    """
    Returns the input image. It is then fed to the MLLM again so it can spot details that were missed earlier.
    :return: The input image.
    """
    return image_path

def zoom_in(image_path, bbox_2d, padding=(0.1,0.1), min_pixels=None, max_pixels=None):
    """
    Crop the image based on the bounding box coordinates.
    """
    image = Image.open(image_path)
    img_x, img_y = image.size

    input_height, input_width = get_resized_image_scales(img_y, img_x, min_pixels, max_pixels)

    ## TODO: very unpythonic way, because smart_resize's default value is given by a global variable
    #if min_pixels is None and max_pixels is None:
    #    input_height, input_width = img_y, img_x
    #elif min_pixels is None and max_pixels is not None:
    #    input_height, input_width = smart_resize(height=img_y, width=img_x,
    #                                             max_pixels=max_pixels)
    #elif min_pixels is not None and max_pixels is None:
    #    input_height, input_width = smart_resize(height=img_y, width=img_x,
    #                                             min_pixels=min_pixels)
    #elif min_pixels is not None and max_pixels is not None:
    #    input_height, input_width = smart_resize(height=img_y, width=img_x,
    #                                             min_pixels=min_pixels,
    #                                             max_pixels=max_pixels)

    logger.info(f"original size: {img_x}x{img_y}")
    logger.info(f"small size: {input_width}x{input_height}")

    padding_tr = (600.0/input_width,600.0/input_height)
    padding = (min(padding[0],padding_tr[0]),min(padding[1],padding_tr[1]))

    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0])-padding[0],
                              float(bbox_2d[1])-padding[1],
                              float(bbox_2d[2])+padding[0],
                              float(bbox_2d[3])+padding[1])
    else:
        normalized_bbox_2d = (float(bbox_2d[0])/input_width-padding[0],
                              float(bbox_2d[1])/input_height-padding[1],
                              float(bbox_2d[2])/input_width+padding[0],
                              float(bbox_2d[3])/input_height+padding[1])
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

def crop_image(image_path, bbox_2d, padding=0.1, min_pixels=None, max_pixels=None):
    """
    Crop the image based on the bounding box coordinates.
    """
    image = Image.open(image_path)
    img_x, img_y = image.size

    input_height, input_width = get_resized_image_scales(img_y, img_x, min_pixels, max_pixels)

    ## TODO: very unpythonic way, because smart_resize's default value is given by a global variable
    #if min_pixels is None and max_pixels is None:
    #    input_height, input_width = img_y, img_x
    #elif min_pixels is None and max_pixels is not None:
    #    input_height, input_width = smart_resize(height=img_y, width=img_x,
    #                                             max_pixels=max_pixels)
    #elif min_pixels is not None and max_pixels is None:
    #    input_height, input_width = smart_resize(height=img_y, width=img_x,
    #                                             min_pixels=min_pixels)
    #elif min_pixels is not None and max_pixels is not None:
    #    input_height, input_width = smart_resize(height=img_y, width=img_x,
    #                                             min_pixels=min_pixels,
    #                                             max_pixels=max_pixels)

    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0])-padding,
                              float(bbox_2d[1])-padding,
                              float(bbox_2d[2])+padding,
                              float(bbox_2d[3])+padding)
    else:
        normalized_bbox_2d = (float(bbox_2d[0])/input_width-padding,
                              float(bbox_2d[1])/input_height-padding,
                              float(bbox_2d[2])/input_width+padding,
                              float(bbox_2d[3])/input_height+padding)
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 =min(max(0, normalized_x1), 1)
    normalized_y1 =min(max(0, normalized_y1), 1)
    normalized_x2 =min(max(0, normalized_x2), 1)
    normalized_y2 =min(max(0, normalized_y2), 1)
    cropped_img = image.crop((normalized_x1*img_x,
                              normalized_y1*img_y,
                              normalized_x2*img_x,
                              normalized_y2*img_y))
    w, h = cropped_img.size
    assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"

    return cropped_img

def select_frames():
    raise NotImplementedError
