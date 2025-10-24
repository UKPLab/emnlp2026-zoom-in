from open_r1.utils.masker import get_boundaries_tokenized
import torch
from trl.data_utils import maybe_apply_chat_template
from open_r1.utils.masker import get_intervals, Masker
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from open_r1.utils.logger import setup_project_logging
from open_r1.utils.parser import ParsedTokenized, rescale, false_intervals_list
import sys
import PIL
import copy
import time

logger = setup_project_logging(None)

inputs = {
    "q": torch.tensor([[0,151644,77091,198,
                         ]], dtype=torch.long),
    "qa": torch.tensor([[0,151644,77091,198,
                         1,1]], dtype=torch.long),
"qat": torch.tensor([[0,151644,77091,198,
                           1,1,
                           151645, 2,2,2,151644,77091,198,
                           ]], dtype=torch.long),
    "qata": torch.tensor([[0,151644,77091,198,
                           1,1,
                           151645, 2,2,2,151644,77091,198,
                           3,3,3,3  ]], dtype=torch.long),
"qatat": torch.tensor([[0,151644,77091,198,
                         1,1,
                         151645,2,2,2,151644,77091,198,
                         3,3,3,3,
                         151645,4,4,4,4,4,151644,77091,198,
                         ]], dtype=torch.long),
"qatata": torch.tensor([[0,151644,77091,198,
                         1,1,
                         151645,2,2,2,151644,77091,198,
                         3,3,3,3,
                         151645,4,4,4,4,4,151644,77091,198,
                         5,5,5,5,5,5 ]], dtype=torch.long),
"q'": torch.tensor([[0,151644,77091,]], dtype=torch.long),
"truncate_right": torch.tensor([[0,151644,77091,198,
                           1,1,
                           151645, 2,2,2,
                           ]], dtype=torch.long),
}

user_simple = {
    'role': 'user',
    'content': [
        {'type': 'image', 'text': None},
        {'type': 'text', 'text': "Hello World"}
        ]
    }

user_simple_swapped = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': "Hello World"},
        {'type': 'image', 'text': None},
        ]

}

user_adversarial = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': "\nHello World"},
        {'type': 'image', 'text': None}
        ]
    }

user_new = {
    'role': 'user',
    'content': [
        {'type': 'image', 'text': None},
        {'type': 'text', 'text': "Hello World hello"}
        ]
    }

assistant_simple = {'role': 'assistant',
     'content': [
         {'type': 'text', 'text': "Hello World"}
     ]
    }

assistant_tool = {'role': 'assistant',
     'content': [
         {'type': 'text', 'text': "Hello <tool_call>World</tool_call>"}
     ]
}

assistant_adversarial = {'role': 'assistant',
     'content': [
         {'type': 'text', 'text': "\nHello World"}
     ]
    }

assistant_box = {'role': 'assistant',
     'content': [
         {'type': 'text', 'text': "Hello World \\boxed{A}"}
     ]
    }

assistant_box_newline = {'role': 'assistant',
     'content': [
         {'type': 'text', 'text': "Hello.\n\n\\boxed{A}"}
     ]
    }

token_dict = {1124: " \\",
              79075: "boxed",
              90: "{",
              92: "}",
              382: ".\n\n",
              59: "\\"
              }

default_image = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/RL_data_without_video/images/a15ab079-1311-444c-8d50-1ba3544b6e06-0.jpg"

test_suite = [
    {"name": "mask until answer",
     "type": "model_tool_call,full_tool_user_response,model_non_answer",
     "input": [[user_simple, assistant_tool, user_simple, assistant_box],
               [user_new, assistant_tool, user_simple, assistant_box_newline]],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [[151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,220, 0,0,0, 0, # three zeros represent 151657, 10134, 151658,
                  0,
                  0, 0, 0, 0] + [0]*1624 + [0, 0, 0, 0, # user prompt
                  0,
                  0, 0, 0, 9707, 4337,        # assistant response
                  151643],[
                  151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 23811, 151645, # user prompt
                  198,
                  151644,77091,198,9707,220, 0,0,0, 0, # three zeros represent 151657, 10134, 151658,
                  0,
                  0, 0, 0, 0] + [0]*1624 + [0, 0, 0, 0, # user prompt
                  0,
                  0, 0, 0, 9707, 4337        # assistant response
     ]],
     "masked_indices": [[[1649, 3289]],
                        [[1650, 3290]]],
    "solution_masked_images": [1]},

{"name": "successful tool input_mask mask_tool_call right_pad",
     "type": "model_tool_call,full_tool_user_response",
     "input": [[user_simple, assistant_tool, user_simple, assistant_simple],
               [user_new, assistant_tool, user_simple, assistant_simple]],
     "images": [[PIL.Image.open(default_image), PIL.Image.open(default_image)],
                [PIL.Image.open(default_image), PIL.Image.open(default_image)]],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     # TODO: this is wrong!
     "solution": [[151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,220, 0,0,0, 0, # three zeros represent 151657, 10134, 151658,
                  0,
                  0, 0, 0, 0] + [0]*1624 + [0, 0, 0, 0, # user prompt
                  0,
                  0, 0, 0, 9707, 4337,        # assistant response
                  151643],[
                  151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 23811, 151645, # user prompt
                  198,
                  151644,77091,198,9707,220, 0,0,0, 0, # three zeros represent 151657, 10134, 151658,
                  0,
                  0, 0, 0, 0] + [0]*1624 + [0, 0, 0, 0, # user prompt
                  0,
                  0, 0, 0, 9707, 4337        # assistant response
     ]],
     "masked_indices": [[[1649, 3289]],
                        [[1650, 3290]]],
    "solution_masked_images": [1]},
    {"name": "successful tool input_mask mask_tool_call left_pad",
     "type": "model_tool_call,full_tool_user_response",
     "input": [[user_simple, assistant_tool, user_simple, assistant_simple],
               [user_new, assistant_tool, user_simple, assistant_simple]],
     "images": [[PIL.Image.open(default_image), PIL.Image.open(default_image)],
                [PIL.Image.open(default_image), PIL.Image.open(default_image)]],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [[151643, 151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,220, 0,0,0, 0, # three zeros represent 151657, 10134, 151658,
                  0,
                  0, 0, 0, 0] + [0]*1624 + [0, 0, 0, 0, # user prompt
                  0,
                  0, 0, 0, 9707, 4337        # assistant response
                  ],[
                  151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 23811, 151645, # user prompt
                  198,
                  151644,77091,198,9707,220, 0,0,0, 0, # three zeros represent 151657, 10134, 151658,
                  0,
                  0, 0, 0, 0] + [0]*1624 + [0, 0, 0, 0, # user prompt
                  0,
                  0, 0, 0, 9707, 4337        # assistant response
     ]],
     "masked_indices": [[[1650, 3290]],
                        [[1650, 3290]]],
    "solution_masked_images": [1]},
    {"name": "successful tool input_mask mask_tool_call",
     "type": "model_tool_call,full_tool_user_response",
     "input": [user_simple, assistant_tool, user_simple, assistant_simple],
     "images": [PIL.Image.open(default_image), PIL.Image.open(default_image)],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,220, 0,0,0, 0, # three zeros represent 151657, 10134, 151658,
                  0,
                  0, 0, 0, 0] + [0]*1624 + [0, 0, 0, 0, # user prompt
                  0,
                  0, 0, 0, 9707, 4337        # assistant response
                  ],
    "masked_indices": [[[1649, 3289]]],
    "solution_masked_images": [1]},
    {"name": "successful tool input_mask empty_response",
     "type": "tool_user_response",
     "input": [user_simple, assistant_simple, user_simple, assistant_simple],
     "images": [PIL.Image.open(default_image), PIL.Image.open(default_image)],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "right",
     "right_boundary": "left",
     "solution": [151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,4337,151645,
                  198,
                  151644, 872, 198, 0] + [0]*1624 + [0, 0, 0, 151645, # user prompt
                  198,
                  151644, 77091, 198, 9707, 4337        # assistant response
                  ],
    "solution_masked_images": [1]},
    {"name": "successful tool input_mask only_mask_image",
     "type": "image",
     "input": [user_simple, assistant_simple, user_simple, assistant_simple],
     "images": [PIL.Image.open(default_image), PIL.Image.open(default_image)],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,4337,151645,
                  198,
                  151644, 872, 198, 0] + [0]*1624 + [0, 9707, 4337, 151645, # user prompt
                  198,
                  151644, 77091, 198, 9707, 4337        # assistant response
                  ],
     "solution_masked_images": [1]},

    {"name": "successful tool input_mask only_mask_image",
     "type": "image",
     "input": [user_simple, assistant_simple, user_simple_swapped, assistant_simple],
     "images": [PIL.Image.open(default_image), PIL.Image.open(default_image)],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,4337,151645,
                  198,
                  151644, 872, 198, 9707, 4337, 0] + [0]*1624 + [0, 151645, # user prompt
                  198,
                  151644, 77091, 198, 9707, 4337        # assistant response
                  ],
     "solution_masked_images": [1]},
    {"name": "successful tool input_mask only_mask_image_pad",
     "type": "image_pad",
     "input": [user_simple, assistant_simple, user_simple, assistant_simple],
     "images": [PIL.Image.open(default_image), PIL.Image.open(default_image)],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,4337,151645,
                  198,
                  151644, 872, 198, 151652] + [0]*1624 + [151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644, 77091, 198, 9707, 4337        # assistant response
                  ],
     "solution_masked_images": [1]},
    {"name": "successful tool input_mask only_mask_image_pad",
     "type": "image_pad",
     "input": [user_simple, assistant_simple, user_simple_swapped, assistant_simple],
     "images": [PIL.Image.open(default_image), PIL.Image.open(default_image)],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652] + [151655]*1624 + [151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,4337,151645,
                  198,
                  151644, 872, 198, 9707, 4337, 151652] + [0]*1624 + [151653, 151645, # user prompt
                  198,
                  151644, 77091, 198, 9707, 4337        # assistant response
                  ],
     "solution_masked_images": [1]},
    {"name": "no tool loss_mask",
     "type": "everything_except_model_generation",
     "input": [user_simple, assistant_simple],
     "left_boundary_tokens": [151644, 77091, 198],
     "right_boundary_tokens": [151645],
     "left_boundary": "right",
     "right_boundary": "left",
     "solution": [0,0,0,0,0,0,0,0,0,0, # system prompt
                  0,
                  0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,0,9707,4337
                  ],
     "solution_masked_images": []},
    {"name": "no tool input_mask",
     "type": "full_tool_user_response",
     "input": [user_simple, assistant_simple],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652, 151655, 151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,4337
                  ],
     "solution_masked_images": []},
    {"name": "failed tool loss_mask",
     "type": "everything_except_model_generation",
     "input": [user_simple, assistant_simple, user_simple],
     "left_boundary_tokens": [151644, 77091, 198],
     "right_boundary_tokens": [151645],
     "left_boundary": "right",
     "right_boundary": "left",
     "solution": [0,0,0,0,0,0,0,0,0,0, # system prompt
                  0,
                  0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,0,9707,4337,0, # assistant response
                  0,
                  0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,0              # assistant response
                  ],
     "solution_masked_images": [1]},
    {"name": "failed tool input_mask",
     "type": "full_tool_user_response",
     "input": [user_simple, assistant_simple, user_simple],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652, 151655, 151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,4337,0,
                  0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, # user prompt
                  0,
                  0, 0, 0              # assistant response
                  ],
     "solution_masked_images": []},
    {"name": "successful tool loss_mask",
     "type": "everything_except_model_generation",
     "input": [user_simple, assistant_simple, user_simple, assistant_simple],
     "left_boundary_tokens": [151644, 77091, 198],
     "right_boundary_tokens": [151645],
     "left_boundary": "right",
     "right_boundary": "left",
     "solution": [0,0,0,0,0,0,0,0,0,0, # system prompt
                  0,
                  0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,0,9707,4337,0, # assistant response
                  0,
                  0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,0,9707,4337    # assistant response
                  ],
     "solution_masked_images": []},
    {"name": "successful tool loss_mask right_pad",
     "type": "everything_except_model_generation",
     "input": [[user_simple, assistant_simple, user_simple, assistant_simple],
        [user_new, assistant_simple, user_simple, assistant_simple]],
     "left_boundary_tokens": [151644, 77091, 198],
     "right_boundary_tokens": [151645],
     "left_boundary": "right",
     "right_boundary": "left",
     "solution": [[
                  0,0,0,0,0,0,0,0,0,0, # system prompt
                  0,
                  0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,0,9707,4337,0, # assistant response
                  0,
                  0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,0,9707,4337,   # assistant response
                  0                  # pad
                  ],[
                  0,0,0,0,0,0,0,0,0,0, # system prompt
                  0,
                  0,0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,0,9707,4337,0, # assistant response
                  0,
                  0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,0,9707,4337,   # assistant response
                  ]
                  ],
     "solution_masked_images": []},
    {"name": "successful tool input_mask",
     "type": "full_tool_user_response",
     "input": [user_simple, assistant_simple, user_simple, assistant_simple],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652, 151655, 151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,4337,0,
                  0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, # user prompt
                  0,
                  0, 0, 0, 9707, 4337        # assistant response
                  ],
     "solution_masked_images": []},
    {"name": "adversarial assistant",
     "type": "everything_except_model_generation",
     "input": [user_simple, assistant_adversarial, user_simple, assistant_simple],
     "left_boundary_tokens": [151644, 77091, 198],
     "right_boundary_tokens": [151645],
     "left_boundary": "right",
     "right_boundary": "left",
     "solution": [0,0,0,0,0,0,0,0,0,0, # system prompt
                  0,
                  0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,271,9707,4337,0, # assistant response
                  0,
                  0,0,0,0,0,0,0,0,0, # user prompt
                  0,
                  0,0,0,9707,4337    # assistant response
                  ],
     "solution_masked_images": []},
    {"name": "adversarial user",
     "type": "full_tool_user_response",
     "input": [user_simple, assistant_simple, user_adversarial, assistant_simple],
     "left_boundary_tokens": [151645, 198, 151644, 872, 198],
     "right_boundary_tokens": [151645, 198, 151644, 77091, 198],
     "left_boundary": "left",
     "right_boundary": "right",
     "solution": [151644,8948,198,2610,525,264,10950,17847,13,151645, # system prompt
                  198,
                  151644, 872, 198, 151652, 151655, 151653, 9707, 4337, 151645, # user prompt
                  198,
                  151644,77091,198,9707,4337,0,
                  0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, # user prompt
                  0,
                  0, 0, 0, 9707, 4337        # assistant response
                  ],
     "solution_masked_images": []},




]

truncate = 0

def eval_with_masker_class(test, tokenized):
    masker = Masker(verbose=True)

    types = test["type"].split(",")
    non_generation_mask = None
    for type_ in types:
        logger.info(f"type_: {masker.MASK_TYPES[type_]}")
        non_generation_mask = masker.get_mask(tokenized["input_ids"], mask_type=masker.MASK_TYPES[type_], mask=non_generation_mask)
    return non_generation_mask

def eval_with_parser_class(test, tokenized):
    #tokenized_copy = tokenized.copy()
    parser = ParsedTokenized(tokenized["input_ids"],
                             tokenized["attention_mask"],
                             tokenized["image_grid_thw"] if "image_grid_thw" in tokenized else None,
                             tokenized["pixel_values"] if "pixel_values" in tokenized else None,
                             verbose=True)

    #logger.info(f"parsed: {parser.parsed}")
    #logger.info(f"left pad: {parser.last_left_pad}")
    #logger.info(f"image features: {parser.image_features}")

    types = test["type"].split(",")
    non_generation_mask = torch.ones_like(tokenized["input_ids"])
    for type_ in types:
        non_generation_mask = parser.get_mask(mode=type_, mask=non_generation_mask)
    return non_generation_mask

def shorten_tokenized(test, tokenized, pad_token_id):
    logger.info(f"attention_mask: {tokenized['attention_mask']}")
    #tokenized_copy = tokenized.copy()
    parser = ParsedTokenized(tokenized["input_ids"],
                             tokenized["attention_mask"],
                             tokenized["image_grid_thw"] if "image_grid_thw" in tokenized else None,
                             tokenized["pixel_values"] if "pixel_values" in tokenized else None,
                             verbose=True)

    logger.info(parser.parsed)

    types = test["type"].split(",")
    full_types = [
    ]
    for type_ in types:
        if type_ == "model_tool_call":
            user_turn_range = None
            model_turn_range = [0,None]
        elif type_ == "everything_except_model_generation":
            user_turn_range = None
            model_turn_range = None
        elif type_ == "full_tool_user_response":
            user_turn_range = [1, 2]
            model_turn_range = [1,2]
        else:
            user_turn_range = [1,2]
            model_turn_range = None
        full_types.append({"mode": type_,
                   "user_turn_range": user_turn_range,
                   "model_turn_range": model_turn_range} )

    shorten_tokenized = parser.get_shortened_tokenized(full_types, pad_token_id,
                                                       padding_side= "right" if "right_pad" in test["name"] else "left")

    loss_mask = None
    if "everything_except_model_generation" in test["type"]:
        loss_mask = parser.get_mask(mode="everything_except_model_generation",
                               mask=torch.ones_like(tokenized["input_ids"]),
                               indices=None)

    return shorten_tokenized, loss_mask




def eval_direct(test, tokenized, truncate=0):

        user_token_boundaries = get_boundaries_tokenized(tokenized["input_ids"][:, :-truncate or None],
                                                                 torch.tensor(test["left_boundary_tokens"], dtype=torch.long),
                                                                 torch.tensor(test["right_boundary_tokens"], dtype=torch.long),
                                                         left_boundary=test["left_boundary"],
                                                         right_boundary=test["right_boundary"])

        logger.info(user_token_boundaries)
        logger.info(get_intervals(user_token_boundaries, kind="inclusion"))
        logger.info(get_intervals(user_token_boundaries, kind="exclusion"))

        if "input_mask" in test["name"]:

            non_generation_mask = torch.ones_like(tokenized["input_ids"][:, :-truncate or None])
            user_token_boundaries = get_intervals(user_token_boundaries, kind="inclusion")

            if "mask_tool_call" in test["name"]:
                assistant_token_boundaries = get_boundaries_tokenized(tokenized["input_ids"][:, :-truncate or None],
                                                                  left_boundary_tokens= torch.tensor([151644, 77091, 198], dtype=torch.long),
                                                                  right_boundary_tokens= torch.tensor([151645], dtype=torch.long),
                                                                  left_boundary= "right",
                                                                  right_boundary= "left",
                                                                  )
                assistant_token_boundaries = get_intervals(assistant_token_boundaries, kind="inclusion")
                for i, assistant_boundaries in enumerate(assistant_token_boundaries):
                    for boundary in assistant_boundaries:
                        logger.info(f"looking at: {tokenized["input_ids"][i, boundary[0]:boundary[1]]}")
                        sub_user_boundaries = get_boundaries_tokenized(
                            tokenized["input_ids"][i, boundary[0]:boundary[1]].unsqueeze(0),
                            torch.tensor([151657], dtype=torch.long),
                            torch.tensor([151658], dtype=torch.long),
                            left_boundary="left",
                            right_boundary="right")
                        sub_user_intervals = get_intervals(sub_user_boundaries, kind="inclusion")
                        for sub_user_boundary in sub_user_intervals[0]:
                            logger.info(f"in mask_tool_call, masking [{i}, {boundary[0]+sub_user_boundary[0]}:{boundary[0]+sub_user_boundary[1]}]")
                            non_generation_mask[i, boundary[0]+sub_user_boundary[0]:boundary[0]+sub_user_boundary[1]] = False
                        #non_generation_mask[i, boundary[0]:boundary[1]] = False


            for i, assistant_boundaries in enumerate(user_token_boundaries):
                for boundary in assistant_boundaries[1:]:
                    if "only_mask_image_pad" in test["name"]:
                        sub_user_boundaries = get_boundaries_tokenized(
                            tokenized["input_ids"][i, boundary[0]:boundary[1]].unsqueeze(0),
                            torch.tensor([151652], dtype=torch.long),
                            torch.tensor([151653], dtype=torch.long),
                            left_boundary="right",
                            right_boundary="left")

                        sub_user_intervals = get_intervals(sub_user_boundaries, kind="inclusion")
                        for sub_user_boundary in sub_user_intervals[0]:
                            logger.info(
                                f"in only_mask_image_pad, masking [{i}, {boundary[0] + sub_user_boundary[0]}:{boundary[0] + sub_user_boundary[1]}]")
                            non_generation_mask[i,
                            boundary[0] + sub_user_boundary[0]:boundary[0] + sub_user_boundary[1]] = False
                    elif "only_mask_image" in test["name"]:
                        sub_user_boundaries = get_boundaries_tokenized(tokenized["input_ids"][i, boundary[0]:boundary[1]].unsqueeze(0),
                                                                        torch.tensor([151652], dtype=torch.long),
                                                                        torch.tensor([151653], dtype=torch.long),
                                                                        left_boundary="left",
                                                                        right_boundary="right")

                        sub_user_intervals = get_intervals(sub_user_boundaries, kind="inclusion")
                        for sub_user_boundary in sub_user_intervals[0]:
                            logger.info(
                                f"in only_mask_image, masking [{i}, {boundary[0] + sub_user_boundary[0]}:{boundary[0] + sub_user_boundary[1]}]")
                            non_generation_mask[i, boundary[0]+sub_user_boundary[0]:boundary[0]+sub_user_boundary[1]] = False
                        #elif "mask_tool_call" in test["name"]:


                    else:
                        logger.info(
                            f"in else, masking [{i}, {boundary[0]}:{boundary[1]}]")
                        non_generation_mask[i, boundary[0]:boundary[1]] = False

        elif "loss_mask" in test["name"]:
            non_generation_mask = torch.ones_like(tokenized["input_ids"][:, :-truncate or None])
            user_token_boundaries = get_intervals(user_token_boundaries, kind="exclusion")
            for i, assistant_boundaries in enumerate(user_token_boundaries):
                for boundary in assistant_boundaries:
                    logger.info(f"masking [{i}, {boundary[0]}:{boundary[0]}]")
                    non_generation_mask[i, boundary[0]:boundary[1]] = False
        return non_generation_mask

def prepare(test):

        solution = test.pop('solution')
        logger.info(f"evaluating {test}")

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        if isinstance(test["input"][0], list):
            wrapped_example = [maybe_apply_chat_template({"prompt": inp}, processor,
                                                    add_generation_prompt=None,
                                                    return_assistant_tokens_mask=False, tools=None)["prompt"] for inp in test["input"]]
        else:
            wrapped_example = maybe_apply_chat_template({"prompt": test["input"]}, processor,
                                                    add_generation_prompt=None,
                                                    return_assistant_tokens_mask=False, tools=None)["prompt"]
        logger.info(f"wrapped: {repr(wrapped_example)}")



        tokenized = processor(
            text=wrapped_example,
            images=test["images"] if "images" in test.keys() else None,
            return_tensors="pt",
            padding=True,
            padding_side= "right" if "right_pad" in test["name"] else "left",
            add_special_tokens=False,
            return_offsets_mapping=False
        )

        if "images" in test.keys():
            reduced_images = [img for idx, img in enumerate(test["images"]) if idx not in test["solution_masked_images"]]
        else:
            reduced_images = None

        tokenized_reduced_images = processor(
            text="test",
            images = reduced_images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            return_offsets_mapping=False
        )

        logger.info(f"tokenized: \n{tokenized['input_ids']}")
        logger.info(f"initial attention mask: {tokenized['attention_mask']}")

        return tokenized, solution, processor.tokenizer.pad_token_id, tokenized_reduced_images

def check_correctness(mask, tokenized, solution, outcome=None):
    logger.info(mask)
    if outcome is None:
        outcome = mask * tokenized["input_ids"][:, :-truncate or None]
    logger.info(f"outcome: {outcome}")
    if isinstance(solution[0], list):
        solution = torch.tensor(solution, dtype=torch.long)[:, :-truncate or None]
    else:
        solution = torch.tensor([solution], dtype=torch.long)[:, :-truncate or None]

    logger.info(f"solution: {solution}")

    logger.info(torch.nonzero(outcome - solution, as_tuple=False))
    assert torch.equal(outcome, solution)

def check_correctness_image(shortened_image_grid_thw, shortened_pixel_values, tokenized_reduced_images):
    if "image_grid_thw" in tokenized_reduced_images:
        solution_image_grid_thw = tokenized_reduced_images['image_grid_thw']

        logger.info(f"shortened_image_grid_thw: {shortened_image_grid_thw}")
        logger.info(f"solution image_grid_thw: {solution_image_grid_thw}")
        assert torch.equal(shortened_image_grid_thw, solution_image_grid_thw)
    else:
        assert shortened_image_grid_thw is None

    if "pixel_values" in tokenized_reduced_images:
        solution_pixel_values = tokenized_reduced_images['pixel_values']
        logger.info(f"shortened_pixel_values: {shortened_pixel_values}")
        logger.info(f"solution pixel_values: {solution_pixel_values}")
        assert torch.equal(shortened_pixel_values, solution_pixel_values)
    else:
        assert shortened_pixel_values is None

def check_correctness_model_generation(model, tokenized, shorten_tokenized):
    logger.info(f"in model gen: tokenized: {tokenized}")
    tokenized.to("cuda")
    full_logits = model(**tokenized)
    logger.info(f"full_logits: {full_logits}")
    #logger.info(f"full_logits: {full_logits}")

def check_correctness_mask_overview(mask_overview, solution):

    outcome = mask_overview
    logger.info(f"mask_overview: {outcome}")
    logger.info(f"solution: {solution}")
    assert torch.equal(torch.tensor(outcome, dtype=torch.long), torch.tensor(solution, dtype=torch.long))



def run_test(test_for:str, test_name:str, evaluate_model=False):

    if evaluate_model:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",
                                                                   torch_dtype=torch.float16)
        model.to("cuda")



    if test_for == "mask":
        for test in test_suite:
            logger.info(f"testing {test['type']}")
            tokenized, solution, _, _ = prepare(test)
            if test_name == "masker_class":
                mask = eval_with_masker_class(test, tokenized)
            elif test_name == "direct":
                mask = eval_direct(test, tokenized, truncate=truncate)
            elif test_name == "parser":
                mask = eval_with_parser_class(test, tokenized)
            check_correctness(mask, tokenized, solution)
    if test_for == "shorten":
        for test in test_suite:
            logger.info(f"testing {test['type']}")
            tokenized, solution, pad_token_id, tokenized_reduced_images = prepare(test)

            if isinstance(solution[0], list):
                short_solution = [[idx for idx in row if idx != 0] for row in solution]
            else:
                short_solution = [idx for idx in solution if idx != 0]

            shorten_result, loss_mask = shorten_tokenized(test, copy.deepcopy(tokenized), pad_token_id)
            shortened_input_ids = shorten_result["input_ids"]
            shortened_attention_mask = shorten_result["attention_mask"]
            shortened_image_grid_thw = shorten_result["image_grid_thw"]
            shortened_pixel_values = shorten_result["pixel_values"]
            shortened_indices = shorten_result["indices"]
            mask_overview = shorten_result["mask_intervals_ignoring_padding"]

            shortened_tokenized = {"input_ids": shortened_input_ids, "attention_mask": shortened_attention_mask,
                                   "image_grid_thw": shortened_image_grid_thw, "pixel_values": shortened_pixel_values}
            logger.info(f"short solution: {short_solution}")
            # check that text was correctly shortened
            check_correctness(None, tokenized, short_solution, shortened_input_ids)
            # check that images are correctly shortened
            check_correctness_image(shortened_image_grid_thw, shortened_pixel_values, tokenized_reduced_images)



            if "everything_except_model_generation" in test["type"]:
                if isinstance(solution[0], list):
                    mask_solution = [[idx if idx != pad_token_id else 0 for idx in row] for row in solution]
                else:
                    mask_solution = [idx if idx != pad_token_id else 0 for idx in solution]
                logger.info(f"mask_solution: {mask_solution}")
                check_correctness(loss_mask, tokenized, mask_solution)
            else:
                rescaled_input_ids = rescale(shortened_input_ids, shortened_indices, tokenized["input_ids"],
                                             tokenized["attention_mask"], pad_token_id)
                # check that rescaling to original dimension has worked
                check_correctness(None, tokenized, solution, rescaled_input_ids)

            if "masked_indices" in test.keys():
                check_correctness_mask_overview(mask_overview, test["masked_indices"])

            if evaluate_model:
                check_correctness_model_generation(model, tokenized, shortened_tokenized)
                sys.exit()


if __name__ == "__main__":
    run_test(test_for = "mask", test_name="parser", evaluate_model=False)







