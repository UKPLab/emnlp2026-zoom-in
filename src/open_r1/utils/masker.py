import torch
from .logger import get_logger

logger = get_logger(__name__)

# 151645, 198, 151644, 872, 198
class Masker:
    def __init__(self, verbose=False):
        self.im_end = [151645]
        self.im_start = [151644]
        self.model_start = [77091]#, 198]
        self.user_start = [872]#, 198]

        self.image_start = [151652]
        self.image_end = [151653]

        self.tool_start = [151657]
        self.tool_end = [151658]

        self.between_turns = [198]

        self.newline = [198]

        self.verbose = verbose

        self.MASK_TYPES = {
            "everything_except_model_generation": {"left_boundary_tokens": self.im_start + self.model_start,
                                                   "right_boundary_tokens": self.im_end,
                                                   "left_boundary": "right",
                                                   "right_boundary": "left",
                                                   "kind": "exclusion",
                                                   "start_interval": 0,
                                                   "sub": None,
                                                   "optional_left_boundary_end": self.newline,
                                                   "optional_right_boundary_end": None
                                                   },

            "full_tool_user_response": {
                "left_boundary_tokens": self.im_end + self.between_turns + self.im_start + self.user_start,
                "right_boundary_tokens": self.im_end + self.between_turns + self.im_start + self.model_start,
                "left_boundary": "left",
                "right_boundary": "right",
                "kind": "inclusion",
                "start_interval": 1,
                "sub": None,
                "optional_left_boundary_end": self.newline,
                "optional_right_boundary_end": self.newline
            },
            "tool_user_response": {
                "left_boundary_tokens": self.im_end + self.between_turns + self.im_start + self.user_start,
                "right_boundary_tokens": self.im_end + self.between_turns + self.im_start + self.model_start,
                "left_boundary": "right",
                "right_boundary": "left",
                "kind": "inclusion",
                "start_interval": 1,
                "sub": None,
                "optional_left_boundary_end": self.newline,
                "optional_right_boundary_end": self.newline
            },
            "image": {
                "left_boundary_tokens": self.im_end + self.between_turns + self.im_start + self.user_start,
                "right_boundary_tokens": self.im_end + self.between_turns + self.im_start + self.model_start,
                "left_boundary": "right",
                "right_boundary": "left",
                "kind": "inclusion",
                "start_interval": 1,
                "optional_left_boundary_end": self.newline,
                "optional_right_boundary_end": self.newline,
                "sub": {"left_boundary_tokens": self.image_start,
                        "right_boundary_tokens": self.image_end,
                        "left_boundary": "left",
                        "right_boundary": "right",
                        "kind": "inclusion",
                        "optional_left_boundary_end": None,
                        "optional_right_boundary_end": None
                        }
            },
            "image_pad": {
                "left_boundary_tokens": self.im_end + self.between_turns + self.im_start + self.user_start,
                "right_boundary_tokens": self.im_end + self.between_turns + self.im_start + self.model_start,
                "left_boundary": "right",
                "right_boundary": "left",
                "kind": "inclusion",
                "start_interval": 1,
                "optional_left_boundary_end": self.newline,
                "optional_right_boundary_end": self.newline,
                "sub": {"left_boundary_tokens": self.image_start,
                        "right_boundary_tokens": self.image_end,
                        "left_boundary": "right",
                        "right_boundary": "left",
                        "kind": "inclusion",
                        "optional_left_boundary_end": None,
                        "optional_right_boundary_end": None

                }
            },

            "model_tool_call": {
                "left_boundary_tokens": self.im_start + self.model_start,
                "right_boundary_tokens": self.im_end,
                "left_boundary": "right",
                "right_boundary": "left",
                "kind": "inclusion",
                "start_interval": 0,
                "optional_left_boundary_end": self.newline,
                "optional_right_boundary_end": None,
                "sub": {"left_boundary_tokens": self.tool_start,
                        "right_boundary_tokens": self.tool_end,
                        "left_boundary": "left",
                        "right_boundary": "right",
                        "kind": "inclusion",
                        "optional_left_boundary_end": None,
                        "optional_right_boundary_end": None

                        }
            },

        }

    def _get_sub_turn_mask(self, input_ids: torch.tensor, row: int, mask: torch.tensor, interval: list[int], mask_type:dict):
        """if mask_type == "only_mask_image_pad":
            left_boundary_tokens = self.image_start
            right_boundary_tokens = self.image_end
            left_boundary = "right"
            right_boundary = "left"
        elif mask_type == "only_mask_image":
            left_boundary_tokens = self.image_start
            right_boundary_tokens = self.image_end
            left_boundary = "left"
            right_boundary = "right"
        elif mask_type == "model_tool_call":
            left_boundary_tokens = self.tool_start
            right_boundary_tokens = self.tool_end
            left_boundary = "left"
            right_boundary = "right"
        else:
            raise ValueError(f"mask_type {mask_type} is not implemented, choose from 'only_mask_image_pad' or 'only_mask_image'")"""

        logger.info(f"interval in _get_sub_turn_mask : {interval}")

        if interval[1] - interval[0] == 0:
            return mask

        sub_user_intervals = get_boundaries_tokenized(
            input_ids[row, interval[0]:interval[1]].unsqueeze(0),
            torch.tensor(mask_type["left_boundary_tokens"], dtype=torch.long),
            torch.tensor(mask_type["right_boundary_tokens"], dtype=torch.long),
            left_boundary=mask_type["left_boundary"],
            right_boundary=mask_type["right_boundary"],
            optional_left_boundary_end=torch.tensor(mask_type["optional_left_boundary_end"], dtype=torch.long)
                                        if mask_type["optional_left_boundary_end"] is not None else None,
            optional_right_boundary_end=torch.tensor(mask_type["optional_right_boundary_end"], dtype=torch.long)
                                        if mask_type["optional_right_boundary_end"] is not None else None
        )

        sub_user_intervals = get_intervals(sub_user_intervals, kind=mask_type["kind"])
        for sub_user_interval in sub_user_intervals[0]:
            mask = self.do_mask(mask, row, interval[0] + sub_user_interval[0], interval[0] + sub_user_interval[1])
        return mask


    def do_mask(self, mask, row, col_begin, col_end):
        if self.verbose:
            logger.info(f"masking [{row}, {col_begin}:{col_end}]")
        mask[row, col_begin:col_end] = False
        return mask

    def get_mask(self, input_ids: torch.tensor, mask_type: dict, mask = None, device = None, rows_to_ignore = None):
        if rows_to_ignore is None:
            rows_to_ignore = []
        if device is None:
            device = input_ids.device
        if mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool, device = device)


        logger.info(f"left: {mask_type['left_boundary']}, right: {mask_type['right_boundary']}")

        full_boundaries = get_boundaries_tokenized(input_ids,
                                 torch.tensor(mask_type["left_boundary_tokens"], dtype=torch.long),
                                 torch.tensor(mask_type["right_boundary_tokens"], dtype=torch.long),
                                 left_boundary=mask_type["left_boundary"],
                                 right_boundary=mask_type["right_boundary"],
                                 optional_left_boundary_end=torch.tensor(mask_type["optional_left_boundary_end"], dtype=torch.long)
                                                   if mask_type["optional_left_boundary_end"] is not None else None,
                                 optional_right_boundary_end=torch.tensor(mask_type["optional_right_boundary_end"], dtype=torch.long)
                                                   if mask_type["optional_right_boundary_end"] is not None else None)

        full_intervals = get_intervals(full_boundaries, kind=mask_type["kind"])
        for i, intervals in enumerate(full_intervals):
            if i not in rows_to_ignore:
                logger.info(f"intervals: {intervals}")
                for interval in intervals[mask_type["start_interval"]:]:
                    if mask_type["sub"] is None:
                        mask = self.do_mask(mask, i, interval[0], interval[1])
                    else:
                        mask = self._get_sub_turn_mask(input_ids, i, mask, interval, mask_type["sub"])
        return mask

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



def get_boundaries_tokenized_old(content: torch.tensor, left_boundary_tokens:torch.tensor, right_boundary_tokens:torch.tensor,
                             left_boundary = "right", right_boundary = "left")-> list[list[list[int]]]:
    """
    while the left and right boundaries are wrt to the model generation, this function creates start and end of the non-model
    generations
    return model_generated_boundaries:list. outermost list is a text sequence, then we have a list of 2-lists (consisting of start non-model, end non-model)
    as the number of rounds

    example: input: ssslluuurraaalluuurraaa
    left_boundary_tokens: ll
    right_boundary_tokens: rr

    left_boundary: right
    right_boundary: left
    output: |ssslluuurr|aaa|lluuurr|aaa

    left_boundary: left
    right_boundary: right
    output: |ssslluuu|rraaall|uuu|rraaa

CHECK! Write proper tests!!
    """
    # Default boundaries must be provided
    if left_boundary_tokens is None or right_boundary_tokens is None:
        raise ValueError("Both left_boundary_tokens and right_boundary_tokens must be provided")

    left_len = len(left_boundary_tokens)
    right_len = len(right_boundary_tokens)

    if left_boundary == "right":
        right_output_offset = 0
    elif left_boundary == "left":
        right_output_offset = -left_len
    else:
        raise NotImplementedError(f"left_boundary={left_boundary} is not implemented, choose 'left' or 'right'")

    if right_boundary == "right":
        left_output_offset = +right_len
    elif right_boundary == "left":
        left_output_offset = 0
    else:
        raise NotImplementedError(f"left_boundary={left_boundary} is not implemented, choose 'left' or 'right'")

    model_generated_boundaries = []

    for token_seq in content:
        seq_len = len(token_seq)
        assert seq_len >= left_len + right_len, f"Sequence length is too short: {seq_len} < {left_len} + {right_len}, for seq: {content}"
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

        boundaries.append([0, right_idx + right_output_offset])

        i = j
        while i <= seq_len - left_len:
            if torch.equal(token_seq[i:i + right_len], right_boundary_tokens):
                left_idx = i
                j = left_idx
                found_closing_idx = False
                while j <= seq_len - left_len:
                    if torch.equal(token_seq[j:j + left_len], left_boundary_tokens):
                        right_idx = j + left_len
                        found_closing_idx = True
                        break
                    j += 1

                if not found_closing_idx:
                    logger.info(f"Could not find closing boundary, user reply/tool use seems malformed: {token_seq}")
                    right_idx = seq_len - right_output_offset

                i = right_idx



                boundaries.append([left_idx + left_output_offset, right_idx + right_output_offset])
            else:
                i += 1

        model_generated_boundaries.append(boundaries)

    return model_generated_boundaries

def get_boundaries_tokenized(content: torch.tensor, left_boundary_tokens:torch.tensor, right_boundary_tokens:torch.tensor,
                             left_boundary = "right", right_boundary = "left",
                             optional_left_boundary_end=None, optional_right_boundary_end=None)-> list[list[int]]:
    """
    while the left and right boundaries are wrt to the model generation, this function creates start and end of the non-model
    generations
    return model_generated_boundaries:list. outermost list is a text sequence, then we have a list of 2-lists (consisting of start non-model, end non-model)
    as the number of rounds

    example: input: ssslluuurraaalluuurraaa
    left_boundary_tokens: ll
    right_boundary_tokens: rr

    left_boundary: right
    right_boundary: left
    output: |ssslluuurr|aaa|lluuurr|aaa

    left_boundary: left
    right_boundary: right
    output: |ssslluuu|rraaall|uuu|rraaa

CHECK! Write proper tests!!
    """
    # Default boundaries must be provided
    if left_boundary_tokens is None or right_boundary_tokens is None:
        raise ValueError("Both left_boundary_tokens and right_boundary_tokens must be provided")


    left_len = len(left_boundary_tokens)
    right_len = len(right_boundary_tokens)

    if left_boundary == "right":
        left_output_offset = left_len
    elif left_boundary == "left":
        left_output_offset = 0
    else:
        raise NotImplementedError(f"left_boundary={left_boundary} is not implemented, choose 'left' or 'right'")

    if right_boundary == "right":
        right_output_offset = right_len
    elif right_boundary == "left":
        right_output_offset = 0
    else:
        raise NotImplementedError(f"left_boundary={left_boundary} is not implemented, choose 'left' or 'right'")

    model_generated_boundaries = []

    for token_seq in content:
        seq_len = len(token_seq)
        assert seq_len > 0, f"Sequence length is zero! seq = {content}"
        assert seq_len >= left_len + right_len, f"Sequence length is too short: {seq_len} < {left_len} + {right_len}, for seq: {content}"
        boundaries = [0]
        #boundaries = []
        i = 0

        while i <= seq_len - left_len:
            #logger.info(f"i: {i}")
            if torch.equal(token_seq[i:i + left_len], left_boundary_tokens):
                if (optional_left_boundary_end is not None
                        and i <= seq_len - left_len - 1
                        and torch.equal(token_seq[i+left_len: i+left_len+len(optional_left_boundary_end)], optional_left_boundary_end)):
                    if left_boundary == "left":
                        boundaries.append(i+left_output_offset)
                    elif left_boundary == "right":
                        boundaries.append(i+left_output_offset+len(optional_left_boundary_end))
                    j = i + left_len + len(optional_left_boundary_end)
                else:
                    boundaries.append(i+left_output_offset)
                    j = i + left_len

                found_right_boundary_end = False
                while j <= seq_len - right_len:
                    #logger.info(f"j: {j}")
                    if torch.equal(token_seq[j:j + right_len], right_boundary_tokens):
                        found_right_boundary_end = False
                        #logger.info(f"found match: {token_seq[j-5: j+5]}")
                        #logger.info(f"optional right boundary end: {optional_right_boundary_end}")

                        #if optional_right_boundary_end is not None:
                            #logger.info(f"{j} ? {seq_len - right_len - len(optional_right_boundary_end)}")
                            #logger.info(f"len: {len(optional_right_boundary_end)}")
                        #    if j <= seq_len - right_len - len(optional_right_boundary_end):
                                #logger.info(f"token seq: {token_seq[j + right_len: j + right_len + len(optional_right_boundary_end)]} vs {optional_right_boundary_end}")

                        #logger.info(f"{seq_len - right_len - len(optional_right_boundary_end)}")
                        #logger.info(f"[{j+right_len}:{j + right_len + len(optional_right_boundary_end)}]")
                        if (optional_right_boundary_end is not None
                                and j <= seq_len - right_len - len(optional_right_boundary_end)
                                and torch.equal(token_seq[j + right_len: j + right_len + len(optional_right_boundary_end)], optional_right_boundary_end)):
                            if right_boundary == "left":
                                #logger.info(f"left optional: appending: {j + right_output_offset}")
                                boundaries.append(j + right_output_offset)
                            elif right_boundary == "right":
                                #logger.info(f"right optional: appending: {j + right_output_offset + len(optional_right_boundary_end)}")
                                boundaries.append(j + right_output_offset + len(optional_right_boundary_end))
                            found_right_boundary_end = True
                            #j = i + left_len + len(optional_right_boundary_end)
                        else:
                            #logger.info(f"not optional: appending: {j + right_output_offset}")
                            boundaries.append(j+right_output_offset)
                        break
                    j += 1
                if found_right_boundary_end:
                    i = j + right_len + len(optional_right_boundary_end)
                else:
                    i = j + right_len
            else:
                i += 1
        boundaries.append(seq_len)

        #if boundaries[1] == 0:
        #    boundaries = boundaries[1:]
        #if boundaries[-2] == seq_len:
        #    boundaries = boundaries[:-1]

        model_generated_boundaries.append(boundaries)

    return model_generated_boundaries

def get_intervals(mmarkers: list[list[int]], kind:str):
    boundaries_list = []
    for markers in mmarkers:
        boundaries = []
        if kind == "inclusion":
            idx = 1
        elif kind == "exclusion":
            idx = 0
        else:
            raise NotImplementedError(f"kind {kind} is not implemented, choose from 'inclusion' or 'exclusion'")
        while idx+1 < len(markers):
            boundaries.append([markers[idx], markers[idx + 1]])
            idx = idx + 2

        boundaries_list.append(boundaries)
    return boundaries_list
