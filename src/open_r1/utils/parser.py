import torch
from .logger import get_logger
from .buffer import pad_and_cat

import torch.nn.functional as F
import numpy as np


logger = get_logger(__name__)



class Utterance:
    def __init__(self, im_start_pos, im_end_pos, role, utterance_start_pos, image_pos, tool_pos, last_pos, is_last=False):
        self.im_start_pos = im_start_pos
        self.im_end_pos = im_end_pos

        self.utterance_start_pos = utterance_start_pos
        # TODO: this is more of a theoretical value at the moment. In the worst case,
        #  the model generates directly the EOS token, then utterance_start_pos will be larger than last_pos.
        #  I don't think it poses an issue with the currently supported modes from 'get_mask_indices'

        self.image_pos = image_pos
        self.tool_pos = tool_pos
        self.role = role

        self.last_pos = last_pos

        self.image_ids = None
        self.is_last = is_last

    def __repr__(self):
        return (f"\n###\nim_start_pos: {self.im_start_pos},\n"
                f"im_end_pos: {self.im_end_pos},\n"
                f"last_pos: {self.last_pos},\n"
                f"role: {self.role},\n"
                f"utterance_start_pos: {self.utterance_start_pos},\n"
                f"image_pos: {self.image_pos},\n"
                f"tool_pos: {self.tool_pos},\n"
                f"image_ids: {self.image_ids}")


class ParsedTokenized:
    def __init__(self, input_ids, attention_mask, image_grid_thw, pixel_values, verbose:bool = False):
        self.im_end = 151645
        self.im_start = 151644

        self.model_start = 77091
        self.user_start = 872
        self.system_start = 8948

        self.image_start = 151652
        self.image_end = 151653

        self.tool_start = 151657
        self.tool_end = 151658

        self.between_turns = 198

        self.newline = 198

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.image_grid_thw = image_grid_thw
        self.pixel_values = pixel_values

        self.batch_size = self.input_ids.shape[0]
        self.num_cols = self.input_ids.shape[1]

        self.parsed = [[] for _ in range(self.batch_size)]

        self.total_images = 0
        self.image_features = None

        self.verbose = verbose

        self.parse()

        self.max_turns = max([len(conv) for conv in self.parsed])

    def parse(self):

        im_start_rows, im_start_cols = torch.where(self.input_ids == self.im_start)
        im_end_rows, im_end_cols = torch.where(self.input_ids == self.im_end)

        #logger.info(f"im_start_rows: {im_start_rows}")
        #logger.info(f"im_end_rows: {im_end_rows}")

        #old_idx = -1
        image_idx = 0
        end_idx = 0
        for im_start_row, im_start_col in zip(im_start_rows, im_start_cols):

            len_without_pad = torch.nonzero(self.attention_mask[im_start_row, :].squeeze(0), as_tuple=False)[-1].item()+1
            if end_idx < len(im_end_rows) and im_start_row == im_end_rows[end_idx]:
                im_end_col = im_end_cols[end_idx]
                utterance = self.utterance_from_parser(self.input_ids[im_start_row, im_start_col:im_end_col+1],
                                                                            im_start_col, im_end_col)
                end_idx += 1
            else:
                utterance = self.utterance_from_parser(self.input_ids[im_start_row, im_start_col:len_without_pad],
                                                       im_start_col, None)

            utterance.image_ids = list(range(image_idx, image_idx + len(utterance.image_pos)))
            image_idx += len(utterance.image_pos)
            #logger.info(f"IMAGE_IDX: {image_idx}")
            self.parsed[im_start_row].append(utterance)
            self.total_images = image_idx

        for conv in self.parsed:
            if len(conv) > 0:
                conv[-1].is_last = True


    def utterance_from_parser(self, tensor, im_start_pos, im_end_pos):

        assert tensor[0] == self.im_start, f"Tensor does not start with im_start, {tensor[0]} != {self.im_start}"

        if tensor[1] == self.model_start:
            role = "model"
        elif tensor[1] == self.user_start:
            role = "user"
        elif tensor[1] == self.system_start:
            role = "system"
        else:
            raise ValueError(f"Invalid role: {tensor[1]}")

        if tensor[2] == self.newline:
            utterance_start_pos = im_start_pos + 3
        else:
            utterance_start_pos = im_start_pos + 2

        image_pos = get_sub_entity(tensor, self.image_start, self.image_end, im_start_pos)
        tool_pos = get_sub_entity(tensor, self.tool_start, self.tool_end, im_start_pos)

        return Utterance(im_start_pos, im_end_pos, role, utterance_start_pos,  image_pos, tool_pos,
                         im_start_pos+len(tensor)-1)

    def get_mask_indices(self, mode:str, user_turn_range: list = None, model_turn_range: list = None):
        if user_turn_range is None:
            user_turn_range = [0, None]
        if model_turn_range is None:
            model_turn_range = [0, None]

        logger.info(f"user_turn_range: {user_turn_range}")
        logger.info(f"model_turn_range: {model_turn_range}")
        logger.info(f"max turns: {self.max_turns}")

        global_turns_to_consider = list(range(1+2*user_turn_range[0],
                                         1+2*user_turn_range[1] if user_turn_range[1] is not None else self.max_turns,
                                         2))
        logger.info(f"global_turns_to_consider before model turns: {global_turns_to_consider}")
        global_turns_to_consider += list(range(2+2*model_turn_range[0],
                                         2+2*model_turn_range[1] if model_turn_range[1] is not None else self.max_turns,
                                         2))

        logger.info(f"global_turns_to_consider: {global_turns_to_consider}")

        global_indices = []

        masked_image_indices = []

        for conv in self.parsed:
            if mode == "everything_except_model_generation":
                indices = [0]
                for utt in conv:
                    if utt.role == "model":
                        indices.append(utt.utterance_start_pos - 1)
                        if utt.im_end_pos is not None:
                            indices.append(utt.im_end_pos)
                        else:
                            indices.append(utt.last_pos+1)
                indices.append(self.num_cols - 1)
                masked_image_indices += list(range(self.total_images))
            else:
                indices = []

                for utt_id, utt in enumerate(conv):
                    if utt_id not in global_turns_to_consider:
                        continue
                    if mode == "full_tool_user_response":
                        if utt.role == "user":
                            indices.append(utt.im_start_pos - 2)
                            masked_image_indices += utt.image_ids
                        if utt.role == "model":
                            indices.append(utt.utterance_start_pos - 1)

                    if mode == "tool_user_response":
                        if utt.role == "user":
                            indices.append(utt.utterance_start_pos)
                            if utt.im_end_pos is not None:
                                indices.append(utt.im_end_pos - 1)
                            else:
                                indices.append(utt.last_pos)
                            masked_image_indices += utt.image_ids

                    if mode == "image":
                        if utt.role == "user":
                            indices.append(utt.image_pos[0][0])
                            indices.append(utt.image_pos[0][1])
                            masked_image_indices += utt.image_ids

                    if mode == "image_pad":
                        if utt.role == "user":
                            indices.append(utt.image_pos[0][0]+1)
                            indices.append(utt.image_pos[0][1]-1)
                            masked_image_indices += utt.image_ids

                    if mode == "model_tool_call":
                        if utt.role == "model" and not utt.is_last:
                            for tool_pos in utt.tool_pos:
                                indices.append(tool_pos[0])
                                indices.append(tool_pos[1] if tool_pos[1] is not None else utt.last_pos)

            global_indices.append(indices)
        return global_indices, masked_image_indices

    def get_shortened_tokenized(self, variants, padding_id, device:str = "cpu", padding_side = "left"):

        masked_image_indices = []
        #mask = self.attention_mask
        mask = torch.ones_like(self.attention_mask, dtype=torch.bool, device="cpu")
        #mask_begin = [self.num_cols for _ in range(self.batch_size)]
        for variant in variants:
            indices, mi_indices= self.get_mask_indices(variant["mode"],
                                                       variant["user_turn_range"],
                                                       variant["model_turn_range"])
            #for i, idx_list in enumerate(indices):
            #    if len(idx_list) > 0 and min(idx_list) < mask_begin[i]:
            #        mask_begin[i] = min(idx_list)
            #logger.info(f"indices: {indices}")
            #logger.info(f"mi_indices: {mi_indices}")

            mask = self.get_mask(mode=None, mask=mask, indices=indices)

            masked_image_indices += mi_indices
        #logger.info(f"masked_image_indices: {masked_image_indices}")

        mask_intervals_ignoring_padding = false_intervals_list(mask)

        mask = mask * self.attention_mask

        masked_image_indices = sorted(list(set(masked_image_indices)))
        #if consider_intervals is not None:
        #    masked_image_indices = [mi_index for i, mi_index in enumerate(masked_image_indices) if i in consider_intervals]
        #logger.info(f"masked_image_indices after dedup: {masked_image_indices}")
        #logger.info(f"total_images: {self.total_images}")

        #logger.info(f"mask before to shorten: {mask.shape}")
        shortened_input_ids, shortened_indices = self.mask_to_shortened_text(mask, padding_id, padding_side=padding_side)

        if self.image_grid_thw is not None and self.pixel_values is not None:
            shortened_image_grid_thw, shortened_pixel_values = self.shorten_image_features(masked_image_indices)
        else:
            shortened_image_grid_thw = None
            shortened_pixel_values = None

        shortened_attention_mask = torch.where(shortened_input_ids == padding_id, 0, 1)

        return {"input_ids": shortened_input_ids.to(device),
                "attention_mask": shortened_attention_mask.to(device),
                "image_grid_thw": shortened_image_grid_thw.to(device) if shortened_image_grid_thw is not None else None,
                "pixel_values": shortened_pixel_values.to(device) if shortened_pixel_values is not None else None,
                "indices": shortened_indices,
                "masked_image_indices":masked_image_indices,
                "mask_intervals_ignoring_padding": mask_intervals_ignoring_padding}

    def shorten_image_features(self, masked_image_indices:list):

        patches_per_image = (self.image_grid_thw[:, 1] * self.image_grid_thw[:, 2]).tolist()
        image_grid_thw_split = []
        pixel_values_split = []
        pixel_start_idx = 0
        for i in range(self.total_images):
            if i not in masked_image_indices:
                image_grid_thw_split.append(self.image_grid_thw[i:i+1])
                num_patches = sum(patches_per_image[i:i+1])
                pixel_end_idx = pixel_start_idx + num_patches
                pixel_values_split.append(self.pixel_values[pixel_start_idx:pixel_end_idx])
                pixel_start_idx = pixel_end_idx
        image_grid_thw_short = torch.cat(image_grid_thw_split, dim=0)
        pixel_values_short = torch.cat(pixel_values_split, dim=0)

        return image_grid_thw_short, pixel_values_short

    def mask_to_shortened_text(self, mask, padding_id, sentinel_value = -1, padding_side = "left"):
        rows = []
        original_indices = []
        indices = torch.arange(self.num_cols)
        for i in range(self.input_ids.shape[0]):
            masked_row = torch.masked_select(self.input_ids[i], mask[i].bool())
            if len(masked_row.shape) == 1:
                masked_row = masked_row.unsqueeze(0)
            original_index = torch.masked_select(indices, mask[i].bool())
            if len(original_index.shape) == 1:
                original_index = original_index.unsqueeze(0)

            rows.append(masked_row)
            original_indices.append(original_index)
        padded_values = pad_and_cat(rows, padding_id, padding_side=padding_side)
        padded_indices = pad_and_cat(original_indices, padding_value=sentinel_value, padding_side=padding_side)

        return padded_values, padded_indices


    def get_mask(self, mode, mask, indices = None):
        if indices is None:
            global_indices, _ = self.get_mask_indices(mode)
        else:
            global_indices = indices

        for idx in range(len(global_indices)):
            local_idx = 0
            while local_idx+1 < len(global_indices[idx]):
                if self.verbose:
                    logger.info(f"masking [{idx}, {global_indices[idx][local_idx]}:{global_indices[idx][local_idx+1]+1}]")
                mask[idx, global_indices[idx][local_idx]:global_indices[idx][local_idx+1]+1] = 0
                local_idx += 2

        return mask

    def get_model_response(self, number:int) -> list[list[tuple]]:
        global_number = number * 2 + 2
        model_response_positions = []
        for conv in self.parsed:
            conv_position = []
            for idx, utt in enumerate(conv):
                if idx == global_number and utt.role == "model":
                    if utt.im_end_pos is None:
                        excluded_right_boundary = utt.last_pos + 1
                    else:
                        excluded_right_boundary = utt.im_end_pos
                    conv_position.append((utt.utterance_start_pos, excluded_right_boundary))
            model_response_positions.append(conv_position)
        return model_response_positions




def rescale(short_input, shortened_indices, original_sized_tensor, original_sized_attention_mask,
            pad_token_id, sentinel_value = -1):
    rescaled = torch.zeros_like(original_sized_tensor, dtype=short_input.dtype, device=short_input.device)
    rescaled[original_sized_attention_mask == 0] = pad_token_id

    valid_mask = shortened_indices != sentinel_value  # Assuming -1 is sentinel value

    for i in range(short_input.shape[0]):
        valid_positions = valid_mask[i]
        rescaled[i, shortened_indices[i][valid_positions]] = short_input[i][valid_positions]


    return rescaled


def get_sub_entity(tensor, start, end, global_start) -> list[tuple[int, int]]:
    pos = []
    start_positions = torch.where(tensor == start)
    end_positions = torch.where(tensor == end)

    #logger.info(f"tensor dim in get_sub_entity: {tensor.shape}")
    #logger.info(f"global_start: {global_start}")
    #logger.info(f"start_positions: {start_positions}")

    for idx, start_position in enumerate(start_positions[0]):
        #logger.info(f"start_position: {start_position}")
        if idx < len(end_positions[0]):
            end_position = (global_start+end_positions[0][idx]).item()
        else:
            end_position = None #len(tensor) - 1
        pos.append(((global_start+start_position).item(), end_position))

    return pos

def all_false_intervals(x: torch.Tensor):
    """
    Given a 2D boolean tensor x [R, N], returns:
      - starts: int64 tensor [R, Kmax] with start indices (inclusive), padded with -1
      - ends:   int64 tensor [R, Kmax] with end indices (inclusive),   padded with -1
      - counts: int64 tensor [R] with the number of intervals per row
    Where Kmax = max(counts) across rows. If a row has no False, its count is 0 and
    its starts/ends row is all -1. Intervals are ordered left-to-right.
    """
    assert x.dtype == torch.bool and x.dim() == 2, "x must be a 2D bool tensor"
    R, N = x.shape

    m = ~x  # True where x is False

    # Pad with False on both sides so every run has a rise (+1) and a fall (-1)
    p = F.pad(m, (1, 1), value=False)  # [R, N+2]
    d = p[:, 1:].to(torch.int8) - p[:, :-1].to(torch.int8)  # [R, N+1]

    rises = (d == 1)   # start markers at column j
    falls = (d == -1)  # end markers at column j-1

    counts = rises.sum(dim=1)  # number of false-intervals per row
    Kmax = int(counts.max().item())

    # Fast path: no intervals anywhere
    if Kmax == 0:
        starts = x.new_full((R, 0), -1, dtype=torch.long)
        ends   = x.new_full((R, 0), -1, dtype=torch.long)
        return starts, ends, counts

    # Indices along the transition axis (0..N)
    idx = torch.arange(d.size(1), device=x.device, dtype=torch.long).expand(R, -1)

    # For each True in rises/falls, compute its "rank" (0-based order in the row)
    rise_rank = (rises.cumsum(dim=1) - 1)  # [R, N+1], valid only where rises is True
    fall_rank = (falls.cumsum(dim=1) - 1)

    # Collect coordinates of True positions
    r_rows, r_pos = torch.nonzero(rises, as_tuple=True)  # where interval starts occur
    f_rows, f_pos = torch.nonzero(falls, as_tuple=True)  # where interval ends occur

    # Their 0-based ranks within each row
    r_k = rise_rank[r_rows, r_pos]
    f_k = fall_rank[f_rows, f_pos]

    # Allocate padded outputs and scatter positions into their k-th slot
    starts = x.new_full((R, Kmax), -1, dtype=torch.long)
    ends   = x.new_full((R, Kmax), -1, dtype=torch.long)

    # Start index is the rise position j (already in original column space)
    starts[r_rows, r_k] = r_pos

    # End index maps from fall transition j to column j-1
    ends[f_rows, f_k] = f_pos - 1

    return starts, ends, counts

def false_intervals_list(x: torch.Tensor) -> list[list[list[int]]]:
    """
    Returns:
      [
        [[s0,e0], [s1,e1], ...],   # row 0
        [[s0,e0], ...],            # row 1
        ...
      ]
    Only real intervals are included (no -1 padding).
    """
    starts, ends, counts = all_false_intervals(x)
    R = x.size(0)
    out: list[list[list[int]]] = []
    for r in range(R):
        k = int(counts[r].item())
        if k == 0:
            out.append([])
        else:
            s = starts[r, :k].tolist()
            e = ends[r, :k].tolist()
            out.append([[si, ei] for si, ei in zip(s, e)])
    return out


def reduce_img_per_sample(img_per_sample: list[int], masked_image_ids: list[int]) -> list[int]:
    """
    Reduces the number of images per sample by the masked image indices. Example: img_per_sample = [2, 1, 2, 3, 1]
    masked_image_ids = [1, 4, 6, 7], output = [1,1,1,1,1]

    """
    img_per_sample = np.array(img_per_sample, dtype=int)
    masked_image_ids = np.array(masked_image_ids, dtype=int)

    cumulative_img_per_sample = np.cumsum(img_per_sample)

    subtract_img_per_sample = np.zeros(len(img_per_sample), dtype=int)

    for mi_idx in masked_image_ids:
        for cum_idx in range(len(cumulative_img_per_sample)):
            if cumulative_img_per_sample[cum_idx] > mi_idx and (
                    cum_idx == 0 or cumulative_img_per_sample[cum_idx - 1] <= mi_idx):
                subtract_img_per_sample[cum_idx] += 1
                break

    reduced_img_per_sample = img_per_sample - subtract_img_per_sample

    return reduced_img_per_sample.tolist()

def get_processing(name: str):
    # the order of T_Text and T_image can be reversed but this does not matter
    # Q; A1_R, A1_T ; T_text, vision_start, T_image, vision_end ; im_start, \n, A2
    if name is None:
        return None
    elif name == "tool_call_and_execution":
        # masks {A1_T ; T_text, T_image ; im_start, \n}
        processing = [{"mode": "model_tool_call",
          "user_turn_range": None,
          "model_turn_range": [0, 1]},  # only consider the first model response
         {"mode": "full_tool_user_response",
          "user_turn_range": [1, 2], # only consider the second user query as there will be the execution of the first tool call
          "model_turn_range": [1, 2]  # only consider the model generation directly after the requested user generation
          },
         ]
    elif name == "full_image":
        # masks {vision_start, T_image, vision_end}
        processing = [
            {"mode": "image",
             "user_turn_range": [1, 2],
             "model_turn_range": None}
        ]
    elif name == "image_pad_only":
        # masks {T_image}
        processing = [
            {"mode": "image_pad",
             "user_turn_range": [1, 2],
             "model_turn_range": None}
        ]
    else:
        raise ValueError(f"Unknown processing name: {name}")
    return processing



