import torch
import torch.distributed as dist
import numpy as np
from .logger import get_logger
from functools import partial
from accelerate.utils import is_peft_model, set_seed, gather_object, broadcast_object_list

# Get logger for this module
logger = get_logger(__name__)


# item is a dict with keys
# prompt_ids: tensor: (num_gen, num_tokens)
# prompt_mask: tensor: (num_gen, num_tokens)
# non_generation_mask: tensor: (num_gen, num_tokens)
# old_per_token_logps: tensor: (num_gen, num_tokens -1)
# ref_per_token_logps: tensor: (num_gen, num_tokens -1)
# advantages: tensor: (num_gen)
# multimodal_inputs: dict
#                     pixel_values: (, 1176)

# TODO only for testing
def _gpu_gather_object(object):
    output_objects = [None for _ in range(dist.get_world_size())]
    torch.distributed.all_gather_object(output_objects, object)
    # all_gather_object returns a list of lists, so we need to flatten it
    return [x for y in output_objects for x in y]

class Sample:
    non_multimodal_keys = ["prompt_ids", "prompt_mask", "non_generation_mask", "old_per_token_logps",
                           "ref_per_token_logps", "advantages", "sampling_weights"]

    multimodal_keys = ["num_images", "image_grid_thw", "pixel_values"]

    output_sample_structure = {"prompt_ids": "",
                               "prompt_mask": "",
                               "non_generation_mask": "",
                               "old_per_token_logps": "",
                               "ref_per_token_logps": "",
                               "advantages": "",
                               "multimodal_inputs": {
                                   "num_images": "",
                                   "image_grid_thw": "",
                                   "pixel_values": ""
                               }}

    input_sample_structure = {"prompt_ids": "",
                              "prompt_mask": "",
                              "non_generation_mask": "",
                              "old_per_token_logps": "",
                              "ref_per_token_logps": "",
                              "advantages": "",
                              "sampling_weights": "",
                              "images_per_sample": "",
                              "multimodal_inputs": {
                                  "image_grid_thw": "",
                                  "pixel_values": ""
                              }}

    def __init__(self, prompt_ids, prompt_mask, non_generation_mask, old_per_token_logps,
                 ref_per_token_logps, advantages, image_grid_thw, pixel_values, images_per_sample=None, num_images=None, sampling_weights=None):

        self.prompt_ids: torch.tensor = prompt_ids # (num_gen, num_tokens)
        self.prompt_mask: torch.tensor = prompt_mask # (num_gen, num_tokens)
        self.non_generation_mask: torch.tensor = non_generation_mask # (num_gen, num_tokens)
        self.old_per_token_logps: torch.tensor = old_per_token_logps # (num_gen, num_tokens -1)
        self.ref_per_token_logps: torch.tensor = ref_per_token_logps # (num_gen, num_tokens -1)
        self.advantages: torch.tensor = advantages # (num_gen)
        if sampling_weights is None:
            self.sampling_weights: torch.tensor = advantages
        else:
            self.sampling_weights: torch.tensor = sampling_weights #(num_gen)

        if images_per_sample is None:
            if num_images is None:
                raise ValueError("Either images_per_sample or num_images must be provided.")
            else:
                self.num_images: torch.tensor = num_images
        else:
            self.num_images: torch.tensor = images_per_sample # (1, num_gen)

        # non batch splittable
        self.image_grid_thw: torch.tensor = image_grid_thw
        self.pixel_values: torch.tensor = pixel_values # (, 1176)

        # batch splittable

    @staticmethod
    def from_external(item):
        return Sample(prompt_ids=item['prompt_ids'],
                      prompt_mask=item['prompt_mask'],
                      non_generation_mask=item['non_generation_mask'],
                      old_per_token_logps=item['old_per_token_logps'],
                      ref_per_token_logps=item['ref_per_token_logps'],
                      advantages=item['advantages'],
                      sampling_weights=item['sampling_weights'],
                      images_per_sample=torch.tensor(item["images_per_sample"], dtype=torch.int,
                                           device=item['prompt_ids'].device).unsqueeze(0),
                      image_grid_thw=item['multimodal_inputs']['image_grid_thw'],
                      pixel_values=item['multimodal_inputs']['pixel_values'])

    def to_external(self, operations):

        postprocess = partial(process, operations=operations)

        return {"prompt_ids": postprocess(self.prompt_ids),
                "prompt_mask": postprocess(self.prompt_mask),
                "non_generation_mask": postprocess(self.non_generation_mask),
                "old_per_token_logps": postprocess(self.old_per_token_logps),
                "ref_per_token_logps": postprocess(self.ref_per_token_logps),
                "advantages": postprocess(self.advantages),
                "multimodal_inputs": {
                    "num_images": postprocess(self.num_images),
                    "image_grid_thw": postprocess(self.image_grid_thw),
                    "pixel_values": postprocess(self.pixel_values)
                    }
                }

    def move_to_cpu(self):
        to_cpu = partial(process, operations=["detach", "move_to_device:cpu"])
        for k in Sample.non_multimodal_keys:
            #logger.info(f"key {k}: {getattr(self, k)}")
            setattr(self, k, to_cpu(getattr(self, k)))
        for k in Sample.multimodal_keys:
            setattr(self, k, to_cpu(getattr(self, k)))


    def split(self):

        # Get the grid information to determine split points
        image_grid_thw = self.image_grid_thw  # Shape: (total_images, 3)
        pixel_values = self.pixel_values # Shape: (total_patches, channels, patch_height, patch_width)

        # Calculate number of patches per image
        patches_per_image = (image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()

        # Split image_grid_thw (assuming one grid per image in batch)
        image_grid_thw_split = []
        start_idx = 0
        pixel_values_split = []
        pixel_start_idx = 0
        for i in range(self.get_batch_size()):
            end_idx = start_idx + int(self.num_images[0, i].item())
            image_grid_thw_split.append(image_grid_thw[start_idx:end_idx])

            num_patches = sum(patches_per_image[start_idx:end_idx])
            pixel_end_idx = pixel_start_idx + num_patches
            pixel_values_split.append(pixel_values[pixel_start_idx:pixel_end_idx])
            start_idx = end_idx
            pixel_start_idx = pixel_end_idx

        # Create individual batch items
        individual_batches = []
        for i in range(self.get_batch_size()):
            individual_batch = {
                'pixel_values': pixel_values_split[i],
                'image_grid_thw': image_grid_thw_split[i],
                'images_per_sample': self.num_images[:, i]
            }
            for k in Sample.non_multimodal_keys:
                individual_batch[k] = getattr(self,k)[i:i+1]

            individual_batches.append(Sample(**individual_batch))
        return individual_batches

    def get_batch_size(self):
        if self.prompt_ids is not None:
            return self.prompt_ids.shape[0]
        else:
            return 0

    def strip(self, padding_id, padding_side="left"):
        """Remove all padding from the sample"""
        self.prompt_ids = remove_left_padding_seq(self.prompt_ids, padding_id, padding_side)
        if padding_side == "left":
            for k in Sample.non_multimodal_keys:
                if k in ["prompt_ids", "advantages", "sampling_weights"]:
                    continue

                if k in ["old_per_token_logps", "ref_per_token_logps"]:
                    setattr(self, k, getattr(self, k)[:, -(self.prompt_ids.shape[1] - 1):])
                else:
                    setattr(self, k, getattr(self, k)[:, -self.prompt_ids.shape[1]:])
        elif padding_side == "right":
            for k in Sample.non_multimodal_keys:
                if k in ["prompt_ids", "advantages", "sampling_weights"]:
                    continue

                if k in ["old_per_token_logps", "ref_per_token_logps"]:
                    setattr(self, k, getattr(self, k)[:, :self.prompt_ids.shape[1]-1])
                else:
                    setattr(self, k, getattr(self, k)[:, :self.prompt_ids.shape[1]])

        else:
            raise ValueError(f"padding_side must be left or right, got {padding_side}")

    def display(self, shape_only=True, params:list = None):
        for k in Sample.multimodal_keys + Sample.non_multimodal_keys:
            if params is not None and k in params:
                if shape_only:
                    logger.info(f"key {k}: {getattr(self, k).shape}")
                else:
                    logger.info(f"key {k}: {getattr(self, k)}")
        logger.info("---")


class Buffer:
    def __init__(self, max_size, padding_id, cpu_buffer=False, sample_with_replacement=True, flush_after_episode=False,
                 use_global_buffer:bool=False):
        self.max_size = max_size
        self.data: list[Sample] = [None] * self.max_size
        # position always points to the index that should be (over-)written next
        self.write_position = 0
        self.read_position = 0
        # alpha = 0 is uniform sampling
        self.alpha = 1.0


        self.cpu_buffer = cpu_buffer
        self.sample_with_replacement = sample_with_replacement
        self.padding_id = padding_id
        # if true we delete all entries from the buffer all at once after an episode.
        # Otherwise we overwrite softly and can still sample very old samples during that time.
        self.flush_after_episode = flush_after_episode

        self.use_global_buffer = use_global_buffer
        if self.use_global_buffer and not self.cpu_buffer:
            raise ValueError("Global buffer only works with cpu buffer.")

        self.global_buffer = [None] * self.max_size * dist.get_world_size() if self.use_global_buffer else None

    def sync_buffers(self, begin:int, end:int):
        dist.barrier()

        # TODO: allow arbitrary batch sizes to be written
        if begin >= end:
            if self.flush_after_episode:
                logger.info("flushing global buffer!")
                self.global_buffer = [None] * self.max_size * dist.get_world_size()
            if begin == self.max_size:
                begin = 0
            else:
                raise ValueError("begin must be smaller than end. This points towards the fact that the local buffer size is not a multiple of the batch sizes used to fill it. ")

        logger.info(f"syncing buffers: begin: '{begin}', end: '{end}'")
        #global_buffer = _gpu_gather_object(self.data[begin:end])
        global_buffer = gather_object(self.data[begin:end])

        logger.info(f"global buffer: {len(global_buffer)}")
        logger.info(f"global buffer first entry: {global_buffer[0]}")
        logger.info(f"local buffer: {len([item for item in self.data if item is not None])}/{len(self.data)}")
        for idx in range(dist.get_world_size()):
            logger.info(f"syncing global buffer between '{idx * self.max_size + begin}' and '{idx * self.max_size + end}' with list slice {idx*(end-begin)}:{(idx+1)*(end-begin)}")
            self.global_buffer[idx * self.max_size + begin:
                           idx * self.max_size + end] = global_buffer[idx*(end-begin):(idx+1)*(end-begin)]



    def add(self, item, padding_side="left"):
        old_write_position = self.write_position
        splitted_batch = self.split_multimodal_batch(item)

        #logger.info(f"add to buffer:")
        for batch_item in splitted_batch:
            batch_item.strip(self.padding_id, padding_side)
            if self.cpu_buffer:
                batch_item.move_to_cpu()
            if self.write_position >= self.max_size:
                self.write_position = 0
                if self.flush_after_episode:
                    logger.info(f"flushing buffer!")
                    self.data = [None] * self.max_size
                # this assumes that we call n times add and then n times read before repeating
                self.read_position = 0
            self.data[self.write_position] = batch_item

            self.write_position += 1
        if self.use_global_buffer:
            self.sync_buffers(old_write_position, self.write_position)

    def get(self, batch_size: int, deterministic=False, device=None, padding_side="left"):
        sampling_source = self.data if not self.use_global_buffer else self.global_buffer
        idxs = self._get_idxs(batch_size, sampling_source, deterministic)
        new_batch = self.combine_multimodal_batches([sampling_source[idx] for idx in idxs], padding_side=padding_side)

        batch_dict = new_batch.to_external(["clone", "detach", "contiguous", f"move_to_device:{device}", "detach"])
        return batch_dict

    def _get_idxs(self, batch_size: int, sampling_source, deterministic=False) -> list[int]:
        if deterministic:
            #logger.info(f"read pos: {self.read_position}")
            if self.read_position >= self.max_size:
                self.read_position -= self.max_size
            idxs = np.arange(self.read_position, self.read_position + batch_size).tolist()
            self.read_position += batch_size
            logger.info(f"sampled_indices: {idxs}")
            return idxs


        valid_items = [item for item in sampling_source if item is not None]

        valid_items_back_map = [idx for idx, item in enumerate(sampling_source) if item is not None]

        if not valid_items:
            raise ValueError("No valid items in buffer.")

        sampling_weights = torch.cat([item.sampling_weights for item in valid_items])

        normalized_sampling_weights = sampling_weights ** self.alpha
        if torch.sum(normalized_sampling_weights) == 0:
            logger.info(f"CAUTION, sampling_weights sum is 0, using uniform sampling")
            normalized_sampling_weights = torch.ones(sampling_weights.shape[0]) / sampling_weights.shape[0]
        else:
            normalized_sampling_weights = normalized_sampling_weights / torch.sum(normalized_sampling_weights)

        logger.info(f"normalized_sampling_weights: {normalized_sampling_weights}")
        # Use torch.multinomial for sampling
        sampled_indices = torch.multinomial(normalized_sampling_weights, batch_size,
                                            replacement=self.sample_with_replacement).tolist()

        back_mapped_indices = [valid_items_back_map[idx] for idx in sampled_indices]

        logger.info(f"sampled_indices: {back_mapped_indices}")
        return back_mapped_indices

    def split_multimodal_batch(self, batch_outputs) -> list[Sample]:
        """
        Split concatenated pixel_values and image_grid_thw back into individual sequences
        """
        big_sample = Sample.from_external(batch_outputs)
        sample_list = big_sample.split()

        return sample_list

    def display_state(self, meta=None, shape_only=True, params:list = None):
        if meta is not None:
            logger.info(f"code position meta: {meta}")
        if self.use_global_buffer:
            logger.info(f"Global buffer:")
            for idx, sample in enumerate(self.global_buffer):
                if sample is not None:
                    logger.info(f"buffer item {idx}:")
                    sample.display(shape_only, params)
        logger.info(f"Local buffer:")
        logger.info(f"read pos: {self.read_position}")
        logger.info(f"write pos: {self.write_position}")
        logger.info(f"buffer space taken: {len([d for d in self.data if d is not None])}")
        for idx, sample in enumerate(self.data):
            if sample is not None:
                logger.info(f"buffer item {idx}:")
                sample.display(shape_only, params)

    def buffer_space_taken(self) -> int:
        return len([d for d in self.data if d is not None])

    def combine_multimodal_batches(self, individual_batches, padding_side="left"):
        combined_batch = {}

        prompt_ids_list = [batch.prompt_ids for batch in individual_batches]
        combined_batch['prompt_ids'] = pad_and_cat(prompt_ids_list, self.padding_id, padding_side=padding_side)

        # Handle standard batch tensors - concatenate along batch dimension
        for k in Sample.non_multimodal_keys:
            list_by_key = [getattr(batch, k) for batch in individual_batches]

            if k == "prompt_ids":
                continue
            elif k == ["advantages", "sampling_weights"]:
                combined_batch[k] = torch.cat(list_by_key, dim=0)
            else:
                if k in ["prompt_mask", "non_generation_mask"]:
                    combined_batch[k] = pad_and_cat(list_by_key, 0, padding_side=padding_side)
                    assert combined_batch["prompt_ids"].shape == combined_batch[k].shape
                elif k in ["old_per_token_logps", "ref_per_token_logps"]:
                    combined_batch[k] = pad_and_cat(list_by_key, padding_value=-1.0875e+01, padding_side=padding_side)
                    #assert combined_batch["prompt_ids"].shape == combined_batch[k].shape, f"combined_batch['prompt_ids'].shape = {combined_batch["prompt_ids"].shape} != {combined_batch[k].shape} = combined_batch[k].shape"
                    assert combined_batch["prompt_ids"].shape[0] == combined_batch[k].shape[0]
                    assert combined_batch["prompt_ids"].shape[1] == combined_batch[k].shape[1]+1

        for image_key in ["pixel_values", "image_grid_thw", "num_images"]:

            # Handle pixel_values - concatenate ALL patches from ALL batch items into one sequence
            # This recreates the original concatenated pixel_values tensor
            values_list = []
            for batch in individual_batches:
                values_list.append(getattr(batch, image_key))

            if values_list:
                combined_batch[image_key] = torch.cat(values_list, dim=0)#.clone().contiguous()

        return Sample(**combined_batch)


# Handle standard batch tensors with left padding
def pad_and_cat(tensors, padding_value, padding_side="left"):
    """Left pad tensors to same length and concatenate along batch dimension"""
    if not tensors:
        return None

    # Find maximum length
    max_len = max(tensor.shape[-1] for tensor in tensors)

    # Left pad each tensor
    padded_tensors = []

    for idx, tensor in enumerate(tensors):
        current_len = tensor.shape[-1]
        if current_len < max_len:
            pad_len = max_len - current_len
            # Create padding shape: (batch_size, ..., pad_len)
            pad_shape = list(tensor.shape)
            pad_shape[-1] = pad_len
            padding = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
            # Left pad: concatenate padding + original tensor
            if padding_side == "left":
                padded_tensor = torch.cat([padding, tensor], dim=-1)
            elif padding_side == "right":
                padded_tensor = torch.cat([tensor, padding], dim=-1)
            else:
                raise ValueError(f"padding_side must be left or right, got {padding_side}")
        else:
            padded_tensor = tensor
        padded_tensors.append(padded_tensor)

    return torch.cat(padded_tensors, dim=0)

def remove_left_padding_seq(seq:torch.tensor, padding_value=0, padding_side="left"):
    """Remove left padding from a tensor based on padding_value with zero-th dimension =1 (i.e. batch size = 1)"""
    if seq.numel() == 0:
        return seq

    #logger.info(f"seq shape: {seq.shape}")

    seq = seq.squeeze(0)

    # Find first occurrence of non-padding value
    non_pad_mask = (seq != padding_value)
    if non_pad_mask.any():
        if padding_side == "left":
            # Find the first non-padding position
            first_non_pad = torch.nonzero(non_pad_mask, as_tuple=False)[0].item()
            unpadded_seq = seq[first_non_pad:].unsqueeze(0)
        elif padding_side == "right":
            last_non_pad = torch.nonzero(non_pad_mask, as_tuple=False)[-1].item()
            unpadded_seq = seq[:last_non_pad+1].unsqueeze(0)
        else:
            raise ValueError(f"padding_side must be left or right, got {padding_side}")

    else:
        raise ValueError("tensor only consists of padding tokens!")
        # All padding - keep at least one token to avoid empty sequences

    return unpadded_seq


def process(tensor, operations=None):
    if operations is not None:
        for op in operations:
            if op == "detach":
                tensor = tensor.detach()
            if op == "clone":
                tensor = tensor.clone()
            if op == "contiguous":
                tensor = tensor.contiguous()
            if op.startswith("move_to_device:"):
                op = op.removeprefix("move_to_device:")
                tensor = tensor.to(op, non_blocking=True)
    return tensor