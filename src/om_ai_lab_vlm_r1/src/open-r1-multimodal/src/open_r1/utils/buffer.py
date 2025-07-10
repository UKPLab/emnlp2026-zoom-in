import torch
import numpy as np


# item is a dict with keys
# prompt_ids: tensor: (num_gen, num_tokens)
# prompt_mask: tensor: (num_gen, num_tokens)
# non_generation_mask: tensor: (num_gen, num_tokens)
# old_per_token_logps: tensor: (num_gen, num_tokens -1)
# ref_per_token_logps: tensor: (num_gen, num_tokens -1)
# advantages: tensor: (num_gen)
# multimodal_inputs: dict
#                     pixel_values: (, 1176)

class Sample:
    def __init__(self, item):

        self.item:dict = item
        # batch splittable
        self.prompt_ids:torch.tensor = item['prompt_ids']
        self.prompt_mask:torch.tensor = item['prompt_mask']
        self.non_generation_mask:torch.tensor = item['non_generation_mask']
        self.old_per_token_logps:torch.tensor = item['old_per_token_logps']
        self.ref_per_token_logps:torch.tensor = item['ref_per_token_logps']
        self.advantages:torch.tensor = item['advantages']
        self.num_images:list = item['images_per_sample']
        # non batch splittable
        self.image_grid_thw:torch.tensor = item['multimodal_inputs']['image_grid_thw']
        self.pixel_values:torch.tensor = item['multimodal_inputs']['pixel_values']


class Buffer:
    def __init__(self, max_size, padding_id, cpu_buffer=False, sample_with_replacement=True):
        self.max_size = max_size
        self.data = [None] * self.max_size
        # position always points to the index that should be (over-)written next
        self.write_position = 0
        self.read_position = 0
        # alpha = 0 is uniform sampling
        self.alpha = 1.0
        self.batch_splittable_keys = ["prompt_ids", "prompt_mask", "non_generation_mask", "old_per_token_logps",
                                      "ref_per_token_logps", "advantages"]
        self.cpu_buffer = cpu_buffer
        self.sample_with_replacement = sample_with_replacement
        self.padding_id = padding_id

    def add(self, item):
        #print(f"in buffer, before split: ")
        #print(f"prompt_ids: {item['prompt_ids'].shape}")
        #print(f"image grid thw: {item['multimodal_inputs']['image_grid_thw'].shape}")
        #print(f"pixel values: {item['multimodal_inputs']['pixel_values'].shape}")
        #print(f"advantages: {item['advantages']}")
        splitted_batch = self.split_multimodal_batch(item)

        #print(f"add to buffer:")
        for batch_item in splitted_batch:
            #print(f"prompt_ids: {batch_item['prompt_ids'].shape}")
            #print(f"image grid thw: {batch_item['image_grid_thw'].shape}")
            #print(f"pixel values: {batch_item['pixel_values'].shape}")
            if self.write_position >= self.max_size:
                self.write_position = 0
                # this assumes that we call n times add and then n times read before repeating
                self.read_position = 0
            # TODO: buffer is softly overwritten and never fully cleared all at once
            self.data[self.write_position] = batch_item
            self.write_position += 1

    def get(self, batch_size: int, deterministic=False):
        idxs = self._get_idxs(batch_size, deterministic)
        #print(f"sampled indices: {idxs}")
        new_batch = self.combine_multimodal_batches([self.data[idx] for idx in idxs])
        #print(f"got new batch: ")
        #print(f"prompt_ids: {new_batch['prompt_ids'].shape}")
        #print(f"image grid thw: {new_batch["multimodal_inputs"]['image_grid_thw'].shape}")
        #print(f"pixel values: {new_batch["multimodal_inputs"]['pixel_values'].shape}")
        return new_batch

    def _get_idxs(self, batch_size: int, deterministic=False) -> list[int]:
        if deterministic:
            #print(f"read pos: {self.read_position}")
            if self.read_position >= self.max_size:
                self.read_position -= self.max_size
            idxs = np.arange(self.read_position, self.read_position + batch_size).tolist()
            self.read_position += batch_size
            return idxs

        valid_items = [item for item in self.data if item is not None]
        if not valid_items:
            raise ValueError("No valid items in buffer.")

        advantages = torch.cat([item["advantages"] for item in valid_items])

        normalized_advantages = torch.abs(advantages) ** self.alpha
        if torch.sum(normalized_advantages) == 0:
            print(f"CAUTION, advantages sum is 0, using uniform sampling")
            normalized_advantages = torch.ones(advantages.shape[0]) / advantages.shape[0]
        else:
            normalized_advantages = normalized_advantages / torch.sum(normalized_advantages)

        print(f"normalized_advantages: {normalized_advantages}")
        # Use torch.multinomial for sampling
        sampled_indices = torch.multinomial(normalized_advantages, batch_size,
                                            replacement=self.sample_with_replacement).tolist()

        print(f"sampled_indices: {sampled_indices}")
        return sampled_indices

    def split_multimodal_batch(self, batch_outputs):
        """
        Split concatenated pixel_values and image_grid_thw back into individual sequences
        """

        batch_size = batch_outputs[self.batch_splittable_keys[0]].shape[0]

        # Get the grid information to determine split points
        image_grid_thw = batch_outputs["multimodal_inputs"]['image_grid_thw']  # Shape: (total_images, 3)
        #print(f"in split: image grid thw: {image_grid_thw}")
        pixel_values = batch_outputs["multimodal_inputs"][
            'pixel_values']  # Shape: (total_patches, channels, patch_height, patch_width)

        # Calculate number of patches per image
        patches_per_image = (image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()

        #print(f"patches per image: {patches_per_image}")

        # Split image_grid_thw (assuming one grid per image in batch)
        #images_per_batch_item = len(patches_per_image) // batch_size
        image_grid_thw_split = []
        start_idx = 0
        pixel_values_split = []
        pixel_start_idx = 0
        for i in range(batch_size):
            end_idx = start_idx + batch_outputs["images_per_sample"][i]
            #start_idx = i * images_per_batch_item
            #end_idx = (i + 1) * images_per_batch_item
            image_grid_thw_split.append(image_grid_thw[start_idx:end_idx])

            num_patches = sum(patches_per_image[start_idx:end_idx])
            pixel_end_idx = pixel_start_idx + num_patches
            pixel_values_split.append(pixel_values[pixel_start_idx:pixel_end_idx])
            start_idx = end_idx
            pixel_start_idx = pixel_end_idx

        # Create individual batch items
        individual_batches = []
        for i in range(batch_size):
            individual_batch = {
                'pixel_values': pixel_values_split[i],
                'image_grid_thw': image_grid_thw_split[i],
                'num_images': batch_outputs["images_per_sample"][i]
            }
            for k in self.batch_splittable_keys:
                individual_batch[k] = batch_outputs[k][i:i + 1]

            if self.cpu_buffer:
                for k in individual_batch.keys():
                    if not k == "num_images":
                        individual_batch[k] = individual_batch[k].detach().cpu()

            individual_batches.append(individual_batch)

        return individual_batches

    def display_state(self):
        print(f"read pos: {self.read_position}")
        print(f"write pos: {self.write_position}")
        print(f"buffer space taken: {len([d for d in self.data if d is not None])}")
        for idx, d in enumerate(self.data):
            if d is not None:
                print(f"{idx}: item keys: {d.keys()}")
                for k in d.keys():
                    if k == "multimodal_inputs":
                        print(f"item {k}: {d[k]['pixel_values'].shape}")
                        print(f"item {k}: {d[k]['image_grid_thw'].shape}")
                    else:
                        print(f"item {k}: {d[k].shape}")
                print("---")


    def combine_multimodal_batches(self, individual_batches):
        combined_batch = {}

        prompt_ids_list = [batch['prompt_ids'] for batch in individual_batches]
        combined_batch['prompt_ids'] = left_pad_and_cat(prompt_ids_list, self.padding_id)


        # Handle standard batch tensors - concatenate along batch dimension
        for k in self.batch_splittable_keys:
            list_by_key = [batch[k] for batch in individual_batches]

            if k == "prompt_ids":
                continue
            elif k == "advantages":
                combined_batch[k] = torch.cat(list_by_key, dim=0)
            else:
                combined_batch[k] = left_pad_and_cat(list_by_key, self.padding_id)
                if k in ["old_per_token_logps", "ref_per_token_logps"]:
                    combined_batch[k] = combined_batch[k][:, -(combined_batch["prompt_ids"].shape[1] - 1):]
                else:
                    combined_batch[k] = combined_batch[k][:, -combined_batch["prompt_ids"].shape[1]:]


        combined_batch['multimodal_inputs'] = {}
        # Handle pixel_values - concatenate ALL patches from ALL batch items into one sequence
        # This recreates the original concatenated pixel_values tensor
        pixel_values_list = []
        for batch in individual_batches:
            if 'pixel_values' in batch and batch['pixel_values'] is not None:
                pixel_values_list.append(batch['pixel_values'])

        if pixel_values_list:
            combined_batch['multimodal_inputs']['pixel_values'] = torch.cat(pixel_values_list, dim=0)

        # Handle image_grid_thw - concatenate ALL grid info from ALL batch items
        # This recreates the original concatenated image_grid_thw tensor
        image_grid_thw_list = []
        for batch in individual_batches:
            if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
                image_grid_thw_list.append(batch['image_grid_thw'])

        if image_grid_thw_list:
            combined_batch['multimodal_inputs']['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)

        return combined_batch


# Handle standard batch tensors with left padding
def left_pad_and_cat(tensors, padding_value):
    """Left pad tensors to same length and concatenate along batch dimension"""
    if not tensors:
        return None

    # First remove existing left padding from each tensor
    unpadded_tensors = []
    for tensor in tensors:
        unpadded_list = remove_left_padding(tensor, padding_value)
        unpadded_tensors.extend(unpadded_list)

    tensors = unpadded_tensors

    # Find maximum length
    max_len = max(tensor.shape[-1] for tensor in tensors)

    # Left pad each tensor
    padded_tensors = []
    for tensor in tensors:
        current_len = tensor.shape[-1]
        if current_len < max_len:
            pad_len = max_len - current_len
            # Create padding shape: (batch_size, ..., pad_len)
            pad_shape = list(tensor.shape)
            pad_shape[-1] = pad_len
            padding = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
            # Left pad: concatenate padding + original tensor
            padded_tensor = torch.cat([padding, tensor], dim=-1)
        else:
            padded_tensor = tensor
        padded_tensors.append(padded_tensor)
    return torch.cat(padded_tensors, dim=0)


def remove_left_padding(tensor, padding_value=0):
    """Remove left padding from a tensor based on padding_value"""
    if tensor.numel() == 0:
        return tensor

    # Find first non-padding token for each sequence in the batch
    unpadded_tensors = []
    for seq in tensor:
        # Find first occurrence of non-padding value
        non_pad_mask = (seq != padding_value)
        if non_pad_mask.any():
            # Find the first non-padding position
            first_non_pad = torch.nonzero(non_pad_mask, as_tuple=False)[0].item()
            unpadded_seq = seq[first_non_pad:].unsqueeze(0)
        else:
            # All padding - keep at least one token to avoid empty sequences
            unpadded_seq = seq[-1:] if seq.numel() > 0 else seq

        unpadded_tensors.append(unpadded_seq)

    return unpadded_tensors