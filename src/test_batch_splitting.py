from transformers import AutoProcessor
import PIL.Image
import torch

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
        splitted_batch = self.split_multimodal_batch(item)

        for batch_item in splitted_batch:

            if self.write_position >= self.max_size:
                self.write_position = 0
                # this assumes that we call n times add and then n times read before repeating
                self.read_position = 0
            # TODO: buffer is softly overwritten and never fully cleared all at once
            self.data[self.write_position] = batch_item
            self.write_position += 1

    def get(self, batch_size: int, deterministic=False):
        idxs = self._get_idxs(batch_size, deterministic)
        new_batch = self.combine_multimodal_batches([self.data[idx] for idx in idxs])
        return new_batch

    def _get_idxs(self, batch_size: int, deterministic=False) -> list[int]:
        if deterministic:
            idxs = np.arange(self.read_position, self.read_position + batch_size).tolist()
            self.read_position += batch_size
            return idxs

        valid_items = [item for item in self.data if item is not None]
        if not valid_items:
            raise ValueError("No valid items in buffer.")

        advantages = torch.cat([item["advantages"] for item in valid_items])
        normalized_advantages = torch.abs(advantages) ** self.alpha
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
        pixel_values = batch_outputs["multimodal_inputs"][
            'pixel_values']  # Shape: (total_patches, channels, patch_height, patch_width)

        # Calculate number of patches per image
        patches_per_image = (image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()

        # Split pixel_values based on patches per image
        pixel_values_split = []
        start_idx = 0
        for num_patches in patches_per_image:
            end_idx = start_idx + num_patches
            pixel_values_split.append(pixel_values[start_idx:end_idx])
            start_idx = end_idx

        # Split image_grid_thw (assuming one grid per image in batch)
        images_per_batch_item = len(patches_per_image) // batch_size
        image_grid_thw_split = []
        for i in range(batch_size):
            start_idx = i * images_per_batch_item
            end_idx = (i + 1) * images_per_batch_item
            image_grid_thw_split.append(image_grid_thw[start_idx:end_idx])

        # Create individual batch items
        individual_batches = []
        for i in range(batch_size):
            individual_batch = {
                'pixel_values': pixel_values_split[i],
                'image_grid_thw': image_grid_thw_split[i]
            }
            for k in self.batch_splittable_keys:
                individual_batch[k] = batch_outputs[k][i:i + 1]

            if self.cpu_buffer:
                for k in individual_batch.keys():
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
        # Handle standard batch tensors - concatenate along batch dimension
        for k in self.batch_splittable_keys:
            list_by_key = [batch[k] for batch in individual_batches]

            if k == "advantages":
                combined_batch[k] = torch.cat(list_by_key, dim=0)
            else:
                combined_batch[k] = left_pad_and_cat(list_by_key, self.padding_id)

        # Handle pixel_values - concatenate ALL patches from ALL batch items into one sequence
        # This recreates the original concatenated pixel_values tensor
        pixel_values_list = []
        for batch in individual_batches:
            if 'pixel_values' in batch and batch['pixel_values'] is not None:
                pixel_values_list.append(batch['pixel_values'])

        if pixel_values_list:
            combined_batch['pixel_values'] = torch.cat(pixel_values_list, dim=0)

        # Handle image_grid_thw - concatenate ALL grid info from ALL batch items
        # This recreates the original concatenated image_grid_thw tensor
        image_grid_thw_list = []
        for batch in individual_batches:
            if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
                image_grid_thw_list.append(batch['image_grid_thw'])

        if image_grid_thw_list:
            combined_batch['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)

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
            padding = torch.full(pad_shape, padding_value, dtype=tensor.dtype)
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


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    image_paths1 = [ # 1151
        "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1000.png",
        "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train/png/multi_col_725.png"]

    full_generations1 = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020,",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year bla<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020,"
    ]

    image_paths2 = [
        ["/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1151.png",
         "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1089.png"],
        ["/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1157.png",
         "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1097.png"]
    ]

    full_generations2 = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailand's budget balance was at -4.7. After this point, there are no further expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's budget balance in relation to GDP is from the year 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected budget balances available to provide for the forecast period ending at 2026 before the final information.\n</think>\n<answer>\n2020\n</answer>",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailanere are no further expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected budget balances available to provide for the forecast period ending at 2026 before the final information.\n</think>\n<answer>\n2020\n</answer>",
        # "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> t </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailand's budget balance was at -4.7. expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's budget balance in relation to GDP is from the year 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected before the final information.\n</think>\n<answer>\n2020\n</answer>",
    ]

    buffer = Buffer(max_size=4, padding_id=0)
    buffer.display_state()

    for image_paths, generations in [[image_paths1, full_generations1], [image_paths2, full_generations2]]:
        #image = [[PIL.Image.open(image_path) for image_path in image_paths] for _ in range(1)]
        if not isinstance(image_paths[0], list):
            im = PIL.Image.open(image_paths[0])
            width, height = im.size
            im = im.resize((int(width*0.8), int(height*0.8)))
            image = [im, PIL.Image.open(image_paths[1])]
        else:
            image = []
            for image_path in image_paths:
                imag = []
                for img in image_path:
                    imag.append(PIL.Image.open(img))
                image.append(imag)



        inputs = processor(
            text=generations.copy(),
            images=image.copy(),
            padding=True,
            return_tensors="pt",
            padding_side="left"
        )

        aligned_inputs = {
            "prompt_ids": inputs["input_ids"],
            "prompt_mask": torch.ones_like(inputs["input_ids"]),
            "non_generation_mask": torch.ones_like(inputs["input_ids"]),
            "old_per_token_logps": torch.ones_like(inputs["input_ids"]),
            "ref_per_token_logps": torch.ones_like(inputs["input_ids"]),
            "advantages": torch.tensor([1,2]),
            "multimodal_inputs": {
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"]
            }
        }

        buffer.add(aligned_inputs)

    buffer.display_state()

    new_batch = buffer.get(batch_size=3, deterministic=False)
    print(f"new batch!")
    for k in new_batch.keys():
        if k == "multimodal_inputs":
            print(f"{k}: {new_batch[k]['pixel_values'].shape}")
            print(f"{k}: {new_batch[k]['image_grid_thw'].shape}")
        print(f"{k}: {new_batch[k].shape}")

    buffer.display_state()