from transformers import AutoProcessor
import PIL.Image
import torch


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

if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    image_paths = ["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1000.png",
                   #"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1000.png",
                   "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1000.png"]

    image = [[PIL.Image.open(image_path) for image_path in image_paths] for _ in range(2)]

    full_generations_with_2_imgs = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailand's budget balance was at -4.7. After this point, there are no further expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's budget balance in relation to GDP is from the year 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected budget balances available to provide for the forecast period ending at 2026 before the final information.\n</think>\n<answer>\n2020\n</answer>",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailanere are no further expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected budget balances available to provide for the forecast period ending at 2026 before the final information.\n</think>\n<answer>\n2020\n</answer>",
        #"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> t </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailand's budget balance was at -4.7. expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's budget balance in relation to GDP is from the year 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected before the final information.\n</think>\n<answer>\n2020\n</answer>",
    ]

    full_generations = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020,",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020,"
    ]

    inputs = processor(
        text=full_generations_with_2_imgs,
        images=image,
        padding=True,
        return_tensors="pt",
        padding_side="left"
    )

    left_boundary = processor(text="<|im_start|>assistant\n", return_tensors="pt",)
    right_boundary = processor(text="<|im_end|>", return_tensors="pt",)

    model_generated_boundaries = get_boundaries_tokenized(content=inputs["input_ids"], left_boundary_tokens=left_boundary["input_ids"][0],
                                                          right_boundary_tokens = right_boundary["input_ids"][0])
    #print(inputs["input_ids"])
    #print(inputs["input_ids"].shape)
    #print(left_boundary["input_ids"][0])
    #print(right_boundary["input_ids"][0])
    print(model_generated_boundaries)

    for idx, seq_bound in enumerate(model_generated_boundaries):
        for boundary in seq_bound:
            print(f"{idx}: {inputs["input_ids"][idx, boundary[1]-5:boundary[1]+5]}")
            print(f'{(inputs["input_ids"] == 151643).sum(dim=1)}')
            print(inputs["input_ids"])

