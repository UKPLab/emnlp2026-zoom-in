from transformers import AutoProcessor
import PIL.Image
import torch

import torch
import numpy as np
from open_r1.utils.buffer import Buffer
from open_r1.utils.logger import setup_project_logging
import torch.multiprocessing as mp


logger = setup_project_logging(None)

def worker_function(rank, img_path, generation, num_processes):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    buffer = Buffer(max_size=4, padding_id=0, use_global_buffer=True, cpu_buffer=True)
    buffer.display_state()

    #logger.info(f"in worker function {rank}: img_path: {img_path}, generation: {generation}")
    for idx in range(len(img_path)):
        image_paths = img_path[idx]
        generations = generation[idx]
        image = []
        for image_path in image_paths:
            for imag in image_path:
                image.append(PIL.Image.open(imag))
            #image.append(imag)

        logger.info(f"before processor: image: {len(image)}, generations: {len(generations)}")

        inputs = processor(
            text=generations.copy(),
            images=image.copy(),
            padding=True,
            return_tensors="pt",
            padding_side="left"
        )

        images_per_sample = torch.tensor([len(b) for b in image_paths], device="cpu")
        logger.info(f"images_per_sample: {images_per_sample}")
        aligned_inputs = {
            "prompt_ids": inputs["input_ids"],
            "prompt_mask": torch.ones_like(inputs["input_ids"]),
            "non_generation_mask": torch.ones_like(inputs["input_ids"]),
            "old_per_token_logps": torch.ones_like(inputs["input_ids"]),
            "ref_per_token_logps": torch.ones_like(inputs["input_ids"]),
            "advantages": torch.tensor([2, 1]),
            "images_per_sample": images_per_sample,
            "multimodal_inputs": {
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"]
            }
        }

        buffer.add(aligned_inputs)

        buffer.display_state(params=["prompt_ids"
            #"prompt_ids",
                                     # "prompt_mask",
                                     # "non_generation_mask",
                                     # "old_per_token_logps",
                                     # "ref_per_token_logps"
                                     ], shape_only = True)

    """for _ in range(100):
        new_batch = buffer.get(batch_size=3, deterministic=False, device="cpu")
        if new_batch['prompt_ids'].shape[1] < 1400:
            logger.info(f"rank {rank}: sampled short sequence!")
            break"""
        #logger.info(f"rank {rank}: {}")
    #print(f"new batch!")
    #print(f"new_batch.keys(): {new_batch.keys()}")
    #for k in new_batch.keys():
    #    if k == "multimodal_inputs":
    #        print(f"{k}: {new_batch[k]['pixel_values'].shape}")
    #        print(f"{k}: {new_batch[k]['image_grid_thw'].shape}")
    #    else:
    #        print(f"{k}: {new_batch[k].shape}")

    #buffer.display_state(params=["prompt_ids"])

if __name__ == "__main__":


    img_paths = [
        # process 1
        [  # 1151
            #batch 1
            [["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1000.png"],
            ["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_725.png"]],
            #batch 2
            [["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1000.png",
             "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_725.png"],
             ["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1151.png",
              "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1089.png"]
             ]
        ],
        # process 2
        [
            # batch 1
            [["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1151.png",
             "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1089.png"],
            ["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1157.png",
             "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1097.png"]],
            #batch 2
            [["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1151.png",
             "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1089.png"],
             ["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1157.png",
             "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/png/multi_col_1097.png"]]
        ]
    ]

    generations = [
        # process 1
        [
            #batch 1
            [
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020,",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year bla<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020,"
            ],
            # batch 2
            [
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailand's budget balance was at -4.7. After this point, there are no further expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's budget balance in relation to GDP is from the year 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected budget balances available to provide for the forecast period ending at 2026 before the final information.\n</think>\n<answer>\n2020\n</answer>",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailanere are no further expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected budget balances available to provide for the forecast period ending at 2026 before the final information.\n</think>\n<answer>\n2020\n</answer>",
               ]
        ],
        # process 2
        [
            # batch 1
            [
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailand's budget balance was at -4.7. After this point, there are no further expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's budget balance in relation to GDP is from the year 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected budget balances available to provide for the forecast period ending at 2026 before the final information.\n</think>\n<answer>\n2020\n</answer>",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailanere are no further expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected budget balances available to provide for the forecast period ending at 2026 before the final information.\n</think>\n<answer>\n2020\n</answer>",
                ],
            # batch 2
            [
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailand's budget balance was at -4.7. After this point, there are no further expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's budget balance in relation to GDP is from the year 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected budget balances available to provide for the forecast period ending at 2026 before the final information.\n</think>\n<answer>\n2020\n</answer>",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailanere are no further expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected budget balances available to provide for the forecast period ending at 2026 before the final information.\n</think>\n<answer>\n2020\n</answer>",
                # "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What year did Thailand's budget balance last in relation to GDP? First output the thinking process in <think> </think> t </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn 2020, Thailand's budget balance was at -4.7. expected figures specified for the given range. Therefore, it seems that the last available budget balance figure for Thailand in relation to the GDP is already provided for 2020.\n</think>\n<answer>\n2020\n</answer><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Are you sure? Look at the image again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nIn the chart, the last data point for Thailand's budget balance in relation to GDP is from the year 2020. The figure provided for 2020 is -4.7%, showing the budget balance that year. For subsequent years, such as 2025 and 2026, the values remain 0%.\n\nThese figures indicate a significant deficit in the Thailand's budget at that time and that there were no further expected before the final information.\n</think>\n<answer>\n2020\n</answer>",
            ]
        ]
    ]

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Create shared tensor
    #data = torch.randn(8, 5)
    #shared_data = data.share_memory_()  # Make tensor shareable between processes

    #print("Original data:")
    #print(shared_data)

    # Number of processes
    num_processes = 2

    # Create and start processes
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=worker_function, args=(rank, img_paths[rank], generations[rank], num_processes))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    #print("\nProcessed data:")
    #print(shared_data)


