from ..buffer import Buffer
import torch
from ..logger import get_logger, setup_project_logging

# Get logger for this module
#logger = get_logger(__name__)
logger = setup_project_logging(log_file=None)

"""
to call, go to the parent folder of utils (open_r1) and run from there:
python -m utils.tests.test_buffer
"""



buffer = Buffer(max_size=4, padding_id = -1, cpu_buffer=True)

data = [
    {
        "prompt_ids": torch.tensor([
                [5,5,3,3,8,8],
                [-1,5,5,3,3,8]
            ], device="cuda:0"),
        "prompt_mask": torch.tensor([
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1]
            ], device="cuda:0"),
        "non_generation_mask":torch.tensor([
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1]
            ], device="cuda:0"),
        "old_per_token_logps": torch.tensor([
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [-1, 2.2, 2.3, 2.4, 2.5]
            ], device="cuda:0"),
        "ref_per_token_logps": torch.tensor([
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [-1, 2.2, 2.3, 2.4, 2.5]
            ], device="cuda:0"),
        "advantages": torch.tensor([5.1, 5.2], device="cuda:0"),
        "images_per_sample": [1,1],
        "multimodal_inputs": {
        "image_grid_thw":torch.tensor([
            [1, 40, 58],
            [1, 40, 58]
            ], device="cuda:0"),
        "pixel_values":torch.tensor([
            [7.1, 7.2, 7.3],
            [8.1, 8.2, 8.3]
            ], device="cuda:0"),
        }

    },
    {"prompt_ids": torch.tensor([
                [5,5,3,3,8,8, 8],
                [-1,5,5,3,3,8, 8]
            ], device="cuda:0"),
        "prompt_mask": torch.tensor([
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1]
            ], device="cuda:0"),
        "non_generation_mask":torch.tensor([
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1]
            ], device="cuda:0"),
        "old_per_token_logps": torch.tensor([
            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            [-1, 2.2, 2.3, 2.4, 2.5, 2.6]
            ], device="cuda:0"),
        "ref_per_token_logps": torch.tensor([
            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            [-1, 2.2, 2.3, 2.4, 2.5, 2.6]
            ], device="cuda:0"),
        "advantages": torch.tensor([5.1, 5.2], device="cuda:0"),
        "images_per_sample": [1,1],
        "multimodal_inputs": {
        "image_grid_thw":torch.tensor([
            [1, 40, 58],
            [1, 40, 58]
            ], device="cuda:0"),
        "pixel_values":torch.tensor([
            [7.1, 7.2, 7.3],
            [8.1, 8.2, 8.3]
            ], device="cuda:0"),
        }
    }
]


for i in range(2):
    buffer.add(data[i])
    sample = buffer.get(batch_size=2, device="cuda:0")
    logger.info(f"Sample: {sample}")