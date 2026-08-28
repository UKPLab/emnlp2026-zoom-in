import os
import json

from datasets import Dataset
from functools import partial
from open_r1.utils.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Keys every record in a canonical JSONL dataset must provide.
CANONICAL_KEYS = ("problem", "image", "solution")
CANONICAL_FORMAT_HELP = (
    "Expected canonical JSONL, one object per line with keys: "
    "'problem' (str), 'image' (str or list[str], paths relative to the image "
    "folder) and 'solution' (str or list[str]). The M&C generator emits this "
    "format directly; third-party datasets are brought into it by download_data.py."
)


def make_hf_dataset(dataset_names, data_folders, image_folders, reward_method=None):
    """Load one or more canonical JSONL files into a single HF ``Dataset``.

    All dataset-specific parsing now lives in the data-preparation scripts
    (``download_data.py`` for third-party sets, ``generate_synthetic_grid_data.py``
    for M&C); every file read here is assumed to be in the canonical format
    described by :data:`CANONICAL_FORMAT_HELP`. Extra keys (e.g. ``bbox``) are
    passed through untouched.
    """
    if len(data_folders) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    if reward_method is None:
        accu_reward_methods = ["default"] * len(data_folders)
    else:
        accu_reward_methods = reward_method.split(":")
        assert len(accu_reward_methods) == len(data_folders), (
            f"Number of reward methods must match number of data files: "
            f"{len(accu_reward_methods)} != {len(data_folders)}"
        )

    logger.info(f"Loading data from {dataset_names}")
    logger.info(f"Data files: {data_folders}")
    logger.info(f"Image folders: {image_folders}")

    all_data = []
    for dataset_name, data_file, image_folder, accu_reward_method in zip(
        dataset_names, data_folders, image_folders, accu_reward_methods
    ):
        logger.info(f"Loading data from {dataset_name} : {data_file}")
        num_loaded = 0
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                missing = [k for k in CANONICAL_KEYS if k not in item]
                if missing:
                    raise KeyError(
                        f"Record in {data_file} is missing required key(s) {missing}. "
                        f"{CANONICAL_FORMAT_HELP} Got keys: {list(item)}"
                    )

                # image -> list of absolute paths under this dataset's image folder
                images = item.pop("image")
                if not isinstance(images, list):
                    images = [images]
                item["image_path"] = [os.path.join(image_folder, img) for img in images]

                # solution -> list (a single string answer is wrapped)
                solution = item["solution"]
                if not isinstance(solution, list):
                    solution = [solution]
                item["solution"] = solution

                item["accu_reward_method"] = accu_reward_method
                all_data.append(item)
                num_loaded += 1

        logger.info(f"Loaded {num_loaded} examples from {dataset_name}")

    dataset = Dataset.from_list(all_data)
    return dataset


def make_conversation_from_jsonl(example, question_prompt):
    if 'image_path' in example and example['image_path'] is not None:
        # Don't load image here, just store the path
        return {
            'image_path': [p for p in example['image_path']],  # Store path instead of loaded image
            'problem': example['problem'],
            'solution': [f"\\boxed{{{sol}}}" for sol in example['solution']],
            'bbox': example['bbox'] if 'bbox' in example else None,
            'accu_reward_method': example['accu_reward_method'],
            'prompt': [{
                'role': 'user',
                'content': [
                    *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                    {'type': 'text', 'text': question_prompt.replace("{Question}", example['problem'])}
                ]
            }]
        }
    else:
        return {
            'problem': example['problem'],
            'solution': [f"\\boxed{{{sol}}}" for sol in example['solution']],
            'accu_reward_method': example['accu_reward_method'],
            'prompt': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question_prompt.replace("{Question}", example['problem'])}
                ]
            }]
        }

def prepare_data(dataset_names, data_folders, image_folders, question_prompt, reward_method):
    hf_dataset = make_hf_dataset(dataset_names, data_folders, image_folders, reward_method)
    make_conv_with_prompt = partial(make_conversation_from_jsonl, question_prompt = question_prompt)
    # Map the conversations
    dataset = hf_dataset.map(make_conv_with_prompt, num_proc=8)
    logger.info(f"dataset: {dataset}")
    logger.info(f"dataset: {dataset[0]}")
    logger.info(f"dataset: {dataset[9]}")
    return dataset
