import os
import json
import sys

from datasets import Dataset
from functools import partial
from open_r1.utils.logger import get_logger
import ast

# Get logger for this module
logger = get_logger(__name__)

def make_hf_dataset(dataset_names, data_folders, image_folders, reward_method=None):
    if len(data_folders) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    if reward_method is None:
        accu_reward_methods = ["default"] * len(data_folders)
    else:
        accu_reward_methods = reward_method.split(":")
        assert len(accu_reward_methods) == len(
            data_folders), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_folders)}"

    if len(data_folders) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    logger.info(f"Loading data from {dataset_names}")
    logger.info(f"Data folders: {data_folders}")
    logger.info(f"Image folders: {image_folders}")

    all_data = []
    for dataset_name, data_file, image_folder, accu_reward_method in zip(dataset_names, data_folders, image_folders, accu_reward_methods):
        logger.info(f"Loading data from {dataset_name} : {data_file}")
        if dataset_name == "chartqa":
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if 'image' in item:
                        if isinstance(item['image'], str):
                            # Store image path instead of loading the image
                            item['image_path'] = [os.path.join(image_folder, item['image'])]
                            del item['image']  # remove the image column so that it can be loaded later
                        elif isinstance(item['image'], list):
                            # if the image is a list, then it is a list of images (for multi-image input)
                            item['image_path'] = [os.path.join(image_folder, image) for image in item['image']]
                            del item['image']  # remove the image column so that it can be loaded later
                        else:
                            raise ValueError(f"Unsupported image type: {type(item['image'])}")
                    # Remove immediate image loading
                    item['problem'] = item['conversations'][0]['value'].replace('<image>', '')

                    # Handle solution that could be a float or string
                    solution_value = item['conversations'][1]['value']
                    if isinstance(solution_value, str):
                        item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                    else:
                        # If it's a float or other non-string type, keep it as is
                        item['solution'] = str(solution_value)

                    del item['conversations']
                    item['accu_reward_method'] = item.get('accu_reward_method',
                                                          accu_reward_method)  # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
                    all_data.append(item)
        elif dataset_name == "pixel_reasoner_vstar": # expected columns: question, image, answer
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    item['problem'] = item.pop('question')

                    item["image_path"] = [os.path.join(image_folder, img) for img in ast.literal_eval(item["image"])]
                    answer = item.pop('answer')
                    # answer is \boxed{x} where x = A,B,C,D
                    answer = answer.removeprefix("\\boxed{").removesuffix("}")

                    item["solution"] = answer

                    del item["image"]
                    item["accu_reward_method"] = accu_reward_method

                    all_data.append(item)
        elif dataset_name == "pixel_reasoner_infovqa":
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    item['problem'] = item.pop('question')

                    item["image_path"] = [os.path.join(image_folder, img) for img in ast.literal_eval(item["image"])]
                    assert len(item["image_path"]) == 1, f"Only one image is expected, but got {item['image_path']}"
                    answer = item.pop('answer')
                    answers = ast.literal_eval(answer)

                    answers = [answer[7:-1] for answer in answers]

                    item["solution"] = answers
                    del item["image"]
                    item["accu_reward_method"] = accu_reward_method

                    all_data.append(item)

        elif dataset_name in ["hr_bench_4k", "hr_bench_8k"]:
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    problem = item.pop('question')


                    item['problem'] = (f"{problem}\n(A) {item.pop('A')}"
                                                f"\n(B) {item.pop('B')}"
                                                f"\n(C) {item.pop('C')}"
                                                f"\n(D) {item.pop('D')}"
                                       f"\nAnswer with the option's letter from the given choices directly.")


                    item["image_path"] = [os.path.join(image_folder, item["image"])]
                    assert len(item["image_path"]) == 1, f"Only one image is expected, but got {item['image_path']}"
                    answers = [item.pop('answer')]

                    print(answers)

                    item["solution"] = answers
                    del item["image"]
                    item["accu_reward_method"] = accu_reward_method

                    all_data.append(item)
        elif dataset_name.startswith("mme"):
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    problem = item.pop('question')

                    item['problem'] = (f"{problem}\n"
                                       f"{"\n".join(item.pop('multi-choice options'))}"
                                       f"\nAnswer with the option's letter from the given choices directly.")

                    item["image_path"] = [os.path.join(image_folder, item["image"])]
                    assert len(item["image_path"]) == 1, f"Only one image is expected, but got {item['image_path']}"
                    answers = [item.pop('answer')]
                    print(answers)

                    item["solution"] = answers
                    del item["image"]
                    item["accu_reward_method"] = accu_reward_method

                    all_data.append(item)

        elif dataset_name.startswith("muffin_chihuahua"):
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)

                    problem = item.pop('question')
                    item['problem'] = (f"In the image you see a grid, whose cells are numbered from left to right and top to bottom. "
                                       f"In each cell, the cell's index is printed in the upper left corner. {problem}")

                    item["image_path"] = [os.path.join(image_folder, item["image"])]
                    assert len(item["image_path"]) == 1, f"Only one image is expected, but got {item['image_path']}"
                    answers = [item.pop('answer')]
                    print(answers)

                    item["solution"] = answers
                    del item["image"]
                    item["accu_reward_method"] = accu_reward_method

                    all_data.append(item)

        elif dataset_name == "visual_probe_train":
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    item["problem"] = item.pop("question")
                    item["solution"] = [item.pop("answer")]

                    assert len(item["image"]) == 1, f"Only one image is expected, but got {item['image']}"

                    item["image_path"] = [os.path.join(image_folder, item["image"][0])]

                    del item["image"]

                    item["accu_reward_method"] = accu_reward_method

                    all_data.append(item)

        elif dataset_name == "deepeyes_train_4k":
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    problem = item.pop("question")

                    item["problem"] = f"{problem}\nIf there are choices given, answer with the option's letter from the given choices directly."

                    item["solution"] = [item.pop("answer")]

                    assert len(item["image"]) == 1, f"Only one image is expected, but got {item['image']}"

                    item["image_path"] = [os.path.join(image_folder, item["image"][0])]

                    del item["image"]

                    item["accu_reward_method"] = accu_reward_method

                    all_data.append(item)


        elif dataset_name == "pixel_reasoner":
            non_singular_answers = {}
            image_path_dist = {}

            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if "qid" in item.keys():
                        del item["qid"]
                    if "is_video" in item.keys():
                        del item["is_video"]
                    if "__index_level_0__" in item.keys():
                        del item["__index_level_0__"]
                    item['problem'] = item.pop('question')

                    item["image_path"] = [os.path.join(image_folder, img) for img in ast.literal_eval(item["image"])]
                    if len(item["image_path"]) != 1:
                        continue
                        logger.info(f"Caution, this sample has {len(item['image_path'])} input images!")
                        if len(item["image_path"]) in image_path_dist.keys():
                            image_path_dist[len(item["image_path"])] += 1
                        else:
                            image_path_dist[len(item["image_path"])] = 1

                    #logger.info(f"item answer: {item['answer']}")
                    #logger.info(f"{isinstance(item['answer'], list)}")
                    #logger.info(f"{isinstance(item['answer'], str)}")
                    answer = item.pop('answer')
                    answer = ast.literal_eval(answer)
                    assert len(answer) == 1, f"Only one answer is expected, but got {answer}"
                    answer = answer[0]
                    answer = answer[7:-1]
                    if answer[0] == "[":
                        try:
                            answer = ast.literal_eval(answer)
                        except (ValueError, SyntaxError):
                            if not answer.isalnum():
                                logger.info(f"Answer is not ALNUM, but got {answer}")
                            #assert answer.isalnum(), f"Answer is neither list nor alnum, but got {answer}"
                            answer = [answer]
                    else:
                        #logger.info(f"Answer is not list, but got {answer}")
                        answer = [answer]


                    # TODO: we could allow for a list of solutions, but then we have to change the accuracy reward
                    # this affects approx 100/7000 examples

                    try:
                        answer_len = len(answer)
                    except Exception:
                        logger.info(f"Answer has no len: {answer}")

                    if len(answer) != 1:
                        if len(answer) in non_singular_answers.keys():
                            non_singular_answers[len(answer)] += 1
                        else:
                            non_singular_answers[len(answer)] = 1
                        #logger.info(f"NON-SINGULAR ANSWER: {answer}")


                    # pick first answer if multiple are provided
                    #answer = str(answer[0])
                    #logger.info(f"answer: {answer}")

                    # TODO


                    item["solution"] = answer
                    #logger.info(f"item img path: {item['image']}")
                    #logger.info(f"{isinstance(ast.literal_eval(item['image']), list)}")
                    #logger.info(f"{isinstance(ast.literal_eval(item['image']), str)}")

                    del item["image"]
                    item["accu_reward_method"] = accu_reward_method
                    #logger.info(f"item keys after process: {item.keys()}")
                    all_data.append(item)
                logger.info(f"Non singular answer dist: {non_singular_answers}")
                logger.info(f"number image dist: {image_path_dist}")

    dataset = Dataset.from_list(all_data)
    return dataset

def make_conversation_from_jsonl(example, question_prompt):
    if 'image_path' in example and example['image_path'] is not None:
        # Don't load image here, just store the path
        return {
            'image_path': [p for p in example['image_path']],  # Store path instead of loaded image
            'problem': example['problem'],
            #'solution': f"<answer> {example['solution']} </answer>",
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
            #'solution': f"<answer> {example['solution']} </answer>",
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
