import re
from .math import compute_score
from math_verify import parse, verify
from Levenshtein import ratio
import numpy as np
import os
import torch
from datetime import datetime

from .utils import calculate_iou


def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-Z])([A-Z])(?=[\.\,\?\!\:\;]|$)', text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos - 20):min(len(text), pos + 20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos + 1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]


def mcq_reward(content, sol, **kwargs):
    # For multiple choice, extract and compare choices
    has_choices = extract_choice(sol)
    correct_choice = has_choices.upper() if has_choices else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()
    student_choice = extract_choice(student_answer)
    if student_choice:
        reward = 1.0 if student_choice == correct_choice else 0.0
    else:
        reward = 0.0

    return reward


def yes_no_reward(content, sol, **kwargs):
    content = content.lower()
    sol = sol.lower()

    # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()

    ground_yes_no = re.search(r'(yes|no)', ground_truth)
    ground_yes_no = ground_yes_no.group(1) if ground_yes_no else ''
    student_yes_no = re.search(r'(yes|no)', student_answer)
    student_yes_no = student_yes_no.group(1) if student_yes_no else ''

    reward = 1.0 if ground_yes_no == student_yes_no else 0.0

    return reward


def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except:
        return None


def math_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return compute_score(content, sol)


def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]

    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')

    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()


def default_accuracy_reward(content, sol, regex_type, enforce_unique, enforce_last):
    reward = 0.0

    if regex_type == "answer_tags":
        regex = r'<answer>(.*?)</answer>'
        search_string = "<answer>"
    elif regex_type == "box":
        regex = r'\\boxed\{(.*?)}'
        search_string = "\\boxed{"

    if enforce_last:
        regex += r"\s{,3}$"

    if enforce_unique:
        if content.count(search_string) != 1:
            return reward

    # Extract answer from solution if it has think/answer tags
    sol_match = re.search(regex, sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(regex, content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()

    # Try symbolic verification first for numeric answers
    try:
        answer = parse(student_answer)
        if float(verify(answer, parse(ground_truth))) > 0:
            reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails

    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try:
            # Check if ground truth contains numbers
            has_numbers = bool(re.search(r'\d', ground_truth))
            # Check if it's a multiple choice question
            has_choices = extract_choice(ground_truth)

            if has_numbers:
                # For numeric answers, use exact matching
                reward = numeric_reward(student_answer, ground_truth)
                if reward is None:
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            elif has_choices:
                # For multiple choice, extract and compare choices
                correct_choice = has_choices.upper()
                student_choice = extract_choice(student_answer)
                if student_choice:
                    reward = 1.0 if student_choice == correct_choice else 0.0
            else:
                # For text answers, use fuzzy matching
                reward = ratio(clean_text(student_answer), clean_text(ground_truth))
        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward

def string_matching_reward(content, sol, regex_type, enforce_unique, enforce_last, match_type, preprocess=None):
    reward = 0.0

    if regex_type == "answer_tags":
        regex = r'<answer>(.*?)</answer>'
        search_string = "<answer>"
    elif regex_type == "box":
        regex = r'\\boxed\{(.*?)}'
        search_string = "\\boxed{"

    if enforce_last:
        regex += r"\s{,3}$"

    if enforce_unique:
        if content.count(search_string) != 1:
            return reward

    # Extract answer from solution if it has think/answer tags
    sol_match = re.search(regex, sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(regex, content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()

    if preprocess == "lowercase":
        ground_truth = ground_truth.lower()
        student_answer = student_answer.lower()

    if match_type == "exact_match":
        reward = 1.0 if clean_text(student_answer) == clean_text(ground_truth) else 0.0
    elif match_type == "levenshtein_similarity":
        reward = ratio(clean_text(student_answer), clean_text(ground_truth))

    return reward




def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    cutoff = kwargs.get("cutoff", 0.0)
    if cutoff is None:
        cutoff = 0.0

    for content, sols, accu_reward_method in zip(contents, solution, kwargs.get("accu_reward_method")):
        reward_candidates = []
        for sol in sols:
            # logger.info(f"in accuracy reward: content: {content}, method: {accu_reward_method}")
            # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
            if accu_reward_method == "mcq":
                reward = mcq_reward(content, sol)
            elif accu_reward_method == 'yes_no':
                reward = yes_no_reward(content, sol)
            elif accu_reward_method == 'math':
                reward = math_reward(content, sol)
            elif accu_reward_method == 'string_matching':
                reward = string_matching_reward(content, sol, regex_type="box", enforce_unique=True, enforce_last=False, # original PR always generates a . after the box on infovqa
                                                match_type=kwargs.get("match_type"),
                                                preprocess=kwargs.get("preprocess"))
            else:
                reward = default_accuracy_reward(content, sol, regex_type="box", enforce_unique=True, enforce_last=True)
            reward_candidates.append(reward if reward >= cutoff else 0.0)
        rewards.append(np.max(np.array(reward_candidates)))

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 1.0:  # this condition can be changed for debug
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"accu_reward_method: {accu_reward_method}\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # logger.info(f"in format_reward: completions: {completions}, kwargs: {kwargs}")
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    # logger.info(f"in format_reward: content: {completion_contents}, matches: {matches}")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]


def format_reward_only_answer(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # logger.info(f"in format_reward: completions: {completions}, kwargs: {kwargs}")
    # TODO: is it necessary to check that there is only a single <answer></answer> tag and not multiple for intermediate turn results?
    pattern = r".*<answer>(.*?)</answer>\s{,2}$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    # logger.info(f"in format_reward: content: {completion_contents}, matches: {matches}")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]


def curiosity_reward(tool_uses, group_size, pixel_reasoning_threshold, **kwargs):
    binary_tool_use = (tool_uses > 0).float() # (280)
    pixel_reasoning_rate = binary_tool_use.view(-1, group_size).mean(dim=1) # (35, 8) -> (35)
    pixel_reasoning_rate = pixel_reasoning_rate.repeat_interleave(group_size, dim=0) # (280)
    curiosity = torch.clamp(pixel_reasoning_threshold - pixel_reasoning_rate, min=0) * binary_tool_use
    return curiosity


def pr_penalty_reward(tool_uses, tool_use_penalty_threshold, **kwargs):
    return torch.clamp(tool_use_penalty_threshold - tool_uses, max=0)

def constant_exploration(tool_uses, **kwargs):
    return (tool_uses > 0).float()

def iou_reward(bbox_estimate:list[list[tuple[int, int, int, int]]], bbox:list[tuple[int, int, int, int]],
               aggregate_over_conv: str, **kwargs):
    rewards = []
    for conv_idx in range(len(bbox_estimate)):
        if len(bbox_estimate[conv_idx]) == 0:
            rewards.append(0.0)
            continue
        conv_ious = []
        for turn_idx in range(len(bbox_estimate[conv_idx])):
            conv_ious.append(calculate_iou(bbox_estimate[conv_idx][turn_idx], bbox[conv_idx]))

        if aggregate_over_conv == "first":
            rewards.append(conv_ious[0])
            continue
        elif aggregate_over_conv == "last":
            rewards.append(conv_ious[-1])
            continue
        elif aggregate_over_conv == "mean":
            rewards.append(np.mean(np.array(conv_ious)))
            continue
        else:
            raise ValueError(f"Invalid aggregate_over_conv: {aggregate_over_conv}")

    return rewards

def mutual_information_reward(absolute_diff:torch.Tensor, contrasted_area: torch.Tensor, contrast_diff_list: list[torch.Tensor],
                              delta:float, gamma: float,
                              alpha:float, length_factor_scaling:int, tau:float, discretize:bool, q:float,
                              ignored_prefix_len: int, tanh: bool, length_factor: float,
                              select_k:int, select_k_type: str, **kwargs):

    if ignored_prefix_len is None:
        ignored_prefix_len = 0

    rewards = []
    if contrast_diff_list is not None:
        for contrast_diff in contrast_diff_list:
            if contrast_diff is None:
                rewards.append(0.0)
            else:
                if select_k is not None and select_k_type is not None:
                    k = min(select_k, len(contrast_diff))
                    if select_k_type == "first":
                        contrast_diff_area = contrast_diff[:k]
                    elif select_k_type == "max":
                        if select_k >= len(contrast_diff):
                            contrast_diff_area = contrast_diff
                        else:
                            contrast_diff_area = torch.topk(contrast_diff, k, largest=True, sorted=True)
                    else:
                        raise ValueError(f"Invalid select_k_type: {select_k_type}")
                else:
                    contrast_diff_area = contrast_diff


                rewards.append(calculate_mi_reward(contrast_diff_area, delta = delta, gamma = gamma, alpha = alpha, tau=tau,
                                                   discretize=discretize, q=q, tanh=tanh,
                                                   length_factor_scaling=length_factor_scaling,
                                                   length_factor=length_factor))
    else:
        if contrasted_area is None:
            return None
        for i in range(len(contrasted_area)):
            reward = 0
            if len(contrasted_area[i]) > 0:
                start_contrast = contrasted_area[i][0][0] - 1 + ignored_prefix_len
                end_contrast = contrasted_area[i][0][1] - 1

                if start_contrast <= end_contrast:
                    contrast_diff = absolute_diff[i, start_contrast:end_contrast]
                    reward = calculate_mi_reward(contrast_diff, delta = delta, gamma = gamma, alpha = alpha, tau=tau,
                                               discretize=discretize, q=q, tanh=tanh,
                                               length_factor_scaling=length_factor_scaling,
                                                 length_factor=length_factor)

            rewards.append(reward)
    return rewards

def calculate_mi_reward(contrast_diff: torch.Tensor, delta:float, gamma: float,
                        alpha:float, length_factor_scaling:int, tau:float,
                        discretize:bool, q:float, tanh: bool, length_factor: float):

    if tau is None:
        threshold = 0
    else:
        threshold = tau

    if alpha is None:
        alpha = 1

    if length_factor_scaling is None:
        length_factor_scaling = 1

    if length_factor is None:
        length_factor = 1
    elif length_factor == "len":
        length_factor = (length_factor_scaling / len(contrast_diff)) ** alpha
    else:
        length_factor = float(length_factor) ** alpha


    reward = 0.0
    if not contrast_diff.numel() == 0:

        if delta is not None:
            reward = torch.mean(torch.where(contrast_diff > delta, 1.0, 0.0) * length_factor)
        elif q is not None:
            distr = torch.clamp(contrast_diff, min=-gamma, max=gamma) - threshold
            original_dtype = distr.dtype
            distr = distr.to(dtype=torch.float)
            reward = torch.quantile(distr, q)
            reward = reward.to(dtype=original_dtype)
        else:
            if gamma is not None:
                contrast_diff = torch.clamp(contrast_diff, min=-gamma, max=gamma)
            reward = length_factor * torch.sum(contrast_diff - threshold)

        if discretize is True:
            if reward < 0:
                reward = -1
            else:
                reward = +1

        if tanh is True:
            reward = torch.tanh(reward)

    return reward
