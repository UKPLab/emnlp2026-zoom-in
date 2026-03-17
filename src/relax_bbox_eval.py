#!/usr/bin/env python3
"""
Relaxed evaluation script for log files with completions and solutions.
Parses the log for completion/solution pairs and compares them using
relaxed answer extraction (not just \\boxed{}).
"""

import re
import json
import copy
from pathlib import Path


def extract_answer_relaxed(text: str) -> str | None:
    """
    Extract the answer from a completion string using multiple strategies,
    ordered from most specific to most lenient:
      1. \\boxed{X}
      2. (X) pattern for letters or numbers, e.g. "(A)" or "(42)"
      3. "answer is X" / "therefore X" near end
      4. Bare number at the very end (1-3 digits)
      5. Bare single letter at the very end
    Returns the extracted answer stripped (uppercased for letters), or None.
    """
    if text is None:
        return None

    # Strategy 1: \boxed{...}
    boxed = re.findall(r'\\boxed\{(.*?)}', text, re.DOTALL)
    if boxed:
        return boxed[-1].strip()

    # Strategy 2: Parenthesized letter or number, e.g. (A), (42)
    parens = re.findall(r'\(([A-Za-z]|\d{1,3})\)', text)
    if parens:
        return parens[-1].strip()

    # Strategy 3: "answer is X" / "choose X" / "therefore X" near end
    answer_phrase = re.search(
        r'(?:answer\s+is|choose|select|therefore|thus)[:\s]*([A-Za-z]|\d{1,3})[\.\,\s]*$',
        text, re.IGNORECASE
    )
    if answer_phrase:
        return answer_phrase.group(1).strip()

    # Strategy 4: Bare number (1-3 digits) at the very end
    bare_num = re.search(r'(?<!\d)(\d{1,3})[\.\s]*$', text)
    if bare_num:
        return bare_num.group(1).strip()

    # Strategy 5: Bare single letter at the very end
    bare_letter = re.search(r'(?<![A-Za-z])([A-Za-z])[\.\s]*$', text)
    if bare_letter:
        return bare_letter.group(1).strip()

    return None


def extract_answer_from_solution(sol_text: str) -> str | None:
    """Extract the ground-truth answer from a solution string (expects \\boxed{X})."""
    boxed = re.findall(r'\\boxed\{(.*?)}', sol_text, re.DOTALL)
    if boxed:
        return boxed[-1].strip()
    return sol_text.strip() if sol_text else None


def normalize_for_comparison(answer: str) -> str:
    """Normalize an answer for comparison: strip whitespace, uppercase letters."""
    answer = answer.strip()
    # If it's purely digits, normalize leading zeros: "07" -> "7"
    if answer.isdigit():
        return str(int(answer))
    return answer.upper()


def parse_log_file(log_path: str) -> tuple[list[str], list[str]]:
    """
    Parse the log file to extract completion contents and solution strings.

    Expected log format (from logger.info with the f-string):
      2026-03-16 04:26:49 - main - INFO - before acc reward: prompts: ...
      \\n\\n completions: <list of lists> \\n\\n solutions: <list of lists>

    We strip the timestamp prefix and use regex-based extraction instead of
    ast.literal_eval to avoid issues with leading zeros in timestamps.
    """
    raw_text = Path(log_path).read_text(encoding="utf-8")

    # Strip all timestamp + logger prefixes
    text = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-\s+\S+\s+-\s+\w+\s+-\s+', '', raw_text,
                  flags=re.MULTILINE)

    # --- Extract completions ---
    comp_match = re.search(r'completions:\s*(.+?)\n\n\s*solutions:', text, re.DOTALL)
    if not comp_match:
        raise ValueError("Could not find 'completions: ... solutions:' block in log file.")
    completions_raw = comp_match.group(1).strip()

    # --- Extract solutions ---
    sol_match = re.search(r'solutions:\s*(.+)', text, re.DOTALL)
    if not sol_match:
        raise ValueError("Could not find 'solutions: ...' block in log file.")
    solutions_raw = sol_match.group(1).strip()

    # --- Parse completions by extracting 'content' values ---
    completion_texts = re.findall(r"'content':\s*['\"](.+?)['\"]}\]", completions_raw, re.DOTALL)
    completion_texts = [t.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
                        for t in completion_texts]

    # --- Parse solutions ---
    solution_texts = re.findall(r"\['(.*?)'\]", solutions_raw)
    solution_texts = [t.replace("\\\\", "\\") for t in solution_texts]

    return completion_texts, solution_texts


def evaluate_relaxed(completions: list[str], solutions: list[str]) -> list[dict]:
    """Compare each completion to its solution using relaxed extraction + exact match."""
    results = []
    n = min(len(completions), len(solutions))

    for i in range(n):
        pred_raw = extract_answer_relaxed(completions[i])
        gold_raw = extract_answer_from_solution(solutions[i])

        pred = normalize_for_comparison(pred_raw) if pred_raw is not None else None
        gold = normalize_for_comparison(gold_raw) if gold_raw is not None else None

        correct = (pred is not None and gold is not None and pred == gold)

        results.append({
            "index": i,
            "predicted_raw": pred_raw,
            "predicted": pred,
            "gold_raw": gold_raw,
            "gold": gold,
            "correct": correct,
            "completion_snippet": completions[i][-200:] if completions[i] else "",
        })

    return results


def relaxed_eval(log_file: str,
                 results_json: str | None = None,
                 output_file: str | None = None,
                 verbose: bool = True) -> dict:
    """
    Run relaxed evaluation on a log file.

    Args:
        log_file: Path to the log file containing completions and solutions.
        results_json: Optional path to an existing results JSON (e.g. full_results.json).
                      If provided, a copy is made with the "accuracy" key updated.
        output_file: Path for the output file. Defaults to <log_file>_relaxed_eval.json.
                     If results_json is provided, defaults to <results_json dir>/full_results_relaxed.json.
        verbose: Whether to print summary to stdout.

    Returns:
        dict with keys: total, correct, accuracy, no_prediction_extracted, details
    """
    if output_file is None:
        if results_json is not None:
            output_file = str(Path(results_json).parent / "full_results_relaxed.json")
        else:
            output_file = str(Path(log_file).with_suffix("")) + "_relaxed_eval.json"

    completions, solutions = parse_log_file(log_file)
    results = evaluate_relaxed(completions, solutions)

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    no_pred = sum(1 for r in results if r["predicted"] is None)
    accuracy = correct / total if total > 0 else 0.0

    if verbose:
        print(f"[relaxed_eval] {log_file}")
        print(f"  {total} samples, {correct} correct, acc={accuracy:.4f}, {no_pred} no-answer")

    # If a results JSON is provided, copy it and replace the "accuracy" key
    if results_json is not None:
        with open(results_json, "r", encoding="utf-8") as f:
            existing = json.load(f)

        updated = copy.deepcopy(existing)
        new_accuracy = [1.0 if r["correct"] else 0.0 for r in results]

        # Pad or truncate to match original length if needed
        orig_len = len(updated.get("accuracy", []))
        if orig_len > len(new_accuracy):
            new_accuracy.extend([0.0] * (orig_len - len(new_accuracy)))

        updated["accuracy"] = new_accuracy

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2, ensure_ascii=False)
    else:
        # Write standalone evaluation results
        summary = {
            "total": total,
            "correct": correct,
            "accuracy_score": accuracy,
            "no_prediction_extracted": no_pred,
            "details": results,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"  Written to: {output_file}")

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "no_prediction_extracted": no_pred,
        "details": results,
    }


if __name__ == "__main__":
    """
    # Example usage for quick standalone testing
    import argparse

    parser = argparse.ArgumentParser(description="Relaxed evaluation of logged completions vs solutions")
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--results_json", type=str, default=None,
                        help="Existing full_results.json to update")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    relaxed_eval(
        log_file=args.log_file,
        results_json=args.results_json,
        output_file=args.output_file,
    )
    """

    import os

    log_base_path = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/runs/Qwen3p5_9B/eval"
    #prompt_type = "zoom_in_absolute_q3p5"
    prompt_type = "no_tool"
    DATASETS = []
    for px in [1, 2, 4, 8]:
        for grid_size in [1, 2, 4, 8, 16]:
            #scq

            log_path = os.path.join(log_base_path,
                                    f"dataset_muffin_chihuahua_grid_pixels_{px * 1024}_gridsize_{grid_size}_samples_class_0_{max(1, grid_size ** 2 // 2)}_prompt_{prompt_type}_max_pixels_16777216_min_pixels_65536_padding_0.0")

            DATASETS.append({"log_file": os.path.join(log_path, "evaluation.log"),
                             "results_json": os.path.join(log_path, "full_results.json"),
                             "output_file": os.path.join(log_path, "full_results_relaxed.json")
                             })
            if grid_size > 1:
                #fo
                log_path = os.path.join(log_base_path,
                                        f"dataset_muffin_chihuahua_grid_pixels_{px * 1024}_gridsize_{grid_size}_samples_class_0_1_prompt_{prompt_type}_max_pixels_16777216_min_pixels_65536_padding_0.0")

                DATASETS.append({"log_file": os.path.join(log_path, "evaluation.log"),
                                 "results_json": os.path.join(log_path, "full_results.json"),
                                 "output_file": os.path.join(log_path, "full_results_relaxed.json")
                                 })

    for log in DATASETS:
        if os.path.isfile(log["results_json"]):
            relaxed_eval(**log)
            print("SUCCESS")
        else:
            print(f"WARNING: {log}")


