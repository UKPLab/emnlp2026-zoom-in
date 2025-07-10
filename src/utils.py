import os
import json
import shutil


def preprocess_chartqa_original(input_path, split, output_path):
    #refocus_format = os.path.join(input_path, split, "train_augmented.json")


    # Load the JSON file refocus_format. It contains a list of JSON objects.

    with open(input_path, 'r') as file:
        data = json.load(file)  # Load the JSON file into a Python object (usually a list of dictionaries).

    #os.makedirs(os.path.join(output_path, split, "png"), exist_ok=True)

    new_jsonl = []
    # Cycle through each JSON object in the list.
    with open(output_path, 'w', encoding='utf-8') as fixed_file:
        for idx, item in enumerate(data):
            print(item)
            #new_img_path = os.path.join(output_path, split, "png", item["imgname"])

            #input_img_path = os.path.join(input_path, split, "png", item["imgname"])

            #shutil.copyfile(input_img_path, new_img_path)

            new_format = {
                "id": idx,
                "image": os.path.join("png", item["imgname"]),
                "conversations": [
                    {"from": "human", "value": f"<image>{item['query']}"},
                    {"from": "gpt", "value": item["label"]}
                ]
            }
            #new_jsonl.append(new_format)
            # Save the processed data to a .jsonl file
            #output_file_path = os.path.join(output_path, split, f"{split}.jsonl")

            # print(obj["image"])
            #obj["image"] = obj["image"].replace("\\", "/")
            # print(obj["image"])
            # Write each JSON object on a new line as a proper JSONL
            fixed_file.write(json.dumps(new_format) + '\n')


if __name__ == "__main__":
    input_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train_full/train_augmented.json"
    output_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train_full/train_augmented_GRPO_format.json"
    split = "train"
    preprocess_chartqa_original(input_path, split, output_path)
