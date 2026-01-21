import json
import os

file = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/src/scripts/zero3.json"

print(os.path.exists(file))

with open(file, 'r', encoding="utf-8") as f:
    data = f.read()

parsed_data = json.loads(data)  # Ensure the data is loaded correctly
print(parsed_data)

def repair_json(corrupted_file_path, fixed_file_path):

    # Open the corrupted file and process it
    with open(corrupted_file_path, 'r', encoding='utf-8') as corrupted_file:
        # Parse the full content as a JSON array
        data = json.load(corrupted_file)



    # Rewrite the data as proper JSONL
    with open(fixed_file_path, 'w', encoding='utf-8') as fixed_file:
        for obj in data:
            #print(obj["image"])
            obj["image"] = obj["image"].replace("\\", "/")
            #print(obj["image"])
            # Write each JSON object on a new line as a proper JSONL
            fixed_file.write(json.dumps(obj) + '\n')

    print(f"The fixed JSONL file has been saved to '{fixed_file_path}'")

if __name__ == "__main__":
    repair_json("/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/train.jsonl",
                "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train/train_fixed_new.jsonl")