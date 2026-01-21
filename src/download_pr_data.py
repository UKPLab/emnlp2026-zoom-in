import requests
import os

import pandas as pd
import json
import numpy as np
import sys
import zipfile
import uuid
import base64
import io
from PIL import Image
from tqdm import tqdm


def save_b64_image(b64_string, output_dir):
    """Decodes b64 string and saves directly as jpeg with uuid name"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    file_id = str(uuid.uuid4())
    filename = f"{file_id}.jpg"
    filepath = os.path.join(output_dir, filename)

    # Decode the base64 string to bytes
    img_data = base64.b64decode(b64_string)

    # Write bytes directly to a file
    with open(filepath, 'wb') as f:
        f.write(img_data)

    return filename

def unzip_images(zip_path, save_path, delete_zip=True):
    """Unzip file from zip_path and save to specified path"""

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)
    if delete_zip:
        os.remove(zip_path)


def download_file(url, save_path):
    """Download file from url and save to specified path"""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)



def parquet_to_jsonl(parquet_file_path, jsonl_file_path, dataset, append=False):
    """
    Convert a parquet file to JSONL format.

    Args:
        parquet_file_path (str): Path to the input parquet file
        jsonl_file_path (str): Path to the output JSONL file
    """
    # Read the parquet file
    df = pd.read_parquet(parquet_file_path)

    print(f"datatypes before conversion: {df.dtypes}")
    if dataset == "pixel_reasoner_train":
        # TODO: it is important that we don't check == False, because some fields are None
        df = df[df["is_video"] != True]
        print(len(df))
        #sys.exit()
        df.drop(columns=["is_video"], inplace=True)
        df["question"] = df["question"].astype(str)
        df["qid"] = df["qid"].astype(str)
        df["image"] = df["image"].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))
        df["answer"] = df["answer"].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))
    elif dataset == "pixel_reasoner_vstar":
        df["question"] = df["question"].astype(str)
        df["image"] = df["image"].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))
        df["answer"] = df["answer"].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))
        df = df[['question', 'image', 'answer']]
    elif dataset == "pixel_reasoner_infovqa":
        df["question"] = df["question"].astype(str)
        df["image"] = df["image"].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))
        df["answer"] = df["answer"].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))
        df = df[['question', 'image', 'answer']]
    elif dataset in ["hr_bench_4k", "hr_bench_8k"]:
        img_output_dir = os.path.join(os.path.dirname(jsonl_file_path), "images")
        df["question"] = df["question"].astype(str)
        df["image"] = df["image"].apply(lambda x: save_b64_image(x, img_output_dir))
        df["answer"] = df["answer"].astype(str)
        df["A"] = df["A"].astype(str)
        df["B"] = df["B"].astype(str)
        df["C"] = df["C"].astype(str)
        df["D"] = df["D"].astype(str)
        df = df[['question', 'answer', 'A', 'B', 'C', 'D', 'image']]
    elif dataset in ["mme", "mme_lite"]:

        full_len = len(df)
        #print(f"full length: {len(df)}")
        df = df[df["bytes"].str.startswith("/9j/", na=False)]
        #print(f"length with only jpg: {len(df)}")
        if full_len != len(df):
            print(f"dropped {full_len - len(df)} rows with non-jpg images")
        img_output_dir = os.path.join(os.path.dirname(jsonl_file_path), "images")
        df["question"] = df["question"].astype(str)
        df["image"] = df["bytes"].apply(lambda x: save_b64_image(x, img_output_dir))
        df["answer"] = df["answer"].astype(str)
        df["multi-choice options"] = df["multi-choice options"].apply(lambda x: [str(y) for y in x.tolist()] if isinstance(x, np.ndarray) else x)


        df = df[['question', 'answer', "multi-choice options", 'image']]
        print(df["multi-choice options"][0])

    else:
        raise NotImplementedError(f"dataset {dataset} not implemented")

    #df["image"] = df["image"].astype(list[str])
    print(df.head())

    print(f"datatypes after conversion: {df.dtypes}")


    # Convert to JSONL
    with open(jsonl_file_path, 'w' if not append else 'a', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Convert row to dictionary and then to JSON string
            json_line = json.dumps(row.to_dict(), ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"Successfully converted {parquet_file_path} to {jsonl_file_path}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":

    datasets = [{"dataset_name": "pixel_reasoner_infovqa",
                 "remote_base": "https://huggingface.co/datasets/JasperHaozhe/InfoVQA-EvalData-PixelReasoner/resolve/main/",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/eval/Infographics_VQA/",
                 "parquet_file": "infographics.parquet"},
                {"dataset_name": "hr_bench_4k",
                 "remote_base": "https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/HR_Bench_4k/",
                 "parquet_file": "hr_bench_4k.parquet"},
                {"dataset_name": "hr_bench_8k",
                 "remote_base": "https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/HR_Bench_8k/",
                 "parquet_file": "hr_bench_8k.parquet"},
                {"dataset_name": "mme_lite",
                 "remote_base": "https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-lite-lmms-eval/resolve/main/data",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld-lite/",
                 "parquet_file": "train-0000NUM-of-00004.parquet",
                 "parquet_file_count": 4},
                {"dataset_name": "mme",
                 "remote_base": "https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lmms-eval/resolve/main/data",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld/",
                 "parquet_file": "train-0000NUM-of-00070.parquet",
                 "parquet_file_count": 70}
                ]

    dataset = datasets[4]

    remote_base = dataset["remote_base"]
    local_base = dataset["local_base"]
    #os.makedirs(local_base, exist_ok=False)

    #zip_path = os.path.join(local_base, "images.zip")

    parquet_path = os.path.join(local_base, "images.parquet")

    #download_file(os.path.join(remote_base, "images.zip"), zip_path)

    #unzip_images(zip_path, local_base, delete_zip=False)


    if "parquet_file_count" in dataset.keys():
        for idx in tqdm(range(dataset["parquet_file_count"])):
            parquet_path = os.path.join(local_base, f"images_{idx}.parquet")
            #download_file(os.path.join(remote_base, dataset["parquet_file"].replace("0"*(len(str(idx))-1) + "NUM", str(idx))),
            #          parquet_path)
            parquet_to_jsonl(parquet_path,
                             os.path.join(local_base, "test.jsonl"),
                             dataset["dataset_name"],
                             append=True)
    else:
        parquet_path = os.path.join(local_base, "images.parquet")
        download_file(os.path.join(remote_base, dataset["parquet_file"]),
                      parquet_path)
        parquet_to_jsonl(parquet_path,
                         os.path.join(local_base, "test.jsonl"),
                         dataset["dataset_name"],
                         append=False)


