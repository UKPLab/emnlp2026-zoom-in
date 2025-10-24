import requests
import os

import pandas as pd
import json
import numpy as np
import sys
import zipfile

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



def parquet_to_jsonl(parquet_file_path, jsonl_file_path, dataset):
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

    #df["image"] = df["image"].astype(list[str])
    print(df.head())
    print(f"datatypes after conversion: {df.dtypes}")


    # Convert to JSONL
    with open(jsonl_file_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Convert row to dictionary and then to JSON string
            json_line = json.dumps(row.to_dict(), ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"Successfully converted {parquet_file_path} to {jsonl_file_path}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":

    remote_base = "https://huggingface.co/datasets/JasperHaozhe/InfoVQA-EvalData-PixelReasoner/resolve/main/"
    local_base = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/eval/Infographics_VQA/"
    #os.makedirs(local_base, exist_ok=False)

    zip_path = os.path.join(local_base, "images.zip")
    parquet_path = os.path.join(local_base, "infographics.parquet")

    #download_file(os.path.join(remote_base, "images.zip"), zip_path)

    #unzip_images(zip_path, local_base, delete_zip=False)

    download_file(os.path.join(remote_base, "infographics.parquet"),
                  parquet_path)

    parquet_to_jsonl(parquet_path,
                     os.path.join(local_base, "test.jsonl"),
                     "pixel_reasoner_infovqa")
