import time

import requests
import os
from pathlib import Path

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
from huggingface_hub import snapshot_download


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

def download_hf_dir(repo_id: str, save_dir: str, revision: str = "main") -> str:
    """
    Downloads only the *files directly inside* repo_subdir (depth=1) from a Hugging Face dataset repo.
    Returns the path to the local snapshot directory.
    """


    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=save_dir,
        local_dir_use_symlinks=False,  # copy files instead of symlinks
        resume_download=True
    )
    return snapshot_path

def compare_column_and_get_diffs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    col1: str,
    col2: str | None = None,
    *,
    treat_nan_as_equal: bool = True,
    require_same_index: bool = True,
) -> tuple[bool, pd.DataFrame]:
    """
    Compare df1[col1] vs df2[col2] row-by-row.

    Returns:
      (same, diff_rows)
        - same: True if the compared Series are identical (values + index)
        - diff_rows: rows where the values differ, with both columns side-by-side

    Notes:
      - This corresponds to the "same row order / same length" case.
      - If require_same_index=True, df1 and df2 must have identical index.
    """
    col2 = col1 if col2 is None else col2

    if require_same_index and not df1.index.equals(df2.index):
        raise ValueError("df1 and df2 indexes differ; align/reindex or set require_same_index=False.")

    s1 = df1[col1]
    s2 = df2[col2]

    same = s1.equals(s2)

    if treat_nan_as_equal:
        mask = ~(s1.eq(s2) | (s1.isna() & s2.isna()))
    else:
        mask = ~s1.eq(s2)

    diff_rows = pd.DataFrame(
        {col1: s1, col2: s2},
        index=df1.index,
    ).loc[mask]

    return same, diff_rows


def convert_to_jsonl(source_file_path, jsonl_file_path, dataset, append=False, dataset_format="parquet"):
    """
    Convert a parquet file to JSONL format.

    Args:
        parquet_file_path (str): Path to the input parquet file
        jsonl_file_path (str): Path to the output JSONL file
    """
    if dataset_format == "parquet":
        # Read the parquet file
        df = pd.read_parquet(source_file_path)
    elif dataset_format == "json":
        df = pd.read_json(source_file_path)

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

    elif dataset in ["visual_probe_train"]:
        df["question"] = df["problem"].apply(lambda s: s.removeprefix("<image>\n"))
        df["answer"] = df["solution"].apply(lambda s: s.strip())
        df["image"] = df["images"].apply(lambda xx: [f"images/{x.split("/")[-1]}" for x in xx])

        df = df[['question', 'answer', "image"]]
    elif dataset in ["deepeyes_train_4k"]:

        df["question"] = df["problem"].apply(lambda s: s.removeprefix("<image>\n"))
        df["image"] = df["images"].apply(lambda xx: [f"images/{x.split("/")[-1]}" for x in xx])
        df["answer"] = df["solution"]

        df.rename(columns={"answer": 'long_answer'}, inplace=True)

        short_answer_path = os.path.join(Path(source_file_path).parent.absolute(),"deepeyes_short_answer.jsonl")
        json_list = []
        with open(short_answer_path, 'r') as f:
            for i, line in enumerate(f):
                json_list.append(json.loads(line))
        df_short_answer = pd.DataFrame(json_list)

        assert df["question"].equals(df_short_answer["question"])

        df_short_answer.drop(columns=["question"], inplace=True)

        joint_df = pd.concat([df, df_short_answer], axis=1)

        df = joint_df[['question', 'answer', "image"]]

    else:
        raise NotImplementedError(f"dataset {dataset} not implemented")

    #df["image"] = df["image"].astype(list[str])
    print(df.head())
    print(df.tail())

    print(f"datatypes after conversion: {df.dtypes}")


    # Convert to JSONL
    with open(jsonl_file_path, 'w' if not append else 'a', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Convert row to dictionary and then to JSON string
            json_line = json.dumps(row.to_dict(), ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"Successfully converted {source_file_path} to {jsonl_file_path}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":

        datasets = [{"dataset_name": "pixel_reasoner_infovqa",
                        "dataset_type": "parquet",
                 "remote_base": "https://huggingface.co/datasets/JasperHaozhe/InfoVQA-EvalData-PixelReasoner/resolve/main/",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/eval/Infographics_VQA/",
                 "parquet_file": "infographics.parquet"},
                {"dataset_name": "hr_bench_4k",
                 "dataset_type": "parquet",
                 "remote_base": "https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/HR_Bench_4k/",
                 "parquet_file": "hr_bench_4k.parquet"},
                {"dataset_name": "hr_bench_8k",
                 "dataset_type": "parquet",
                 "remote_base": "https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/HR_Bench_8k/",
                 "parquet_file": "hr_bench_8k.parquet"},
                {"dataset_name": "mme_lite",
                 "dataset_type": "parquet",
                 "remote_base": "https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-lite-lmms-eval/resolve/main/data",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld-lite/",
                 "parquet_file": "train-0000NUM-of-00004.parquet",
                 "parquet_file_count": 4},
                {"dataset_name": "mme",
                 "dataset_type": "parquet",
                 "remote_base": "https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lmms-eval/resolve/main/data",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld/",
                 "parquet_file": "train-0000NUM-of-00070.parquet",
                 "parquet_file_count": 70},

                {"dataset_name": "visual_probe_train",
                 "dataset_type": "directory",
                 "remote_repo_id": "Mini-o3/VisualProbe_train",
                 "remote_text_data": "https://huggingface.co/datasets/Mini-o3/VisualProbe_train/resolve/main/train.json",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/mini_o3/visual_probe_train",
                 },

                {"dataset_name": "deepeyes_train_4k",
                 "dataset_type": "directory",
                 "remote_repo_id": "Mini-o3/DeepEyes_train_4K",
                 "remote_text_data": "https://huggingface.co/datasets/Mini-o3/VisualProbe_train/resolve/main/train.json",
                 "local_base": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/mini_o3/deepeyes_train_4k",
                 }

                ]

        do_download = False

        for idx in [6]:
            dataset = datasets[idx]
            if dataset["dataset_type"] == "parquet":
                remote_base = dataset["remote_base"]
                local_base = dataset["local_base"]
                #os.makedirs(local_base, exist_ok=False)

                #zip_path = os.path.join(local_base, "images.zip")

                parquet_path = os.path.join(local_base, "images.parquet")
                #if do_download:
                #    download_file(os.path.join(remote_base, "images.zip"), zip_path)

                #unzip_images(zip_path, local_base, delete_zip=False)


                if "parquet_file_count" in dataset.keys():
                    for idx in tqdm(range(dataset["parquet_file_count"])):
                        parquet_path = os.path.join(local_base, f"images_{idx}.parquet")
                        #download_file(os.path.join(remote_base, dataset["parquet_file"].replace("0"*(len(str(idx))-1) + "NUM", str(idx))),
                        #          parquet_path)
                        convert_to_jsonl(parquet_path,
                                         os.path.join(local_base, "test.jsonl"),
                                         dataset["dataset_name"],
                                         append=True)
                else:
                    parquet_path = os.path.join(local_base, "images.parquet")
                    if do_download:
                        download_file(os.path.join(remote_base, dataset["parquet_file"]),
                                  parquet_path)
                    convert_to_jsonl(parquet_path,
                                     os.path.join(local_base, "test.jsonl"),
                                     dataset["dataset_name"],
                                     append=False)

            elif dataset["dataset_type"] == "directory":
                if do_download:
                    while True:
                        try:
                            download_hf_dir(repo_id = dataset["remote_repo_id"],
                                        save_dir = dataset["local_base"])
                            break
                        except Exception as e:
                            time.sleep(5*60)
                convert_to_jsonl(
                    os.path.join(dataset["local_base"], "original.json"),
                    os.path.join(dataset["local_base"], "train.jsonl"),
                    dataset["dataset_name"],
                    dataset_format="json",
                )



