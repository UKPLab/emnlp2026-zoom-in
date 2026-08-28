"""Download the third-party VQA benchmarks / training sets used in the paper and
convert them into the **canonical JSONL** format consumed by
``open_r1.preprocess_data`` (the paper's own M&C dataset is emitted in that
format directly by ``generate_synthetic_grid_data.py``, so it needs no step here).

Canonical record (one JSON object per line)::

    {"problem": <str>, "image": <str|list[str]>, "solution": <str|list[str]>}

``image`` paths are written relative to an ``images/`` folder that sits next to
the produced ``test.jsonl``; point the train/eval scripts at that folder.

Usage::

    python download_data.py --dataset hr_bench_4k --out_dir /data/hr_bench_4k
    # partial download of a sharded set (e.g. just the first shard of MME):
    python download_data.py --dataset mme --out_dir /data/mme --max_shards 1
    # re-convert already-downloaded files without downloading again:
    python download_data.py --dataset mme_lite --out_dir /data/mme_lite --no_download

Provenance note: for V* we use PixelReasoner's packaging of the benchmark (as in
the paper), whose canonical original is ``craigwu/vstar_bench``. The exact
download layout of the training sets (Pixel-Reasoner, VisualProbe, DeepEyes) and
V* can vary between releases -- verify the produced ``test.jsonl`` before a full
run (a partial download is enough to check the format).
"""
import argparse
import ast
import base64
import json
import os
import uuid

import numpy as np
import pandas as pd
import requests
from huggingface_hub import snapshot_download


# --------------------------------------------------------------------------- #
# Download / image helpers
# --------------------------------------------------------------------------- #
def save_b64_image(b64_string, output_dir):
    """Decode a base64 string and save it as a jpeg with a uuid name."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}.jpg"
    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(base64.b64decode(b64_string))
    return filename


def download_file(url, save_path):
    """Stream a file from ``url`` to ``save_path``."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def download_hf_dir(repo_id, save_dir, revision="main"):
    """Download a whole HF dataset repo into ``save_dir`` (files, not symlinks)."""
    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )


# --------------------------------------------------------------------------- #
# Per-record converters: intermediate row (dict) -> canonical record.
# Return None to drop a record. These hold the only dataset-specific logic.
# --------------------------------------------------------------------------- #
def convert_pixel_reasoner(item):
    """Training set. Drops multi-image records (single-image only)."""
    images = ast.literal_eval(item["image"])
    if len(images) != 1:
        return None
    answer = ast.literal_eval(item["answer"])
    assert len(answer) == 1, f"Only one answer is expected, but got {answer}"
    answer = answer[0][7:-1]  # strip the surrounding \boxed{ ... }
    if answer and answer[0] == "[":
        try:
            answer = ast.literal_eval(answer)
        except (ValueError, SyntaxError):
            answer = [answer]
    else:
        answer = [answer]
    return {"problem": item["question"], "image": images, "solution": answer}


def convert_pixel_reasoner_vstar(item):
    problem = str(item["question"]).removeprefix("<image>\n")
    image = item["image"]
    if isinstance(image, str):
        image = ast.literal_eval(image) if image.startswith("[") else [image]
    elif hasattr(image, "tolist"):  # numpy array -> plain list (JSON-serialisable)
        image = image.tolist()
    answer = str(item["answer"]).removeprefix("\\boxed{").removesuffix("}")
    return {"problem": problem, "image": image, "solution": answer}


def convert_hr_bench(item):
    problem = (f"{item['question']}\n(A) {item['A']}"
               f"\n(B) {item['B']}"
               f"\n(C) {item['C']}"
               f"\n(D) {item['D']}"
               f"\nAnswer with the option's letter from the given choices directly.")
    return {"problem": problem, "image": item["image"], "solution": item["answer"]}


def convert_mme(item):
    options = "\n".join(item["multi-choice options"])
    problem = (f"{item['question']}\n{options}"
               f"\nAnswer with the option's letter from the given choices directly.")
    return {"problem": problem, "image": item["image"], "solution": item["answer"]}


def convert_visual_probe(item):
    assert len(item["image"]) == 1, f"Only one image is expected, but got {item['image']}"
    return {"problem": item["question"], "image": item["image"][0], "solution": item["answer"]}


def convert_deepeyes(item):
    assert len(item["image"]) == 1, f"Only one image is expected, but got {item['image']}"
    problem = (f"{item['question']}\nIf there are choices given, answer with the "
               f"option's letter from the given choices directly.")
    return {"problem": problem, "image": item["image"][0], "solution": item["answer"]}


CANONICAL_CONVERTERS = {
    "pixel_reasoner": convert_pixel_reasoner,
    "pixel_reasoner_vstar": convert_pixel_reasoner_vstar,
    "hr_bench_4k": convert_hr_bench,
    "hr_bench_8k": convert_hr_bench,
    "mme": convert_mme,
    "mme_lite": convert_mme,
    "visual_probe_train": convert_visual_probe,
    "deepeyes_train_4k": convert_deepeyes,
}


# --------------------------------------------------------------------------- #
# Stage A: raw parquet / json -> intermediate DataFrame (extracts images,
# filters, normalises columns). Kept per-dataset, faithful to the original prep.
# --------------------------------------------------------------------------- #
def load_intermediate_df(source_file_path, dataset, img_output_dir, dataset_format="parquet"):
    if dataset_format == "parquet":
        df = pd.read_parquet(source_file_path)
    elif dataset_format == "json":
        # .jsonl is line-delimited; .json is a single array.
        df = pd.read_json(source_file_path, lines=source_file_path.endswith(".jsonl"))
    else:
        raise ValueError(f"unknown dataset_format {dataset_format}")

    if dataset == "pixel_reasoner":
        # NB: compare against True explicitly -- some rows have is_video == None.
        df = df[df["is_video"] != True]
        df = df.drop(columns=["is_video"])
        df["question"] = df["question"].astype(str)
        df["image"] = df["image"].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))
        df["answer"] = df["answer"].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))

    elif dataset == "pixel_reasoner_vstar":
        # V* references images by path and already has boxed answers; the
        # per-record converter strips the <image> prefix and the \boxed{...}.
        df["question"] = df["question"].astype(str)
        df["answer"] = df["answer"].astype(str)

    elif dataset in ("hr_bench_4k", "hr_bench_8k"):
        df["question"] = df["question"].astype(str)
        df["image"] = df["image"].apply(lambda x: save_b64_image(x, img_output_dir))
        df["answer"] = df["answer"].astype(str)
        for col in ("A", "B", "C", "D"):
            df[col] = df[col].astype(str)

    elif dataset in ("mme", "mme_lite"):
        full_len = len(df)
        df = df[df["bytes"].str.startswith("/9j/", na=False)]  # keep jpeg-encoded rows
        if full_len != len(df):
            print(f"dropped {full_len - len(df)} rows with non-jpg images")
        df["question"] = df["question"].astype(str)
        df["image"] = df["bytes"].apply(lambda x: save_b64_image(x, img_output_dir))
        df["answer"] = df["answer"].astype(str)
        df["multi-choice options"] = df["multi-choice options"].apply(
            lambda x: [str(y) for y in x.tolist()] if isinstance(x, np.ndarray) else x)

    elif dataset == "visual_probe_train":
        df["question"] = df["problem"].apply(lambda s: s.removeprefix("<image>\n"))
        df["answer"] = df["solution"].apply(lambda s: s.strip())
        df["image"] = df["images"].apply(lambda xx: [f"images/{x.split('/')[-1]}" for x in xx])

    elif dataset == "deepeyes_train_4k":
        # DeepEyes ships long answers; short answers come from a sibling file
        # (deepeyes_short_answer.jsonl) that must be present next to the source.
        df["question"] = df["problem"].apply(lambda s: s.removeprefix("<image>\n"))
        df["image"] = df["images"].apply(lambda xx: [f"images/{x.split('/')[-1]}" for x in xx])
        short_answer_path = os.path.join(os.path.dirname(os.path.abspath(source_file_path)),
                                         "deepeyes_short_answer.jsonl")
        df_short = pd.read_json(short_answer_path, lines=True)
        assert df["question"].equals(df_short["question"]), "short-answer file is misaligned"
        df["answer"] = df_short["answer"].values

    else:
        raise NotImplementedError(f"dataset {dataset} not implemented")

    return df


def convert_to_jsonl(source_file_path, jsonl_file_path, dataset, append=False, dataset_format="parquet"):
    """Read a raw source file and append its canonical records to ``jsonl_file_path``."""
    img_output_dir = os.path.join(os.path.dirname(jsonl_file_path), "images")
    df = load_intermediate_df(source_file_path, dataset, img_output_dir, dataset_format)
    convert = CANONICAL_CONVERTERS[dataset]

    num_written = 0
    with open(jsonl_file_path, "a" if append else "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = convert(row.to_dict())
            if record is None:  # dropped (e.g. multi-image Pixel-Reasoner rows)
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1
    return num_written


# --------------------------------------------------------------------------- #
# Dataset registry (remote sources). `type` selects the download strategy.
# --------------------------------------------------------------------------- #
DATASETS = {
    "hr_bench_4k": {
        "type": "parquet",
        "url": "https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_4k.parquet",
    },
    "hr_bench_8k": {
        "type": "parquet",
        "url": "https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_8k.parquet",
    },
    "mme_lite": {
        "type": "parquet_sharded",
        "url_tmpl": "https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-lite-lmms-eval/resolve/main/data/train-{i:05d}-of-{n:05d}.parquet",
        "shards": 4,
    },
    "mme": {
        "type": "parquet_sharded",
        "url_tmpl": "https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lmms-eval/resolve/main/data/train-{i:05d}-of-{n:05d}.parquet",
        "shards": 70,
    },
    # Training / eval sets pulled as whole HF repos. `source_file` is the text
    # file inside the repo; verify it for your download (layouts vary by release).
    "pixel_reasoner": {  # training data (drops video + multi-image records)
        "type": "hf_dir",
        "repo_id": "TIGER-Lab/PixelReasoner-RL-Data",
        "source_file": "train.jsonl",
        "source_format": "json",
    },
    "visual_probe_train": {
        "type": "hf_dir",
        "repo_id": "Mini-o3/VisualProbe_train",
        "source_file": "train.json",
        "source_format": "json",
    },
    "deepeyes_train_4k": {
        "type": "hf_dir",
        "repo_id": "Mini-o3/DeepEyes_train_4K",
        "source_file": "train.json",
        "source_format": "json",
    },
    "pixel_reasoner_vstar": {  # paper uses PixelReasoner's packaging; original below
        "type": "hf_dir",
        "repo_id": "craigwu/vstar_bench",
        "source_file": "test.jsonl",
        "source_format": "json",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Download a third-party dataset and convert it to canonical JSONL")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--out_dir", required=True,
                        help="output dir for downloads, images/ and the canonical test.jsonl")
    parser.add_argument("--max_shards", type=int, default=None,
                        help="for sharded datasets, only fetch/convert the first N shards")
    parser.add_argument("--no_download", action="store_true",
                        help="skip downloading; convert files already present in out_dir")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "test.jsonl")

    if cfg["type"] == "parquet":
        parquet_path = os.path.join(args.out_dir, "images.parquet")
        if not args.no_download:
            download_file(cfg["url"], parquet_path)
        total = convert_to_jsonl(parquet_path, out_jsonl, args.dataset, append=False)

    elif cfg["type"] == "parquet_sharded":
        n_shards = cfg["shards"] if args.max_shards is None else min(args.max_shards, cfg["shards"])
        total = 0
        for i in range(n_shards):
            parquet_path = os.path.join(args.out_dir, f"images_{i}.parquet")
            if not args.no_download:
                download_file(cfg["url_tmpl"].format(i=i, n=cfg["shards"]), parquet_path)
            total += convert_to_jsonl(parquet_path, out_jsonl, args.dataset, append=(i > 0))

    elif cfg["type"] == "hf_dir":
        if not args.no_download:
            download_hf_dir(cfg["repo_id"], args.out_dir)
        source_path = os.path.join(args.out_dir, cfg["source_file"])
        total = convert_to_jsonl(source_path, out_jsonl, args.dataset,
                                 dataset_format=cfg["source_format"])

    else:
        raise ValueError(f"unknown dataset type {cfg['type']}")

    print(f"Wrote {total} canonical records to {out_jsonl}")


if __name__ == "__main__":
    main()
