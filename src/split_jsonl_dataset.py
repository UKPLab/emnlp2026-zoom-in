# split_dataset.py
import argparse
import math
import os


def split_dataset_file(data_filepath: str, num_shards: int, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)

    with open(data_filepath, "r") as f:
        lines = f.readlines()

    total = len(lines)
    shard_size = math.ceil(total / num_shards)
    shard_paths = []

    for i in range(num_shards):
        start = i * shard_size
        end = min(start + shard_size, total)
        if start >= total:
            break
        shard_path = os.path.join(output_dir, f"test_shard_{i}.jsonl")
        with open(shard_path, "w") as f:
            f.writelines(lines[start:end])
        shard_paths.append(shard_path)

    print(f"Split {total} samples into {len(shard_paths)} shards of ~{shard_size} each")
    for p in shard_paths:
        print(f"  {p}")
    return shard_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a JSONL dataset into N shards")
    parser.add_argument("--data_filepath", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--num_shards", type=int, required=True, help="Number of shards to create")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write shard files into")
    args = parser.parse_args()

    split_dataset_file(args.data_filepath, args.num_shards, args.output_dir)