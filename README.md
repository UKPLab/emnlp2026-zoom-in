<p  align="center">
  <img src='Figure_1.png' width='1000'>
</p>

# Learning to Zoom Efficiently with a Contrastive Curriculum

## Overview

This repository provides the code for our paper "Learning to Zoom Efficiently with a Contrastive Curriculum", accepted to EMNLP 2026.
It contains code to recreate our proposed M&C dataset and run training and evaluation to reproduce our results.

## Setup
First, do ``pip install --no-deps -r requirements.txt`` in python 3.12.

## Dataset construction
Note: Due to copyright issues we can not distribute the dataset directly. You need to rebuild it.

For construction of the M&C dataset, do

`
generate_synthetic_grid_data.py --download_dir=/image/download/path --save_path_prefix=/save/path/generated/splits
`

which downloads the source images into `download_dir`, preprocesses them, generates image grids and finally generates textual questions for the 
M&C VQA dataset which are saved under `save_path_prefix`. The image grid generation takes several hours but can be resumed.

## Data format

Training and evaluation read datasets in one **canonical format**: a JSONL file plus a
separate image folder. Each line is one example:

```json
{"problem": "<full question text>", "image": "img_0.png", "solution": "A"}
```

- `problem`: the complete question text (any multiple-choice options are already part of it).
- `image`: a filename **or** a list of filenames, relative to the `--image_folders` / `--image_filepath` argument.
- `solution`: the gold answer as a string, or a list of strings.
- extra keys (e.g. `bbox`) are passed through unchanged.

Point the scripts at the JSONL with `--data_file_paths` / `--data_filepath` and at the image
folder with `--image_folders` / `--image_filepath` (during training, use `:` to combine several
datasets). The M&C generator above already emits this format, so its output is used directly.

The other (third-party) benchmarks are fetched and converted to the canonical format with
`download_data.py`, e.g.

```
python download_data.py --dataset hr_bench_4k --out_dir /data/hr_bench_4k
python download_data.py --dataset mme --out_dir /data/mme --max_shards 1   # partial download of a sharded set
```

Supported `--dataset` values: `hr_bench_4k`, `hr_bench_8k`, `mme`, `mme_lite`,
`pixel_reasoner`, `pixel_reasoner_vstar`, `visual_probe_train`, `deepeyes_train_4k`. For V* we use
PixelReasoner's packaging (original: `craigwu/vstar_bench`). Dataset layouts vary between releases,
so verify the produced `test.jsonl` before a full run — a partial download (`--max_shards 1`) is
enough to check the format.

## Training environment: CUDA toolkit & DeepSpeed

DeepSpeed (ZeRO-3 + CPU-Adam) JIT-compiles CUDA ops at training start, so the env needs an `nvcc`
matching your PyTorch CUDA build (12.4, pinned in `requirements.txt`) plus a compiler. conda is the
easiest rootless way; system CUDA / `module load` / a CUDA-devel image work too:
```
conda install -c conda-forge gxx_linux-64=14.3.0
conda install --override-channels -c nvidia/label/cuda-12.4.1 cuda-toolkit=12.4.1 cuda-nvcc cuda-cudart-dev
nvcc --version   # must report 12.4, not 13.x
```
`--override-channels` and the pinned version are required — otherwise conda pulls CUDA 13.x from
`defaults` and `nvcc` won't match. Then export these (persist via `conda env config vars set`, since
`train_scheduler.py` starts `trl vllm-serve` before any Python runs):
```
export CUDA_HOME=$CONDA_PREFIX
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=$CONDA_PREFIX/deepspeed_torch_extensions
```
Verify with `python -c 'import deepspeed; deepspeed.ops.op_builder.CPUAdamBuilder().load()'`.

## Training
**Requirements:** the scheduler uses GNU `screen` (install with `sudo apt install screen`, or
without root via `conda install -c conda-forge screen`) and assumes a single 8-GPU node (GPU 0
serves vLLM, GPUs 1–7 run training). It launches two detached background sessions per run —
`<shell>_auto_vllm` (vLLM server) and `<shell>_auto_run` (trainer) — and then polls until training
finishes; you don't need to interact with them. To watch progress, either `tail -f` the
`vllm_log.txt` / `run_log.txt` files in the run's output directory, or attach with `screen -r <name>`
(detach again with `Ctrl-A` then `D`).

To start the training, specify the script in `train_scheduler.py` and execute it 

``
train_scheduler.py
``

Sample scripts are given in the `scripts` directory but the paths need to be adapted to your setup.
## Evaluation

For evaluation, specify the dataset paths in `initiate_analysis.py` and enable evaluation
in its `get_models_input` method. 



## Model analysis

For model analysis, specify the dataset paths in `initiate_analysis.py` and enable evaluation
in its `get_models_input` method. 

## Contact
Responsible person: Falko Helm, mail: falko.helm@tu-darmstadt.de, github: falko1

Affiliations: 
- UKP Lab: https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp
- TU Darmstadt: https://www.tu-darmstadt.de/
- Hessian AI: https://www.hessian.ai/
- JAIF: https://www.fz-juelich.de/en/jsc/jupiter/jaif-jupiter-ai-factory

If you find this repo helpful, please consider citing us:

Helm, Falko and Gurevych, Iryna: Learning to Zoom Efficiently with a Contrastive Curriculum. To appear in Empirical Methods of Natural Language Processing (EMNLP) 2026



## Disclaimer
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 