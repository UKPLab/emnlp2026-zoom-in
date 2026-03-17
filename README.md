# Code for ARR March Submission "Learning to Zoom Efficiently with a Contrastive Curriculum"

## Setup
First, do ``pip install -r requirements.txt``.

## Dataset construction

For construction of the M&C dataset, set `train_dir` in `generate_synthetic_grid_data.py` and then run the script

``
generate_synthetic_grid_data.py
``

## Training
To start the training, specify the script in `train_scheduler.py` and execute it 

``
train_scheduler.py
``

## Evaluation

For evaluation, specify the dataset paths in `initiate_analysis.py` and enable evaluation
in its `get_models_input` method. 



## Model analysis

For model analysis, specify the dataset paths in `initiate_analysis.py` and enable evaluation
in its `get_models_input` method. 
