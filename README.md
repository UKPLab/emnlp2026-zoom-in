# Code for ARR March Submission "Learning to Zoom Efficiently with a Contrastive Curriculum"

## Setup
First, do ``pip install -r requirements.txt``.

## Dataset construction

For construction of the M&C dataset, do

`
generate_synthetic_grid_data.py --download_dir=/image/download/path --save_path_prefix=/save/path/generated/splits
`

which downloads the source images, preprocesses them, generates image grids and finally generates textual questions for the 
M&C VQA dataset. The image grid generation takes several hours but can be resumed.
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
