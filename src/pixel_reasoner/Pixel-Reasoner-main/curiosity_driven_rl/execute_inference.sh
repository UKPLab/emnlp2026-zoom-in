benchmark=infographics
export working_dir="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src/pixel_reasoner/Pixel-Reasoner-main/curiosity_driven_rl"
export policy="TIGER-Lab/PixelReasoner-RL-v1"
export savefolder=tooleval
#export nvj_path="/path/to/nvidia/nvjitlink/lib" # in case the system cannot fiind the nvjit library
############
export sys=vcot # define the system prompt
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution
export eval_bsz=64 # vllm will processes this many queries
export tagname=eval_infographics_test
export testdata="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/pixel_reasoner/data/${benchmark}.parquet"
export num_vllm=1
export num_gpus=8
export tp=1
bash ${working_dir}/scripts/eval_vlm_new.sh