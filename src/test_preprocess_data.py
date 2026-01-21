from open_r1.preprocess_data import prepare_data
from open_r1.utils.prompts import get_question_template
from open_r1.utils.logger import setup_project_logging

setup_project_logging(None)

prepare_data(dataset_names=["pixel_reasoner_infovqa"],
             data_folders=["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/eval/Infographics_VQA/test.jsonl"],
             image_folders=["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/eval/Infographics_VQA"],
             question_prompt=get_question_template(task_type="no_tool"),
             reward_method=None)