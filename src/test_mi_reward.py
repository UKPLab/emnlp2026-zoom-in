from open_r1.utils.rewards import mutual_information_reward
import torch

contrast_diff_list = [
    None,
    torch.arange(50)
]


r = mutual_information_reward(
    contrast_diff_list=contrast_diff_list,
    gamma=1.5,
    length_factor_scaling=1.0,
    tanh=True,
    select_k=40,
    select_k_type="first",
    absolute_diff=None,
    contrasted_area=None,
    alpha=None,
    delta=None,
    q=None,
    tau=None,
    discretize=False,
    ignored_prefix_len=None,
    length_factor=1.0
)

print(r)