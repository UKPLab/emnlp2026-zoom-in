from aim.storage.context import Context
from aim import Repo
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from scipy.stats import spearmanr
import tikzplotlib

def tikz_export(a:np.ndarray, b:np.ndarray):
    assert len(a) == len(b)
    a = a.round(0)
    b = b.round(2)
    #a.tolist()
    #b.tolist()
    joined_list = [str((int(a[i]),b[i])) for i in range(len(a)) if not np.isnan(b[i])]
    #print(joined_list)
    return " ".join(joined_list)


repo = Repo("/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/")              # path to your .aim repo (often ".")
run = repo.get_run("0833ae641bed4f3bba6b8c5d")
#ours: 0f56e1861e134ecb87759ef6
# cold absolute: 0833ae641bed4f3bba6b8c5d
# no tool 3 epochs: bd4d1bf4cff1423d95851e34
# no tool 1 epoch: 0116dcd95a4d48be9474c8ec

# If you used contexts when tracking, you may need to pass the same context.
# Otherwise omit context.
#metric = run.get_metric("score_time", context={ 'subset':'train' })  # or: run.get_metric("loss", context={...})
#metric = run.metrics().dataframe()
#cols = metric.columns
ctx = Context(context=None)
generate_time_analysis = False
if generate_time_analysis:
    metrics = {}
    for metric_name in ["vllm_generate_time", "completion_length", "mean_tool_use"]:
         steps, values = run.get_metric(metric_name, ctx).values.sparse_numpy()
         if "steps" not in metrics:
             metrics["steps"] = steps
         else:
             assert (steps == metrics["steps"]).all(), f"{steps} != {metrics['steps']}"
         metrics[metric_name] = values

    metrics["mean_tool_use"] = metrics["mean_tool_use"] #* 280

    X = np.stack([metrics["completion_length"],metrics["mean_tool_use"]],axis=1)
    y = metrics["vllm_generate_time"]

    #print(X, y)

    clf = linear_model.LinearRegression()
    clf.fit(X, y)
    #print(clf.coef_)

    for metric_name in ["completion_length", "mean_tool_use"]:
        correl = np.corrcoef(metrics[metric_name], metrics["vllm_generate_time"])
        #print(f"{metric_name} : {correl}")
        correl_spearman = spearmanr(metrics[metric_name], metrics["vllm_generate_time"])

        #print(f"{metric_name}, spearman: {correl_spearman}")

    print(f"overall vllm generate time: {np.sum(metrics['vllm_generate_time'])}")
    print(f"overall number of tool uses: {np.sum(metrics['mean_tool_use'])*280}")
    print(f"overall generated tokens: {np.sum(metrics['completion_length'])*280}")

tool_use_behaviour = True
if tool_use_behaviour:
    repo = Repo("/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/")  # path to your .aim repo (often ".")

    runs = [
        {"hash": "f56b9115cfac4ff39999aafe",
         "name": "random"},
        {
            "hash": "0f56e1861e134ecb87759ef6",
            "name": "ours"
        },
        {
            "hash": "fff44c49d5704883a18ff5bb",
            "name": "per seq"
        },
        {
            "hash": "23813dda1910495f9a3a7fd7",
            "name": "two neg"
        },
        {
            "hash": "8694f9d9d4934147bed4018e",
            "name": "15"
        },
        {
            "hash": "cb5f41cbe20e43058e7b5544",
            "name": "20"
        }

    ]

    #metric = "mean_tool_use"
    #metric = "rewards_non_zero/mutual_information"
    metric = "completion_length_second"

    for run in runs:
        full_run = repo.get_run(run["hash"])#mean_tool_use
        _, tool_use_values = full_run.get_metric(metric, ctx).values.sparse_numpy()
        _, tool_use_steps = full_run.get_metric(metric, ctx).epochs.sparse_numpy()
        #print(len(tool_use_steps))
        #print(len(tool_use_values))
        sort_idxs = np.argsort(tool_use_steps)

        #print(tool_use_values)
        tikz = tikz_export(tool_use_steps[sort_idxs]*382, tool_use_values[sort_idxs])
        print(f"{run['name']}")
        print(tikz)

        #values.sort()
        plt.plot(tool_use_steps[sort_idxs]*382, tool_use_values[sort_idxs], label=run["name"])

    plt.plot([382 * 0.3, 382 * 0.3], [0, 1])
    #plt.ylim([0, 150])
    plt.legend()
    plt.title(metric)
    plt.show()
    #tikzplotlib.save(f"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/paper_export/{metric}.tex")
