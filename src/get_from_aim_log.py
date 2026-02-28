from aim.storage.context import Context
from aim import Repo
import numpy as np

repo = Repo("/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/")              # path to your .aim repo (often ".")
run = repo.get_run("bd4d1bf4cff1423d95851e34")
#ours: 0f56e1861e134ecb87759ef6
# cold absolute: 0833ae641bed4f3bba6b8c5d
# no tool 3 epochs: bd4d1bf4cff1423d95851e34

# If you used contexts when tracking, you may need to pass the same context.
# Otherwise omit context.
#metric = run.get_metric("score_time", context={ 'subset':'train' })  # or: run.get_metric("loss", context={...})
#metric = run.metrics().dataframe()
#cols = metric.columns
ctx = Context(context=None)
metric = run.get_metric("score_time", ctx)

# Get values as numpy arrays (sparse) then turn into lists if you want
steps_np, values_np = metric.values.sparse_numpy()  # (x, y)
steps = steps_np.tolist()
values = values_np.tolist()

print(len(values), values[:5])
print("sum(values) =", float(np.sum(values_np)))