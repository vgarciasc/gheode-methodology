import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# lag_times[i, k] is the lag time for the i-th attribute and the k-th cluster.
# timesteps[i, k] is the number of time steps for the i-th attribute and the k-th cluster.
# masks[k] is a boolean indicating whether the k-th cluster is considered or not.
def filter_dataset(df, K, attr_names, lag_times, timesteps, masks, agg_method=None, cols_to_maintain=None):
    new_col_names = []
    new_col_data = []

    for i, attr_name in enumerate(attr_names):
        for k in range(K):
            if masks[k] == 0:
                continue

            lag_time = lag_times[i, k]

            if agg_method is None:
                col = f"{attr_name}_{k}"
            elif type(agg_method) == str:
                col = f"{attr_name}_{agg_method}_{k}"
            elif type(agg_method) == list:
                col = f"{attr_name}_{agg_method[i]}_{k}"
            else:
                raise Exception("agg_method must be a string or a list of strings")

            for t in range(timesteps[i, k]):
                new_col_names.append(f"{col}_t-{lag_time + t}")
                new_col_data.append(df[col].shift(lag_time + t))

    if cols_to_maintain is not None:
        for col in cols_to_maintain:
            new_col_data.append(df[col])
            new_col_names.append(col)

    return pd.DataFrame(np.array(new_col_data).T, columns=new_col_names).dropna()
