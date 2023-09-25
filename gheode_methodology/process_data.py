import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap

from gheode_methodology.plots import plot_clusterings, plot_clusterings_2, plot_season_functions

def load_xarrays(attr_names, filepaths):
    return {n: xr.load_dataset(path) for n, path in zip(attr_names, filepaths)}


def preprocess_data_for_clustering(data, should_normalize_clusterings=False):
    if should_normalize_clusterings:
        data_std = np.std(data, axis=0)
        data_std[np.where(data_std == 0)] = 1
        return (data - np.mean(data, axis=0)) / data_std

    return data


def get_region(xarrays, X_names, n_days, dims):
    X_values, X_grid = {}, {}
    for X_name in X_names:
        values = getattr(xarrays[X_name], X_name).values
        X_values[X_name] = np.reshape(values, (-1, n_days))
        X_grid[X_name] = np.reshape(values, (-1, *dims))

    return X_values, X_grid


def normalize_values(X_values, should_normalize_clusterings=False):
    return [preprocess_data_for_clustering(data, should_normalize_clusterings)
            for data in X_values.values()]


# Return the indices of each of the K centroids for each of the clusterings
def get_centroid_indices(clusterings, K):
    return [[[idx for idx, label in enumerate(clustering.labels_) if label == k]
             for k in range(K)]
            for clustering in clusterings]


# Return the index of the closest centroid, for each value in the original map
def get_centroids_all(X_values_n, clusterings, K, dims):
    get_closest = lambda target_cell_val: np.argmin(
        [np.linalg.norm(cell_val - target_cell_val) for cell_val in attr_data.T])

    centroids_all = []
    for i, (clustering, attr_data) in enumerate(zip(clusterings, X_values_n)):
        centroids = [get_closest(clustering.cluster_centers_[k]) for k in range(K)]
        centroids = np.unravel_index(centroids, dims)
        centroids_all.append(centroids)
    return np.array(centroids_all)


def get_X_cluster_values(centroids_all, centroids_indices, X_names, X_grid, X_values, K):
    attr_cluster_centroids = {}
    for name, centroids in zip(X_names, centroids_all):
        attr_cluster_centroids[name] = X_grid[name][:, centroids[0], centroids[1]]

    attr_cluster_means = {}
    for name, attr_idxs_by_cluster in zip(X_names, centroids_indices):
        attr_cluster_means[name] = np.array(
            [np.mean(X_values[name][:, attr_idxs_by_cluster[k]], axis=1) for k in range(K)]).T

    return attr_cluster_centroids, attr_cluster_means

def season_fn(n, m_i, sigma):
    return np.exp(- 1 / 2 * ((np.arange(0, n) - m_i) ** 2) / sigma)

def get_season_data():
    season_middles = [8, 21, 34, 47]
    weeks_in_year = 52
    years = 43 #TODO: make this dynamic

    # Create a vector with the index of the start of each year
    # (needs to take leap years into account, otherwise it would be just jumping 52 weeks ahead)
    N = X_values[attr_names[0]].shape[0]
    year_offsets = []
    leap_year_offset = 0

    for i in range(years):
        year_offsets.append(i * weeks_in_year + leap_year_offset)
        leap_year_offset += ((i + 1) % 4 == 0)
    year_offsets = np.array(year_offsets[:T // weeks_in_year])

    season_data = [np.mean([season_fn(N, season, 10)
                            for season in (year_offsets + season_middle)], axis=0) * 1000
                   for season_middle in season_middles]

    return np.array(season_data).T

def get_attribute_cols(X_cluster_centroids, X_cluster_means, X_id, X_flag):
    if X_flag == 0:
        return np.zeros((0, 0))
    elif X_flag == 1:
        return X_cluster_centroids[X_id]
    elif X_flag == 2:
        return X_cluster_means[X_id]
    else:
        raise ValueError("Attribute flag must be 0, 1 or 2")


def make_dataframe(xarrays, X_cluster_centroids, X_cluster_means,
                   X_names, y_name, season_data, lag_time):
    columns, data = [], []
    for name in X_names:
        for values, agg_mode in zip([X_cluster_centroids, X_cluster_means], ["cc", "cm"]):
            columns.append([f"{name}_{agg_mode}_{i}" for i, _ in enumerate(values[name][0])])
            data.append(values[name])

    columns = [i for k in columns for i in k]
    data = np.concatenate(data, axis=1)
    df = pd.DataFrame(data, columns=columns)

    if should_include_season_fns:
        df["season_spring"] = season_data[:, 0]
        df["season_summer"] = season_data[:, 1]
        df["season_fall"] = season_data[:, 2]
        df["season_winter"] = season_data[:, 3]

    y_col = f"target_{y_name}"
    df[y_col] = getattr(xarrays[y_name], y_name).to_numpy()
    df[y_col] = df[y_col] * 1000  # Convert to mm
    df[y_col] = df[y_col].shift(- lag_time)

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    K = 2
    lag_time = 1
    time_window_size = 8
    netherland_coords = [52.5, 4.5]
    center_coords = [0, 0]
    should_normalize_clusterings = False
    should_include_season_fns = True
    date_start, date_end = "1979-01-07", "2021-12-31"

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    version_name = "v3"

    attr_names = ["msl", "t2m", "sst", "r", "tp"]
    X_names = attr_names[:-1]
    y_name = attr_names[-1]

    filepaths = ["../data/msl_r-global_t-weekly.nc",
                 "../data/t2m_r-global_t-weekly.nc",
                 "../data/sst_r-global_t-weekly.nc",
                 "../data/rh_r-global_t-weekly.nc",
                 "../data/tp_r-netherlands_t-weekly-tp.nc"]

    # Loading xarrays
    xarrays = load_xarrays(attr_names, filepaths)

    # Loading metadata from xarrays
    _xarray = xarrays[attr_names[0]]
    resolution = (_xarray.lon.values.max() - _xarray.lon.values.min()) / (_xarray.lon.values.shape[0] - 1)
    lats = np.arange(_xarray.lat.values.min(), _xarray.lat.values.max() + resolution, resolution)
    lons = np.arange(_xarray.lon.values.min(), _xarray.lon.values.max() + resolution, resolution)
    dims = (getattr(_xarray, attr_names[0]).to_numpy().shape[1:])
    n_days = dims[0] * dims[1]

    print("resolution:", resolution)
    print("lats:", lats)
    print("lons:", lons)
    print("dims:", dims)
    print("n_days:", n_days)

    # Filtering time window
    for name in attr_names:
        xarrays[name] = xarrays[name].sel(time=slice(date_start, date_end))

    T = len(xarrays[attr_names[0]].time)
    print("The dataset has {} time steps".format(T))

    # Filter by region
    X_values, X_grid = get_region(xarrays, X_names, n_days, dims)

    # In cells with terrain, the SST is NaN. We replace it with 0
    if "sst" in X_names:
        X_values["sst"][np.where(np.isnan(X_values["sst"]) == True)] = 0

    # Add season functions
    season_data = get_season_data()
    # plot_season_functions(xarrays, y_name, season_data)

    # Run K-means clustering
    X_values_n = normalize_values(X_values, should_normalize_clusterings)
    clusterings = [KMeans(n_clusters=K, random_state=0).fit(data.T) for data in X_values_n]

    # Get centroids
    centroids_all = get_centroids_all(X_values_n, clusterings, K, dims)
    centroids_indices = get_centroid_indices(clusterings, K)

    # Get cluster values
    X_cluster_centroids, X_cluster_means = get_X_cluster_values(
        centroids_all, centroids_indices, X_names, X_grid, X_values, K)

    # Plot clusterings
    # plot_clusterings(attr_names, clusterings, centroids_all, lats, lons, dims)
    # plot_clusterings_2(attr_names, clusterings, centroids_all, X_cluster_means, lats, lons, dims)

    # Save the data
    df = make_dataframe(xarrays, X_cluster_centroids, X_cluster_means,
                        X_names, y_name, season_data, lag_time)
    filepath = f"../data/dataframe_lag-{lag_time}_K-{K}_{'cnorm' if should_normalize_clusterings else 'no-cnorm'}_{version_name}.csv"
    df.to_csv(filepath, index=False)
    print(f"Saved dataframe to {filepath}")