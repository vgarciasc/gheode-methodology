import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def plot_clusterings(attr_names, clusterings, centroids_all, lats, lons, dims):
    # coords = netherlands_coords
    # coord_slice = (np.tile(coords, 2) + np.tile([24, 24], 2) * np.array([-1, -1, 1, 1])).reshape([2, 2])
    coord_slice = np.array([[lats[0], lons[0]], [lats[-1], lons[-1]]])

    fig, axs = plt.subplots(2, 2, figsize=(18, 8))

    for i, (name, clustering, centroids) in enumerate(zip(attr_names, clusterings, centroids_all)):
        ax = axs[i // 2, i % 2]

        m = Basemap(
            ax=ax,
            resolution='l',
            llcrnrlat=coord_slice[0, 0], llcrnrlon=coord_slice[0, 1],
            urcrnrlat=coord_slice[1, 0], urcrnrlon=coord_slice[1, 1])

        # Draw coast
        coast = m.drawcoastlines(color='#00000033')

        # Draw clusterization
        x = np.reshape(clustering.labels_, dims)
        img = m.imshow(x, cmap='tab20')

        # Draw centroids. 'j'-th centroid has the same color as the 'j'-th cluster
        lats_c = [lats[idx] for idx in centroids[0]]
        lons_c = [lons[idx] for idx in centroids[1]]
        cx, cy = m(lons_c, lats_c)
        # m.scatter(cx, cy, marker='o', c=range(10), cmap='tab20', s=100, edgecolor='white')

        # On top of each of the j-th centroids, draw 'j'
        for j, (x, y) in enumerate(zip(cx, cy)):
            ax.text(x, y, str(j), fontsize=12, color="white", horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

        ax.set_title(name)

    plt.show()


def plot_clusterings_2(attr_names, clusterings, centroids_all, X_cluster_means, lats, lons, dims):
    # coords = netherlands_coords
    # coord_slice = (np.tile(coords, 2) + np.tile([24, 24], 2) * np.array([-1, -1, 1, 1])).reshape([2, 2])
    coord_slice = np.array([[lats[0], lons[0]], [lats[-1], lons[-1]]])

    fig, axs = plt.subplots(2, 2, figsize=(18, 8))

    for i, (name, clustering, centroids) in enumerate(zip(attr_names, clusterings, centroids_all)):
        ax = axs[i // 2, i % 2]
        vals = X_cluster_means[name]

        m = Basemap(
            ax=ax,
            resolution='l',
            llcrnrlat=coord_slice[0, 0], llcrnrlon=coord_slice[0, 1],
            urcrnrlat=coord_slice[1, 0], urcrnrlon=coord_slice[1, 1]
        )

        # Draw coast
        coast = m.drawcoastlines(color='#00000033')

        # Draw clusterization
        x = np.reshape(np.mean(vals, axis=0)[clustering.labels_], dims)
        img = m.imshow(x, cmap='viridis')

        # Draw centroids. 'j'-th centroid has the same color as the 'j'-th cluster
        lats_c = [lats[idx] for idx in centroids[0]]
        lons_c = [lons[idx] for idx in centroids[1]]
        cx, cy = m(lons_c, lats_c)
        # m.scatter(cx, cy, marker='o', c=range(10), cmap='tab20', s=100, edgecolor='white')

        # On top of each of the j-th centroids, draw 'j'
        for j, (x, y) in enumerate(zip(cx, cy)):
            ax.text(x, y, str(j), fontsize=12, color="white", horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

        plt.colorbar(img, ax=ax, orientation="vertical")
        ax.set_title(name)

    plt.show()


def plot_season_functions(xarrays, y_name, season_data):
    plt.figure(figsize=(20, 5))
    plt.plot(xarrays[y_name].time, getattr(xarrays[y_name], y_name).to_numpy() * 1000, color="blue")
    for season, x_i in zip(["spring", "summer", "fall", "winter"], season_data.T):
        plt.plot(xarrays[y_name].time, x_i, label=season)

    plt.xlim(datetime.datetime(1997, 1, 1), datetime.datetime(2000, 12, 1))
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Precipitation (mm)")
    plt.show()
