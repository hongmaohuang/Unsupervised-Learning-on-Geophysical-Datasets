# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm
import string
from matplotlib.colors import LogNorm

# ================================
# 1. Data Loading
# ================================
# Load clustering results
clusters_results = pd.read_csv('../cluster_results.csv')

# Specify the cluster number to plot
Cluster_number_forplot = 0

# Load NetCDF data
file_path = "../tomo.nc"
data = xr.open_dataset(file_path)

# ================================
# 2. Configuration
# ================================
# Configure cluster colors
cluster_numbers = clusters_results.Clusters.max() + 1
colors = cm.Set3.colors
deep_yellow = cm.Set3.colors[-1]
index_of_light_yellow = 1  # Adjust based on your scenario
colors_cluster_all = list(colors)
colors_cluster_all[index_of_light_yellow] = deep_yellow
colors_cluster = colors_cluster_all[:cluster_numbers]

# Extract related data
datasets = clusters_results[clusters_results.Clusters == Cluster_number_forplot]
Lon = clusters_results.XX
Lat = clusters_results.YY
Depth = clusters_results.ZZ
clusters = data['clusters']
lon = data['lon'].values
lat = data['lat'].values
depth = data['depth'].values
vpt = data['vpt'].values  
mt = data['mt'].values  

# Set font and size
plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['font.size'] = 15

# Select depth indices
depth_indices = np.arange(0, len(depth), 10)
selected_clusters = clusters.isel(depth=depth_indices).values
selected_vpt = vpt[depth_indices, :, :]  # Match selected depths
selected_mt = mt[depth_indices, :, :]  # Match selected depths

selected_depth = depth[depth_indices]

# Color map for Vp
vpt_colormap = cm.jet_r
mt_colormap = cm.jet_r
uppercase_letters = string.ascii_uppercase

# ================================
# 3. Plotting a Single Cluster
# ================================
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot surfaces for each depth layer (only for the specified cluster)
for i, d in enumerate(selected_depth):
    X, Y = np.meshgrid(lon, lat)
    Z = np.full_like(X, d)  # Depth as Z-axis values
    cluster_layer = selected_clusters[i, :, :]

    # Filter data for the specified cluster
    mask = (cluster_layer == Cluster_number_forplot)
    if mask.any():  # Plot only if there is relevant data
        X_masked = np.ma.masked_where(~mask, X)
        Y_masked = np.ma.masked_where(~mask, Y)
        Z_masked = np.ma.masked_where(~mask, Z)

        # Plot the surface
        ax.plot_surface(
            X_masked, Y_masked, Z_masked, 
            color=colors_cluster[Cluster_number_forplot], alpha=0.5
        )

# Invert the Z-axis direction (suitable for geological data)
ax.invert_zaxis()
plt.title(f'Cluster {uppercase_letters[Cluster_number_forplot]}')
# Adjust view angle
ax.view_init(elev=20, azim=-120)  # Custom view angle
ax.set_xticks([121.65, 121.67, 121.69, 121.71, 121.73])
ax.set_yticks([24.68, 24.70, 24.72])
ax.set_zticks([0.2, 0.4, 0.6, 0.8])
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))
#plt.savefig('Test.png')
plt.show()


# ================================
# 4. Plotting a Single Cluster on Vpt
# ================================
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot surfaces for each depth layer (only for the specified cluster)
for i, d in enumerate(selected_depth):
    X, Y = np.meshgrid(lon, lat)
    Z = np.full_like(X, d)  # Depth as Z-axis values
    cluster_layer = selected_clusters[i, :, :]
    vpt_layer = selected_vpt[i, :, :]
    # Filter data for the specified cluster
    mask = (cluster_layer == Cluster_number_forplot)
    if mask.any():  # Plot only if there is relevant data
        X_masked = np.ma.masked_where(~mask, X)
        Y_masked = np.ma.masked_where(~mask, Y)
        Z_masked = np.ma.masked_where(~mask, Z)
        vpt_masked = vpt_layer[mask]

        # Normalize Vpt values for coloring
        norm = plt.Normalize(vmin=-15, vmax=15)
        colors = vpt_colormap(norm(vpt_masked))

        # Plot points with Vpt colors
        ax.scatter(
            X_masked, Y_masked, Z_masked, 
            c=colors, marker='o', s=10, alpha=0.8, label=f"Depth {d} km"
        )

# Invert the Z-axis direction (suitable for geological data)
ax.invert_zaxis()
plt.title(f'Cluster {uppercase_letters[Cluster_number_forplot]} on dVp')
# Adjust view angle
ax.view_init(elev=20, azim=-120)  # Custom view angle
ax.set_xticks([121.65, 121.67, 121.69, 121.71, 121.73])
ax.set_yticks([24.68, 24.70, 24.72])
ax.set_zticks([0.2, 0.4, 0.6, 0.8])
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))
#plt.savefig('Test.png')
plt.show()


# ================================
# 5. Plotting a Single Cluster on MT
# ================================
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot surfaces for each depth layer (only for the specified cluster)
for i, d in enumerate(selected_depth):
    X, Y = np.meshgrid(lon, lat)
    Z = np.full_like(X, d)  # Depth as Z-axis values
    cluster_layer = selected_clusters[i, :, :]
    mt_layer = selected_mt[i, :, :]
    
    # Filter data for the specified cluster
    mask = (cluster_layer == Cluster_number_forplot)
    if mask.any():  # Plot only if there is relevant data
        X_masked = X[mask]
        Y_masked = Y[mask]
        Z_masked = Z[mask]
        mt_masked = mt_layer[mask]

        # Normalize mt values for coloring with LogNorm
        norm = LogNorm(vmin=0.1, vmax=3)  # Set the appropriate range for your data
        colors = mt_colormap(norm(mt_masked))

        # Plot points with mt colors
        ax.scatter(
            X_masked, Y_masked, Z_masked, 
            c=colors, marker='o', s=10, alpha=0.8, label=f"Depth {d} km"
        )

        
# Invert the Z-axis direction (suitable for geological data)
ax.invert_zaxis()
plt.title(f'Cluster {uppercase_letters[Cluster_number_forplot]} on Resistivity (Log Scale)')
# Adjust view angle
ax.view_init(elev=20, azim=-120)  # Custom view angle
ax.set_xticks([121.65, 121.67, 121.69, 121.71, 121.73])
ax.set_yticks([24.68, 24.70, 24.72])
ax.set_zticks([0.2, 0.4, 0.6, 0.8])
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))

# Show the plot
plt.show()


# %%

# ================================
# 4. Plotting All Clusters
# ================================
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot surfaces for all depth layers, iterate over all clusters
for cluster_id in range(cluster_numbers):
    for i, d in enumerate(selected_depth):
        X, Y = np.meshgrid(lon, lat)
        Z = np.full_like(X, d)  # Depth as Z-axis values
        cluster_layer = selected_clusters[i, :, :]

        # Filter data for the selected cluster
        mask = (cluster_layer == cluster_id)
        if mask.any():  # Plot only if there is relevant data
            X_masked = np.ma.masked_where(~mask, X)
            Y_masked = np.ma.masked_where(~mask, Y)
            Z_masked = np.ma.masked_where(~mask, Z)

            # Plot the surface with custom color
            ax.plot_surface(
                X_masked, Y_masked, Z_masked, 
                color=colors_cluster[cluster_id], alpha=0.8
            )

# Invert the Z-axis direction (suitable for geological data)
ax.invert_zaxis()

# Adjust view angle
ax.view_init(elev=20, azim=-120)  # Custom view angle
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))
ax.set_xticks([121.65, 121.67, 121.69, 121.71, 121.73])
ax.set_yticks([24.68, 24.70, 24.72])
ax.set_zticks([0.2, 0.4, 0.6, 0.8])
# Display the plot
plt.show()
# %%