
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from geopy.distance import geodesic
import matplotlib
import string
import subprocess
import glob
import os 
from pykrige.ok import OrdinaryKriging

# General matplotlib settings
matplotlib.rcParams['font.family'] = 'Nimbus Sans'
matplotlib.rcParams['font.size'] = 20

# Profile definitions
prof_line = [
    [121.67416, 121.67416, 24.715182, 24.684155],
    [121.68900, 121.68900, 24.715182, 24.684155],
    [121.70672, 121.70672, 24.715182, 24.684155],
    [121.67302000000001, 121.710639, 24.7107, 24.7107],
    [121.67302000000001, 121.710639, 24.69773, 24.69773],
    [121.67302000000001, 121.710639, 24.685, 24.685]
]
uppercase_letters = string.ascii_uppercase
name_prof = [f"{letter}{letter}'" for letter in uppercase_letters]

# Adjustable parameters
file_path = '../cluster_results.csv'  # Path to input data file
tolerance = 0.01                     # Tolerance for filtering data near the profile line
depth_limit = 1                       # Depth range limit (e.g., 1 km)
grid_resolution = 10                  # Resolution of the interpolation grid
depth_ticks = np.array([-0.8, -0.6, -0.4, -0.2, 0.])  # Depth ticks (customizable)
Vp_range = (1, 5)                     # Range for Vp_ori colorbar
Vpt_range = (-15, 15)                 # Range for Vpt_ori colorbar
MT_range = (0, 4)                   # Range for MT_ori colorbar
cluster_colors = cm.Set3.colors       # Color map for clusters
font_size = 25                        # Font size for labels and ticks
output_dir = '../Fig/'                # Output directory for saving figures
color_labelpad = 20
interval_interpol_hori = 0.002 # degree
interval_interpol_vertical = 0.002  # km

# Load data
data = pd.read_csv(file_path)

# Extract necessary fields
longitude = data['XX'].values
latitude = data['YY'].values
depth = data['ZZ'].values

# Fields to be interpolated
fields_to_interpolate = ['Vp', 'Ohm', 'Ohm_Vp_ratio', 'Clusters', 'Vp_ori', 'MT_ori', 'Vpt_ori']

new_longitude = np.arange(121.673, 121.720, interval_interpol_hori)
new_latitude = np.arange(24.684, 24.716, interval_interpol_hori)
new_depth = np.arange(0, 0.8, interval_interpol_vertical)

new_lon, new_lat, new_dep = np.meshgrid(new_longitude, new_latitude, new_depth, indexing='ij')

interpolated_data = {
    'XX': new_lon.ravel(),
    'YY': new_lat.ravel(),
    'ZZ': new_dep.ravel()
}

points = np.array([longitude, latitude, depth]).T
grid = np.array([new_lon.ravel(), new_lat.ravel(), new_dep.ravel()]).T

for field in fields_to_interpolate:
    values = data[field].values
    interpolated_values = griddata(points, values, grid, method='linear')
    if field == 'Clusters':
        interpolated_values = np.rint(interpolated_values).astype(int)
    interpolated_data[field] = interpolated_values.ravel()

interpolated_df = pd.DataFrame(interpolated_data)

output_path = 'interpolated_3d_model_with_XX_YY_ZZ.csv'
interpolated_df.to_csv(output_path, index=False)

data = interpolated_df
# %%
# Adjust cluster colors
deep_yellow = cluster_colors[-1]

index_of_light_yellow = 1
colors_cluster_all = list(cluster_colors)
colors_cluster_all[index_of_light_yellow] = deep_yellow

cluster_number = len(data.Clusters.unique())
colors_cluster = colors_cluster_all[:cluster_number]

# Function to calculate profile distances
def calculate_distance_km(x, y, start, end):
    start_point = (start[1], start[0])
    point = (y, x)
    return geodesic(start_point, point).km

# Loop through each profile line
for index, profile in enumerate(prof_line):
    print(f"Processing profile {index + 1}")
    
    start_point = [profile[0], profile[2]]  # [longitude, latitude]
    end_point = [profile[1], profile[3]]    # [longitude, latitude]
    profile_name = name_prof[index]
    
    # Calculate profile distances
    data['Distance_km'] = data.apply(lambda row: calculate_distance_km(row['XX'], row['YY'], start_point, end_point), axis=1)
    
    # Filter data points near the profile
    line_vector = np.array([end_point[0] - start_point[0], end_point[1] - start_point[1]])
    line_norm = np.linalg.norm(line_vector)
    longitudes = data['XX']
    latitudes = data['YY']
    
    perpendicular_distance = abs((longitudes - start_point[0]) * line_vector[1] -
                                  (latitudes - start_point[1]) * line_vector[0]) / line_norm
    filtered_data = data[perpendicular_distance <= tolerance]
    
    # Prepare interpolation grid
    points_km = filtered_data[['Distance_km', 'ZZ']].values
    grid_x_km, grid_y_km = np.meshgrid(
        np.linspace(0, geodesic((start_point[1], start_point[0]), (end_point[1], end_point[0])).km, grid_resolution),
        np.linspace(filtered_data['ZZ'].min(), filtered_data['ZZ'].max(), grid_resolution)
    )
    params = ['Vp_ori', 'Vpt_ori', 'MT_ori', 'Clusters']
    fig, axes = plt.subplots(4, 1, figsize=(20, 15), sharex=True)
    axes = axes.flatten()

    for i, param in enumerate(params):
        print(param)
        if param != 'Clusters':
            values = filtered_data[param].values
            # Griddata interpolation
            grid_z = griddata((points_km[:, 0], points_km[:, 1]), values, (grid_x_km, grid_y_km), method='nearest')

            if param == 'Vp_ori':
                contourf = axes[i].contourf(grid_x_km, -grid_y_km, grid_z, cmap='jet_r', levels=np.linspace(*Vp_range, 100), extend='both')
                cbar = plt.colorbar(contourf, ax=axes[i], orientation='vertical', pad=0.02)
                cbar.set_ticks(np.arange(Vp_range[0], Vp_range[1] + 1, 1))
                cbar.set_label('Vp (m/s)', labelpad=color_labelpad)

            elif param == 'Vpt_ori':
                contourf = axes[i].contourf(grid_x_km, -grid_y_km, grid_z, cmap='jet_r', levels=np.linspace(*Vpt_range, 100), extend='both')
                cbar = plt.colorbar(contourf, ax=axes[i], orientation='vertical', pad=0.02)
                cbar.set_ticks(np.arange(Vpt_range[0], Vpt_range[1] + 1, 5))
                cbar.set_label('dVp (%)', labelpad=color_labelpad)

            elif param == 'MT_ori':
                contourf = axes[i].contourf(grid_x_km, -grid_y_km, grid_z, cmap='jet_r', levels=np.linspace(*MT_range, 100), extend='both')
                cbar = plt.colorbar(contourf, ax=axes[i], orientation='vertical', pad=0.02)
                cbar.set_ticks(np.arange(MT_range[0], MT_range[1] + 0.1, 1))
                cbar.set_label('Log Resistivity (Ωm)', labelpad=color_labelpad)

        elif param == 'Clusters':
            cmap_cluster = mcolors.ListedColormap(colors_cluster)
            bounds = np.arange(-1, cluster_number, 1)
            ticks_cluster = np.arange(-0.5, cluster_number - 0.5, 1)
            grid_z_ori = griddata(points_km, filtered_data[param].values, (grid_x_km, grid_y_km), method='nearest')
            contourf_clus = axes[i].contourf(grid_x_km, -grid_y_km, grid_z_ori, levels=bounds, cmap=cmap_cluster)
            cbar = plt.colorbar(contourf_clus, ticks=ticks_cluster, ax=axes[i], orientation='vertical', pad=0.02)
            cbar.ax.set_yticklabels(np.array([chr(k) for k in range(ord('A'), ord('Z') + 1)])[:cluster_number])
            cbar.set_label('Clusters', labelpad=color_labelpad)
            cbar.ax.tick_params(size=0)

        yticks = depth_ticks
        axes[i].set_yticks(yticks)
        axes[i].set_yticklabels([f"{abs(y):.1f}" for y in yticks])
        axes[i].set_ylim([-0.75, 0])

    fig.text(0.45, 0.05, 'Distance (km)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Depth (km)', ha='center', va='center', rotation='vertical')
    plt.subplots_adjust(hspace=0.1)
    fig.savefig(f"{output_dir}{cluster_number}_{profile_name}.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

pattern = f'../Fig/{cluster_number}_*.png'
images = sorted(glob.glob(pattern))
args = ["montage", "-geometry", "+0+0", "-tile", "3x2"] + images + ["../Fig/profiles.png"]

subprocess.run(args)

files = glob.glob(f'../Fig/{cluster_number}_*.png')
for file in files:
    os.remove(file)
    print(f"Removed: {file}")

'''
# %%


import pygmt
import numpy as np
import xarray as xr
from scipy.interpolate import interpn, griddata
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
import string
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import subprocess
import glob
import os 
from matplotlib.patheffects import withStroke

clusters_resultss = pd.read_csv('../cluster_results.csv')

raw_data = xr.open_dataset('../tomo.nc')
sta_Hong_path = '../stations.csv'
sta_Hong_data = pd.read_csv(sta_Hong_path, delimiter=',', header=None, skiprows=1)
sta_Hong_data.columns = ['sta', 'lon', 'lat', 'H']

output = '../Fig/'

prof_line = [[121.67416, 121.67416, 24.715182, 24.684155],
             [121.68900, 121.68900, 24.715182, 24.684155],
             [121.70672, 121.70672, 24.715182, 24.684155],
             [121.67302000000001, 121.710639 , 24.7107, 24.7107],
             [121.67302000000001, 121.710639 , 24.69773, 24.69773],
             [121.67302000000001, 121.710639 , 24.685, 24.685],
             ]

uppercase_letters = string.ascii_uppercase
name_prof = [f"{letter}{letter}'" for letter in uppercase_letters]

#  General Setting
cluster_method = 'GMM'
cluster_number = raw_data.clusters.data.max() + 1
ytickslabelll = [1, 2, 3]
prof_range_plot = [0.8, 0]
prof_range_for_plot = [0.75, 0]
ticks_color_abs = [1, 2, 3, 4, 5]
ticks_color_ptb = [-15, -5, -10, 0, 10, 5, 15]
ticks_cluster = np.arange(-0.5, -0.5 + cluster_number*1, 1)
vmin_abs, vmax_abs = 1, 5.1
vmin_mt, vmax_mt = 1, math.log10(1000)
vmin_ptb, vmax_ptb = -15, 16
cmap_style = 'jet_r'
interpo_value = 0.005
# %%
colors = cm.Set3.colors
deep_yellow = cm.Set3.colors[-1]
index_of_light_yellow = 1  # this depends on your using scenario
colors_cluster_all = list(colors)
colors_cluster_all[index_of_light_yellow] = deep_yellow

colors_cluster = colors_cluster_all[0:int(cluster_number)]

depth = np.arange(prof_range_plot[1], prof_range_plot[0], 0.01)
points2d = np.empty([0, 4])

matplotlib.rcParams['font.family'] = 'Nimbus Sans'
matplotlib.rcParams['font.size'] = 25

for i in range(len(prof_line)):
    print(' =============================== ')
    print('Processing: ' + name_prof[i])
    points = pygmt.project(center='{}/{}'.format(prof_line[i][0], prof_line[i][2]),
                        endpoint='{}/{}'.format(prof_line[i][1], prof_line[i][3]),
                        generate = interpo_value, unit=True)
    points2d = np.empty((0, 4))
    for x in points.values:
        for d in depth:
            points2d = np.vstack((points2d, np.append(x, d)))

    xi = np.linspace(np.array(points.p)[0], np.array(points.p)[-1], int((np.array(points.p)[-1]-np.array(points.p)[0])/interpo_value))
    yi = np.arange(depth[0], depth[-1], interpo_value)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    points_value_vp = interpn((raw_data.depth.values, raw_data.lat.values, raw_data.lon.values),
                           raw_data.vp.values, points2d[:, [3, 1, 0]])
    zi_vp = griddata((points2d[:, 2], points2d[:, 3]), points_value_vp, (xi_grid, yi_grid), method='cubic')
    df = pd.DataFrame(zi_vp)
    zi_vppp = df.interpolate()
    zi_vppp_ar = zi_vppp.to_numpy()
    zi_vp = zi_vppp_ar
    
    points_value_vpt = interpn((raw_data.depth.values, raw_data.lat.values, raw_data.lon.values),
                           raw_data.vpt.values, points2d[:, [3, 1, 0]])
    zi_vpt = griddata((points2d[:, 2], points2d[:, 3]), points_value_vpt, (xi_grid, yi_grid), method='nearest')
    df = pd.DataFrame(zi_vpt)
    zi_vpttt = df.interpolate()
    zi_vpttt_ar = zi_vpttt.to_numpy()
    zi_vpt = zi_vpttt_ar


    points_value_mt = interpn((raw_data.depth.values, raw_data.lat.values, raw_data.lon.values),
                           raw_data.mt.values, points2d[:, [3, 1, 0]])
    zi_mt = griddata((points2d[:, 2], points2d[:, 3]), points_value_mt, (xi_grid, yi_grid), method='cubic')
    df = pd.DataFrame(zi_mt)
    df_mtt = df.interpolate()
    df_mtt_ar = df_mtt.to_numpy()
    zi_mt = df_mtt_ar
    #print(zi_mt)
    #zi_mt[zi_mt <= 0] = 0.01

    points_value_cluster = interpn((raw_data.depth.values, raw_data.lat.values, raw_data.lon.values),
                            raw_data.clusters.data, points2d[:, [3, 1, 0]])
    zi_cluster = griddata((points2d[:, 2], points2d[:, 3]), points_value_cluster, (xi_grid, yi_grid), method='cubic')
    zi_cluster_int = np.round(zi_cluster).astype(int)
    df = pd.DataFrame(zi_cluster)
    df_clisterrrr = df.interpolate()
    df_clisterrrr_ar = df_clisterrrr.to_numpy()
    df_clisterrrr_ar_int = np.round(df_clisterrrr_ar).astype(int)
    zi_cluster_int = df_clisterrrr_ar_int


    fig, axs = plt.subplots(4, 1, figsize=(20, 15), sharex=True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.tight_layout()
    ax1, ax2, ax3, ax4 = axs

    # VP
    print('Plot: Vp')
    contourf_vp = ax1.contourf(xi, yi, zi_vp, np.arange(vmin_abs, vmax_abs, 0.03), cmap = cmap_style, extend='both')
    cbar1 = plt.colorbar(contourf_vp, ticks=ticks_color_abs, ax = ax1, label=' Vp (km/s)', location='right', pad = 0.02)
    C = ax1.contour(xi, yi, zi_vp, 10, colors='w', linestyles='dotted')
    ax1.clabel(C, inline=1, fontsize=15)
    ax1.set_ylim(prof_range_for_plot)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    cbar1.set_label('Vp (km/s)', labelpad=10)
    cbar1.ax.yaxis.set_label_position('right')
    cbar1.ax.yaxis.set_label_coords(7, 0.5)

    # dVP
    print('Plot: dVp')
    contourf_dvp = ax2.contourf(xi, yi, zi_vpt, np.arange(vmin_ptb, vmax_ptb), cmap = cmap_style, extend='both')
    cbar2 = plt.colorbar(contourf_dvp, ticks=ticks_color_ptb, ax = ax2, label='dVp (%)', location='right', pad = 0.02)
    C = ax2.contour(xi, yi, zi_vpt, 5, colors='w', linestyles='dotted')
    ax2.clabel(C, inline=10, fontsize=15)
    ax2.set_ylim(prof_range_for_plot)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    cbar2.set_label('dVp (%)', labelpad=10)
    cbar2.ax.yaxis.set_label_position('right')
    cbar2.ax.yaxis.set_label_coords(7, 0.5)

    # MT
    print('Plot: MT')
    #print(np.arange(vmin_mt, vmax_mt, 5))
    contourf_mt = ax3.contourf(xi, yi, zi_mt, np.arange(vmin_mt, vmax_mt, 0.002), cmap = cmap_style, extend='both')
    cbar3 = plt.colorbar(contourf_mt, ticks=ytickslabelll, ax=ax3, label='Log resistivity (Ωm)', location='right', pad = 0.02)
    cbar3.ax.set_yticklabels(ytickslabelll) 
    C = ax3.contour(xi, yi, zi_mt, 2, colors='w', linestyles='dotted')
    plt.clabel(C, inline=1, fontsize=15)
    ax3.set_ylim(prof_range_for_plot)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    cbar3.set_label('Log resistivity (Ωm)', labelpad=10)
    cbar3.ax.yaxis.set_label_position('right')
    cbar3.ax.yaxis.set_label_coords(7, 0.5)

    # Clusters
    print('Plot: Clusters')
    cmap_cluster = mcolors.ListedColormap(colors_cluster)
    #print(cmap_cluster)
    contourf_clus = ax4.contourf(xi, yi, zi_cluster_int, np.arange(-1, cluster_number, 1), cmap=cmap_cluster)
    ax4.set_ylim(prof_range_for_plot)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    cbar4 = plt.colorbar(contourf_clus, ticks=ticks_cluster, ax = ax4, label='Clusters', orientation='vertical', location='right', pad = 0.02)
    cbar4.ax.set_yticklabels(np.array([chr(i) for i in range(ord('A'), ord('Z')+1)])[0:int(cluster_number)]) 
    cbar4.ax.tick_params(size=0)
    cbar4.set_label('Clusters', labelpad=10)
    cbar4.ax.yaxis.set_label_position('right')
    cbar4.ax.yaxis.set_label_coords(7, 0.5)
    #ax4.set_xlabel('Distance (km)')
    #ax4.set_ylabel('Depth (km)')

    for ax in (ax1, ax2, ax3, ax4):
        major_ticks = np.arange(0, 0.85, 0.25)
        minor_ticks = np.arange(0, 0.85, 0.125)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_xticks(np.arange(np.array(points.p)[0], np.array(points.p)[-1], 0.5), minor=True)
        ax.set_yticklabels([f'{d:.2f}' for d in major_ticks])
        ax.tick_params(which='both', color='black', width=1.5)
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)
        ax.invert_xaxis()  # Make the Lat to be left to right

    plt.suptitle(name_prof[i], y=1, x=0.45)
    # Set common labelsxw
    fig.text(0.45, -0.02, 'Distance (km)', ha='center', va='center')
    fig.text(-0.03, 0.5, 'Depth (km)', ha='center', va='center', rotation='vertical')
    plt.subplots_adjust(hspace=0.1)
    fig.savefig(output  + cluster_method + '_' + str(cluster_number) + '_' + name_prof[i] + '.png', dpi=300, bbox_inches='tight', transparent=True)

pattern = f'../Fig/{cluster_method}_{cluster_number}_*.png'
images = sorted(glob.glob(pattern))
args = ["montage", "-geometry", "+0+0", "-tile", "3x2"] + images + ["../Fig/profiles.png"]

subprocess.run(args)

files = glob.glob('../Fig/' + cluster_method + '_*.png')
for file in files:
    os.remove(file)
    print(f"Removed: {file}")


# %%


interp_depth = 0.8
geophysics_data = 'Vpt_ori'

# 篩選深度為 0.4 的數據
filtered_data = clusters_resultss[clusters_resultss['ZZ'] == interp_depth]

# 提取經緯度和 Vp 數據
x = filtered_data['XX']
y = filtered_data['YY']
z = filtered_data[geophysics_data]

region = [x.min(), x.max(), y.min(), y.max()]

# 將數據保存為網格文件
grid_file = "temp_grid.nc"
pygmt.xyz2grd(
    data=pd.DataFrame({"x": x, "y": y, "z": z}),
    region=region,
    spacing=(0.002, 0.002),  # 設定網格間距
    outgrid=grid_file,
)
with pygmt.config(FORMAT_GEO_MAP = 'D', FORMAT_FLOAT_OUT = '%.3f'):
    fig = pygmt.Figure()
    pygmt.makecpt(cmap='jet', background = 'o',series=[vmin_ptb, vmax_ptb], reverse=True)  # 自訂顏色映射表

    fig.grdimage(
        grid=grid_file,
        cmap=True,
        region=region,
        projection="M15c",
        frame=["a"],       
    )
    fig.grdcontour(
        region=region,
        projection="M15c",
        frame=['a'],
        pen="0.5p,white",
        grid = grid_file,
        interval=5,
        annotation=5,
    )




    # 添加標記文字
    fig.text(x=region[1]-0.01, y=region[2]+0.004, text=str(interp_depth)+' km', font='30p,Helvetica-Bold,black')

    # 顯示圖表
    fig.show()

    # 清理臨時文件
    import os
    os.remove(grid_file)