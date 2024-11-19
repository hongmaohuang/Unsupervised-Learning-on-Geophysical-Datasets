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

# 可以check interpolation of sampling rate 會不會差很多

raw_data = xr.open_dataset('../tomo.nc')
sta_Hong_path = '../stations.csv'
sta_Hong_data = pd.read_csv(sta_Hong_path, delimiter=',', header=None, skiprows=1)
sta_Hong_data.columns = ['sta', 'lon', 'lat', 'H']
# %%

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
ytickslabelll = [0, 1, 2, 3, 4]
prof_range_plot = [0.8, 0]
prof_range_for_plot = [0.75, 0]
ticks_color_abs = [1, 2, 3, 4, 5]
ticks_color_ptb = [-15, -5, -10, 0, 10, 5, 15]
ticks_cluster = np.arange(-0.5, -0.5 + cluster_number*1, 1)
vmin_abs, vmax_abs = 1, 5.1
vmin_mt, vmax_mt = 0, math.log10(10000)
vmin_ptb, vmax_ptb = -15, 16
cmap_style = 'jet_r'
interpo_value = 0.005

colors = cm.Set3.colors
deep_yellow = cm.Set3.colors[-1]
index_of_light_yellow = 1  # 這個index根據實際情況調整
colors_cluster_all = list(colors)
colors_cluster_all[index_of_light_yellow] = deep_yellow

colors_cluster = colors_cluster_all[0:int(cluster_number)]

depth = np.arange(prof_range_plot[1], prof_range_plot[0], 0.01)
points2d = np.empty([0, 4])

matplotlib.rcParams['font.family'] = 'Nimbus Sans'
matplotlib.rcParams['font.size'] = 25
# %%
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
    zi_vpt = griddata((points2d[:, 2], points2d[:, 3]), points_value_vpt, (xi_grid, yi_grid), method='cubic')
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
        ax.invert_xaxis()  # 讓緯度從右到左增加

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

