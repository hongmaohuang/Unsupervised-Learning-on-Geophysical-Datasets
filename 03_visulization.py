''' 
# %%
import pygmt
import numpy as np
import xarray as xr
import pandas as pd
import math
from matplotlib import pyplot as plt, cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import subprocess
import glob
import os
import geopandas as gpd
import string 
import matplotlib
from scipy.interpolate import interpn, griddata
import matplotlib.colors as mcolors

clusters_resultss = pd.read_csv('../cluster_results.csv')
raw_data = xr.open_dataset('../tomo.nc')
sta_Hong_path = '../stations.csv'
sta_Hong_data = pd.read_csv(sta_Hong_path, delimiter=',', header=None, skiprows=1)
sta_Hong_data.columns = ['sta', 'lon', 'lat', 'H']
well_data = pd.read_csv('../well_all_loc_hong.csv', sep=',')
LYR = gpd.read_file('/home/hmhuang/Research/Hongchailin/clustering_vp_mt/lanyang_poly/lanyang_poly.shp')
output = '../Fig/'

prof_line = [[121.67416, 121.67416, 24.715182, 24.68],
             [121.68900, 121.68900, 24.715182, 24.68],
             [121.70672, 121.70672, 24.715182, 24.68],
             [121.67, 121.717 , 24.7107, 24.7107],
             [121.67, 121.717 , 24.69773, 24.69773],
             [121.67, 121.717 , 24.685, 24.685],
             ]
all_depth = [0.2, 0.4, 0.6, 0.8]
all_geophysics_data = ['Vpt_ori', 'Vp_ori', 'MT_ori']

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
vmin_abs, vmax_abs = 1, 5
vmin_mt, vmax_mt = 1, math.log10(1000)
vmin_ptb, vmax_ptb = -15, 15
cmap_style = 'jet_r'
interpo_value = 0.005
ckb ='n'

colors = cm.Set3.colors
deep_yellow = cm.Set3.colors[-1]
index_of_light_yellow = 1  # this depends on your using scenario
colors_cluster_all = list(colors)
colors_cluster_all[index_of_light_yellow] = deep_yellow

colors_cluster = colors_cluster_all[0:int(cluster_number)]

depth = np.arange(prof_range_plot[1], prof_range_plot[0], 0.01)
points2d = np.empty([0, 4])

# %%
# Cross-Sections Plot #
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


# Mapview Plot #

for index_Gdata in all_geophysics_data:

    geophysics_data = index_Gdata

    for index_depth in all_depth:
        print(f'depth plot now: {index_depth}')
        interp_depth = index_depth

        # 篩選深度
        filtered_data = clusters_resultss[clusters_resultss['ZZ'] == interp_depth]

        # 提取經緯度和數據
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

            if geophysics_data == 'Vpt_ori':
                pygmt.makecpt(cmap='jet', background = 'o',series=[vmin_ptb, vmax_ptb], reverse=True) 
                
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
                if ckb == 'y':
                    fig.colorbar(frame=["a", "x+ldVp (%)"], position="JBC+w15c/0.5c+e")
                if interp_depth == 0.8:
                    map_width_cm = 15
                    lon_min, lon_max = region[0], region[1]  
                    lat_min, lat_max = region[2], region[3]  
                    map_lon_range_cm = map_width_cm
                    map_lat_range_cm = map_width_cm * (lat_max - lat_min) / (lon_max - lon_min)
                    lon_diff_degree = 1  
                    lat_diff_degree = 1  
                    lon_diff_cm = lon_diff_degree / (lon_max - lon_min) * map_lon_range_cm
                    lat_diff_cm = lat_diff_degree / (lat_max - lat_min) * map_lat_range_cm

                    fig.plot(x=121.698, y=24.7041, style="l13p+tLanyang River+fHelvetica-BoldOblique,white", pen="0.3p", fill="black", transparency=70)
                    fig.text(text="W1", x=HCL1[0], y=HCL1[1]+0.0015, font="10p,31,black", transparency=80, fill="white")
                    fig.text(text="W2", x=HCL2[0], y=HCL2[1]+0.0015, font="10p,31,black", transparency=80, fill="white")
                    fig.text(text="CTCN", x=CTCN[0], y=CTCN[1]+0.0015, font="10p,31,black", transparency=80, fill="white")

                    fig.plot(x=sta_Hong_data.lon, y=sta_Hong_data.lat, style='t0.3', fill='#90A4AE', region=region, label = 'Stations', pen="0.3p,black", transparency=70)
                    fig.plot(data=LYR, color="#C7C8CC", transparency=65)
                    for i in range(len(prof_line)):
                        name_prof_start = name_prof[i][0]
                        name_prof_end = name_prof[i][1:3]
                        points = pygmt.project(center='{}/{}'.format(prof_line[i][0], prof_line[i][2]),
                                            endpoint='{}/{}'.format(prof_line[i][1], prof_line[i][3]),
                                            generate =0.002, unit=True)
                        if prof_line[i][2]==prof_line[i][3] :
                            fig.text(text=name_prof_start, x = points.r.min()-0.0025, y = points.s.max(), font="15p,8,black", transparency=20)
                            fig.text(text=name_prof_end, x = points.r.max()+0.004, y = points.s.min(), font="15p,8,black", transparency=20)
                            length = [(prof_line[i][1] - prof_line[i][0])*lon_diff_cm]
                            angle = [0]
                            #print(length, angle)
                            fig.plot(x = prof_line[i][0], y = prof_line[i][2], style="v0.2c+bt+et+a80", direction=(angle, length), pen = "0.4p" )

                        else:
                            fig.text(text=name_prof_start, x = points.r.max(), y = points.s.max()+0.0025, font="15p,8,black", transparency=20)
                            fig.text(text=name_prof_end, x = points.r.min(), y = points.s.min()+0.001, font="15p,8,black", transparency=20)
                            length = [(points.s.max() - points.s.min())*lat_diff_cm]
                            angle = [270]
                            fig.plot(x = prof_line[i][0], y = prof_line[i][2], style="v0.2c+bt+et+a80", direction=(angle, length), pen = "0.4p")

            if geophysics_data == 'Vp_ori':

                pygmt.makecpt(cmap='jet', background = 'o',series=[vmin_abs, vmax_abs], reverse=True) 
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
                    interval=0.3,
                    annotation=1,
                )
                if ckb == 'y':
                    fig.colorbar(frame=["a", "x+lVp (km/s)"], position="JBC+w15c/0.5c+e")

            if geophysics_data == 'MT_ori':

                pygmt.makecpt(cmap='jet', background = 'o',series=[vmin_mt, vmax_mt], reverse=True) 
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
                    interval=0.5,
                    annotation=1,
                )
                if ckb == 'y':
                    fig.colorbar(frame=["a", "x+lLog10 resistivity (@~\127@~-m)"], position="JBC+w15c/0.5c+e")

            HCL1 = well_data[well_data.ID=='HCL-1T'].iloc[0].X, well_data[well_data.ID=='HCL-1T'].iloc[0].Y
            HCL2 = well_data[well_data.ID=='HCL-2T'].iloc[0].X, well_data[well_data.ID=='HCL-2T'].iloc[0].Y
            CTCN = well_data[well_data.ID=='CTCN'].iloc[0].X, well_data[well_data.ID=='CTCN'].iloc[0].Y

            fig.plot(x = HCL1[0], y = HCL1[1], style='s9p', fill='gray', pen="0.5p,white")
            fig.plot(x = HCL2[0], y = HCL2[1], style='s9p', fill='gray', pen="0.5p,white")
            fig.plot(x = CTCN[0], y = CTCN[1], style='s9p', fill='gray', pen="0.5p,white")

            fig.plot(x = well_data.X.iloc[-1], y = well_data.Y.iloc[-1], style='s9p', fill='gray', pen="0.5p,white")

            # 添加標記文字
            fig.text(x=region[1]-0.01, y=region[2]+0.004, text=str(interp_depth)+' km', font='30p,Helvetica-Bold,black')

            # 顯示圖表
            fig.savefig(f'../Fig/{geophysics_data}_{interp_depth}.png', show=False, transparent=True)

            # 清理臨時文件
            os.remove(grid_file)

        # Colorbar Plot Only #
        with pygmt.config(FONT_ANNOT_PRIMARY="40p", FONT_LABEL="50p", MAP_TICK_LENGTH_PRIMARY="10p", MAP_FRAME_PEN="black", MAP_TICK_PEN_PRIMARY="1.5p, black"):
            fig = pygmt.Figure()
            if geophysics_data == 'Vpt_ori':
                grid_Vp = pygmt.surface(x = filtered_data.XX, y = filtered_data.YY, z = filtered_data.Vpt_ori, region=region, spacing=0.0003 ,convergence=0,  verbose=True, tension=0)
                cpt = pygmt.makecpt(cmap='jet', series=[vmin_ptb, vmax_ptb], background = "o", reverse=True)
                fig.colorbar(frame=["a5", "x+ldVp (%)"], cmap=True, position='JBC+w100c/1c+edbf')
                fig.savefig(f'../Fig/colorbar_Vpt_ori.png', show=False, transparent=True)
            
            if geophysics_data == 'Vp_ori':
                grid_Vp = pygmt.surface(x = filtered_data.XX, y = filtered_data.YY, z = filtered_data.Vp_ori, region=region, spacing=0.0003 ,convergence=0,  verbose=True, tension=0)
                cpt = pygmt.makecpt(cmap='jet', series=[vmin_abs, vmax_abs], background = "o", reverse=True)
                fig.colorbar(frame=["a1", "x+lVp (km/s)"], cmap=True, position='JBC+w100c/1c+edbf')
                fig.savefig(f'../Fig/colorbar_Vp_ori.png', show=False, transparent=True)

            if geophysics_data == 'MT_ori':
                grid_Vp = pygmt.surface(x = filtered_data.XX, y = filtered_data.YY, z = filtered_data.MT_ori, region=region, spacing=0.0003 ,convergence=0,  verbose=True, tension=0)
                cpt = pygmt.makecpt(cmap='jet', series=[vmin_mt, vmax_mt], background = "o", reverse=True)
                fig.colorbar(frame=["a0.5", "x+lLog10 resistivity (@~\127@~-m)"], cmap=True, position='JBC+w100c/1c+edbf')
                fig.savefig(f'../Fig/colorbar_MT_ori.png', show=False, transparent=True)


    # Combine all maps
    pattern = f'../Fig/{geophysics_data}_*.png'
    images = sorted(glob.glob(pattern))
    args = ["montage", "-geometry", "+0+0", "-tile", "4x1"] + images + [f'../Fig/maps_{geophysics_data}.png']
    subprocess.run(args)
    
    files = glob.glob(pattern)
    for file in files:
        os.remove(file)
        print(f"Removed: {file}")
    

''' 
# %%

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import geopandas as gpd
import os
from scipy.interpolate import griddata
import matplotlib

# 讀取數據
clusters_results = pd.read_csv('../cluster_results.csv')
sta_Hong_data = pd.read_csv('../stations.csv', delimiter=',', header=None, skiprows=1)
sta_Hong_data.columns = ['sta', 'lon', 'lat', 'H']
well_data = pd.read_csv('../well_all_loc_hong.csv', sep=',')
LYR = gpd.read_file('/home/hmhuang/Research/Hongchailin/clustering_vp_mt/lanyang_poly/lanyang_poly.shp')

output = '../Fig/'
prof_line = [[121.67416, 121.67416, 24.715182, 24.68],
             [121.68900, 121.68900, 24.715182, 24.68],
             [121.70672, 121.70672, 24.715182, 24.68],
             [121.67, 121.717, 24.7107, 24.7107],
             [121.67, 121.717, 24.69773, 24.69773],
             [121.67, 121.717, 24.685, 24.685]]
all_depth = [0.2, 0.4, 0.6, 0.8]
all_geophysics_data = ['Vpt_ori', 'Vp_ori', 'MT_ori']

vmin_abs, vmax_abs = 1, 5
vmin_mt, vmax_mt = 1, np.log10(1000)
vmin_ptb, vmax_ptb = -15, 15
cmap_style = 'jet_r'

# 定義繪圖函數
def plot_map(ax, data, geophysics_data, depth, region):
    lon_min, lon_max, lat_min, lat_max = region

    # 數據過濾
    filtered_data = data[data['ZZ'] == depth]
    x = filtered_data['XX']
    y = filtered_data['YY']
    z = filtered_data[geophysics_data]

    # 插值數據生成網格
    grid_x, grid_y = np.linspace(lon_min, lon_max, 50), np.linspace(lat_min, lat_max, 50)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

    
    # 繪製影像
    if geophysics_data == 'Vpt_ori':
        norm = Normalize(vmin=vmin_ptb, vmax=vmax_ptb)
         
    elif geophysics_data == 'Vp_ori':
        norm = Normalize(vmin=vmin_abs, vmax=vmax_abs)
        
    elif geophysics_data == 'MT_ori':
        norm = Normalize(vmin=vmin_mt, vmax=vmax_mt)
    
    im = ax.pcolormesh(grid_x, grid_y, grid_z, transform=ccrs.PlateCarree(), cmap=cmap_style, norm=norm)
    
    # 繪製等高線
    contour = ax.contour(grid_x, grid_y, grid_z, levels=10, colors='w', linewidths=0.5, transform=ccrs.PlateCarree())
    ax.clabel(contour, inline=True, fontsize=10, fmt='%1.0f')

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent(region, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False 
    gl.ylines = False   
    return im

# 主循環

matplotlib.rcParams['font.family'] = 'Nimbus Sans'
matplotlib.rcParams['font.size'] = 12

for geophysics_data in all_geophysics_data:
    
    for depth in all_depth:
        print(f"Plotting depth {depth} for {geophysics_data}...")

        region = [
            clusters_results['XX'].min(),
            clusters_results['XX'].max(),
            clusters_results['YY'].min(),
            clusters_results['YY'].max()
        ]

        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Mercator()})
        
        im = plot_map(ax, clusters_results, geophysics_data, depth, region)

        # 添加測站與井位
        if depth == 0.8:
            ax.scatter(sta_Hong_data['lon'], sta_Hong_data['lat'], color='blue', s=10, transform=ccrs.PlateCarree(), label='Stations')
            ax.scatter(well_data['X'], well_data['Y'], color='red', s=10, transform=ccrs.PlateCarree(), label='Wells')
            for i, line in enumerate(prof_line):
                ax.plot([line[0], line[1]], [line[2], line[3]], transform=ccrs.PlateCarree(), color='black', linestyle='--')
                ax.text(line[0], line[2], f"{chr(65 + i)}", transform=ccrs.PlateCarree())

        # 添加標題與色標
        ax.set_title(f"{geophysics_data} at Depth {depth} km")
        #cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        #cbar.set_label(f"{geophysics_data}")

        # 保存圖表
        output_file = os.path.join(output, f"{geophysics_data}_{depth:.1f}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()