# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import subprocess
import glob
import os
data_clusters = pd.read_csv('../cluster_results.csv')
#data_clusters = pd.read_csv('../cluster_results_combine.csv')

binss = 50
#resolution

cluster_number = max(data_clusters.Clusters) + 1
colors = cm.Set3.colors
deep_yellow = colors[-1]
index_of_light_yellow = 1  # 這個索引根據實際情況調整
colors_cluster_all = list(colors)
colors_cluster_all[index_of_light_yellow] = deep_yellow
cluster_number = max(data_clusters.Clusters) + 1
colors_cluster = colors_cluster_all[0:cluster_number]

all_data_x = data_clusters['Vp_ori']
all_data_y = data_clusters['MT_ori']
coefficients = np.polyfit(all_data_x, np.log(all_data_y), 1)
polynomial = np.poly1d(coefficients)
regression_line_x = np.linspace(min(all_data_x), max(all_data_x), 100)
regression_line_y = np.exp(polynomial(regression_line_x))


cluster_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
plt.figure(figsize=(10, 10))
matplotlib.rcParams['font.family'] = 'Liberation Sans'
matplotlib.rcParams['font.size'] = 15
num_clusters = len(colors_cluster)
for I in range(num_clusters):
    cluster_data = data_clusters[data_clusters.Clusters == I]
    plt.scatter(x=cluster_data.Vp_ori, y=cluster_data.MT_ori, color=colors_cluster[I], label=f'Cluster {cluster_labels[I]}', s=0.5)

plt.plot(regression_line_x, regression_line_y, color='gray', linestyle='--', alpha=0.5)
plt.grid(alpha=0.3)
plt.xlabel('Vp (km/s)')
plt.ylabel('Log resistivity (Ωm)')
#plt.yscale('log')
legend = plt.legend(markerscale=10)
legend.get_frame().set_alpha(0.3) 
plt.savefig('../Fig/scatterplot_overall.png', dpi=300)

# %%
for i in range(len(colors_cluster)):
    x = data_clusters.Vp_ori[data_clusters.Clusters == i]
    y = np.log(data_clusters.MT_ori[data_clusters.Clusters == i])
    color = colors_cluster_all[i]
    cmap = LinearSegmentedColormap.from_list("mycmap", ["white", color])
    # 創建 figure 和三個子圖
    matplotlib.rcParams['font.family'] = 'Liberation Sans'
    matplotlib.rcParams['font.size'] = 25
    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(223)
    ax2 = fig.add_subplot(221)
    ax3 = fig.add_subplot(224)

    # 第一個子圖：散點圖
    hist = ax1.hist2d(x, y, bins=(binss, binss), cmap=cmap, vmin=0)
    #plt.colorbar(hist[3], ax=ax1)

    ax1.set_xlabel('Vp (km/s)')
    ax1.set_ylabel('Log Resistivity (Ωm)')
    ax1.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)
    # 進行一次多項式擬合
    coefficients, cov_matrix = np.polyfit(x, y, 1, cov=True)
    polynomial = np.poly1d(coefficients)

    # 計算擬合線的誤差範圍
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = polynomial(x_fit)
    error = np.sqrt(np.diag(cov_matrix))
    upper_bound = y_fit + 1.96 * error[0]  # 1.96 是 95% 信賴區間的上界
    lower_bound = y_fit - 1.96 * error[0]  # 1.96 是 95% 信賴區間的下界

    # 繪製擬合線及誤差範圍
    ax1.plot(x_fit, y_fit, color='black', linestyle='-', linewidth=4, label='Fit Line')
    ax1.plot(x_fit, upper_bound, color='gray', linestyle='--', linewidth=1, label='Upper Bound')
    ax1.plot(x_fit, lower_bound, color='gray', linestyle='--', linewidth=1, label='Lower Bound')
    ax1.fill_between(x_fit, upper_bound, lower_bound, color='gray', alpha=0.2)

    # 第二個子圖：柱狀圖
    ax2.hist(x, bins=binss, color=color, alpha=0.3, label='Vp Histogram')
    ax2.set_ylabel('Counts')
    ax2.set_xticklabels([]) 

    # 第三個子圖：柱狀圖
    ax3.hist(y, bins=binss, color=color, alpha=0.3, orientation='horizontal')
    ax3.set_xlabel('Counts')
    ax3.set_yticklabels([]) 

    # 調整子圖的間距和柱狀圖的大小
    fig.tight_layout()

    # 調整柱狀圖的大小
    divider = ax2.get_position()
    ax2.set_position([divider.x0, divider.y0, divider.width , divider.height* 0.3])

    divider = ax3.get_position()
    ax3.set_position([divider.x0, divider.y0, divider.width* 0.3, divider.height ])

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    plt.savefig('../Fig/Clusters_' + str(i) + '_hist.png', bbox_inches='tight')

images = []
for i in range(0, len(colors_cluster)):
    images.append('../Fig/Clusters_' + str(i) + '_hist.png')
args = ["montage", "-geometry", "-0-0", "-tile", "3x3"] + images + ["../Fig/scatter-plots.png"]
subprocess.run(args)

files = glob.glob('../Fig/Clusters*')
for file in files:
    os.remove(file)
    print(f"Removed: {file}")
# %%
