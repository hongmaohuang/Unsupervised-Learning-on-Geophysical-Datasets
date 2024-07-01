# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import subprocess

data_clusters = pd.read_csv('../cluster_results.csv')
#data_clusters = pd.read_csv('../cluster_results_combine.csv')

binss = 50
#resolution

cluster_number = max(data_clusters.Clusters) + 1
colors = cm.Set3.colors
colors_cluster_all = cm.Set3.colors
colors_cluster = colors_cluster_all[0:cluster_number]



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
args = ["montage", "-geometry", "-0-0", "-tile", "3x3"] + images + ["../Fig/output.png"]
subprocess.run(args)
