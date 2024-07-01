# %%
import pandas as pd
import matplotlib.pyplot as plt
import string
import matplotlib
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import spearmanr

uppercase_letters = list(string.ascii_uppercase)
# ============ Setting  ============ #
data_clusters = pd.read_csv('../cluster_results.csv')
cluster_method = data_clusters.clustering_method[0]
vporvpt = 'vp'
y_scale_logornot = 'no'
# %%

if vporvpt=='vpt':
    xlim = [-30, 30]
if vporvpt=='vp':
    xlim = [1.7, 4]   
ytickssss = [1, 10, 100, 1000, 10000, 100000]
ytickslabelll = ['0', '1', '2', '3', '4', '5']
y_nolog_zoom = [0, 1500]


cluster_number = max(data_clusters.Clusters) + 1
colors = cm.Set3.colors
colors_cluster_all = cm.Set3.colors
colors_cluster = colors_cluster_all[0:cluster_number]


# ============ Plotting  ============ #
print('===== 3D View  =====')
matplotlib.rcParams['font.family'] = 'Nimbus Roman'
matplotlib.rcParams['font.size'] = 15
fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(projection='3d')
for i in range(cluster_number):
    print('Cluster ' + str(i))
    ax.scatter(data_clusters.Vp[data_clusters.Clusters==i], data_clusters.Ohm[data_clusters.Clusters==i], data_clusters.Vp_Ohm_ratio[data_clusters.Clusters==i],s = 0.5, c=colors[i])

ax.set_xlabel('Normalized Vp', labelpad=15)
ax.set_ylabel('Normalized Resistivity', labelpad=15)
ax.set_zlabel('Normalized Vp/Resistivity', labelpad=15)
ax.grid(True)
plt.savefig('../Fig/' + cluster_method + '_' + str(cluster_number) + '_3Dview.png', dpi=300)
plt.close()


print('===== Original Observations =====')
matplotlib.rcParams['font.family'] = 'Nimbus Roman'
matplotlib.rcParams['font.size'] = 20
fig, ax = plt.subplots(figsize=(13, 10))
for i in range(cluster_number):
    ax.grid(True)
    if vporvpt == 'vpt':
        ax.scatter(data_clusters.Vpt_ori[data_clusters.Clusters==i], 
                data_clusters.MT_ori[data_clusters.Clusters==i],
                s=0.05, c=colors[i], label='Cluster ' + uppercase_letters[i])
    if vporvpt == 'vp':
        ax.scatter(data_clusters.Vp_ori[data_clusters.Clusters==i], 
                data_clusters.MT_ori[data_clusters.Clusters==i],
                s=0.05, c=colors[i], label='Cluster ' + uppercase_letters[i])        
        #ax.scatter(data_clusters.Vp_ori[data_clusters.Clusters==i], 
        #        data_clusters.Vp_ori[data_clusters.Clusters==i]/data_clusters.MT_ori[data_clusters.Clusters==i],
        #        s=0.05, c=colors[i], label='Cluster ' + uppercase_letters[i])          
    ax.set_xlim(xlim) 

    if y_scale_logornot == 'yes':
        ax.set_yscale('log')  
        ax.set_yticks(ytickssss)  
        ax.set_yticklabels(ytickslabelll)  
    else:
        try: 
            ax.set_ylim(y_nolog_zoom)
        except:
            pass

leg = ax.legend(loc='lower right', facecolor='black', edgecolor='black', framealpha=0.3, markerscale=30)
plt.setp(leg.get_texts(), color='white')

if vporvpt == 'vpt':
    ax.set_xlabel('dVp (%)') 
if vporvpt == 'vp':
    ax.set_xlabel('Vp (km/s)')   
ax.set_ylabel('Log10 resistivity (Ωm)')  
ax.set_title(cluster_method + ' for ' + str(cluster_number) + ' clusters') 
plt.savefig('../Fig/' + cluster_method + '_' + str(cluster_number) + '.png', dpi=300)
plt.close()

print('===== Normalized Values =====')
fig, ax = plt.subplots(figsize=(13, 10))
for i in range(cluster_number):
    ax.grid(True)
    ax.scatter(data_clusters.Vp[data_clusters.Clusters==i], 
               data_clusters.Ohm[data_clusters.Clusters==i],
               s=0.05, c=colors[i], label='Cluster ' + uppercase_letters[i])
leg = ax.legend(loc='lower right', facecolor='black', edgecolor='black', framealpha=0.3, markerscale=30)
plt.setp(leg.get_texts(), color='white')
ax.set_xlabel('Feature 1')  
ax.set_ylabel('Feature 2') 
ax.set_title(cluster_method + ' for ' + str(cluster_number) + ' clusters in normalized values')  

plt.savefig('../Fig/' + cluster_method + '_' + str(cluster_number) + '_Normalized.png', dpi=300)
plt.close()
# %%
''' 
print('===== each cluster =====')
matplotlib.rcParams['font.family'] = 'Nimbus Roman'
matplotlib.rcParams['font.size'] = 10

for i in range(len(colors_cluster)):
    plt.figure(i)
    #plt.scatter(x = data_clusters.Vp[data_clusters.Clusters==i], y = data_clusters.Ohm[data_clusters.Clusters==i], s=0.05, c=colors[i])
    sns.regplot(x = data_clusters.Vp[data_clusters.Clusters==i], 
                y = data_clusters.Ohm[data_clusters.Clusters==i], 
                color = colors[i], 
                scatter_kws={'s':0.05}, 
                line_kws={'color':'black', 'linewidth':1, 'linestyle':'--'})    
    plt.xlabel('Feature 1')  
    plt.ylabel('Feature 2')
    plt.savefig('../Fig/each_cluster_cluster_' + str(i) + '.png', dpi=300)
    plt.close()
'''
# %%
print('===== Latent Space =====')
matplotlib.rcParams['font.family'] = 'Nimbus Roman'
matplotlib.rcParams['font.size'] =  20
fig, ax = plt.subplots(figsize=(13, 10))
for i in range(cluster_number):
    ax.grid(True)
    ax.scatter(data_clusters.Vp_encode[data_clusters.Clusters==i], 
               data_clusters.Ohm_encode[data_clusters.Clusters==i],
               s=0.05, c=colors[i], label='Cluster ' + uppercase_letters[i])
leg = ax.legend(loc='lower right', facecolor='black', edgecolor='black', framealpha=0.3, markerscale=30)
plt.setp(leg.get_texts(), color='white')
ax.set_xlabel('Latent 1')  
ax.set_ylabel('Latent 2') 
ax.set_title(cluster_method + ' for ' + str(cluster_number) + ' clusters in latent space')  

plt.savefig('../Fig/' + cluster_method + '_' + str(cluster_number) + '_Latent.png', dpi=300)
plt.close()

# %%
