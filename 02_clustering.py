# %%

import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.interpolate import griddata

# Function to remove unnecessary files
def clean_files():
    files_to_remove = ['../tomo.nc', '../cluster_results.csv']
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)

clean_files()

# Settings
interval_interpol = '0.002'
gmm_clusters = 2
cluster_method = 'GMM'
n_init = 10
cov_type = 'full'
# 'full' (each cluster has its own full covariance matrix),
# 'tied' (all clusters share the same covariance matrix),
# 'diag' (diagonal covariance matrices), or 'spherical' (isotropic, single variance per cluster)
init_params = 'kmeans'
# Method for initializing parameters
max_iter = 100
reg_covar = 1e-4
# Used to avoid singular covariance matrices; for high-dimensional data, this can be set higher. Default is 1e-6
tol = 1e-3
# Convergence tolerance, default is 1e-3

evaluation_of_num_cluster = 'no'
max_clusters = 30

# Load data
data_nona = pd.read_csv(f'../data_nona_{interval_interpol}.csv')


# Data processing
df = pd.DataFrame(data_nona)
df = df.drop(columns=['Lon', 'Lat', 'Dep', 'Vp', 'Resis', 'Vpt', 'Vpt_norm'])
data = df.values
data_trans = np.transpose(data)
data = data_trans
data_xy_tran = np.stack(data)
data_xy_clu = np.transpose(data_xy_tran)

print(data_xy_clu)

# Clustering
if evaluation_of_num_cluster == 'yes':
    bic = []
    aic = []
    for i in range(1, max_clusters+1):
        print(str(i) + '/' + str(max_clusters))
        gmm = GaussianMixture(n_components=i)
        gmm.fit(data_xy_clu)
        bic.append(gmm.bic(data_xy_clu))
        aic.append(gmm.aic(data_xy_clu))

    # BIC Curve Plot
    plt.figure(figsize=(10, 6))
    matplotlib.rcParams['font.family'] = 'Nimbus Sans'
    matplotlib.rcParams['font.size'] = 10
    plt.grid(True)
    plt.plot(range(1, max_clusters+1), bic, marker='o', color='gray')
    #plt.title('BIC for GMM with different number of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('BIC')
    plt.savefig('../Fig/gmm_bic_num_cluster.png', dpi=300)

# GMM
print('GMM clustering')
gmm = GaussianMixture(n_components=gmm_clusters, covariance_type=cov_type, n_init=n_init, init_params=init_params, max_iter=max_iter, reg_covar=reg_covar, tol=tol)
gmm.fit(data_xy_clu)
labels = gmm.predict(data_xy_clu)
print('Compelet GMM clustering')

# Create DataFrame for clustering results
cluster_data = {
    'Vp': data_nona['Vp_norm'],
    'Ohm': data_nona['Resis_norm'],
    'Ohm_Vp_ratio': data_nona['Resis_Vp_norm'],
    'Clusters': labels,
    'YY': data_nona['Lat'],
    'XX': data_nona['Lon'],
    'ZZ': data_nona['Dep'],
    'Vp_ori': data_nona['Vp'],
    'MT_ori': data_nona['Resis'],
    'Vpt_ori': data_nona['Vpt'],
    'clustering_method': [cluster_method] * len(data_nona)
}
df_cluster = pd.DataFrame(cluster_data)
df_cluster.to_csv('../cluster_results.csv', index=False)

''' 
# NC File Generation 
# Here the code transfers the output file of TomoFlex (e.g., vpvstommo.dat) to NetCDF file #

data = df_cluster
Depth_all = sorted(data.ZZ.unique())
ndepth = len(Depth_all)
lon = np.unique(np.array(data['XX']))
lat = np.unique(np.array(data['YY']))
ny, nx = (len(lat), len(lon))
Vpppout = []
for i in Depth_all:
    filtered_data_dep = data[data['ZZ'] == i]
    new = []
    for j in lat:
        filtered_data_lat = filtered_data_dep[filtered_data_dep['YY']==j]
        row = list(filtered_data_lat.Vp_ori)
        # If the length of row is smaller than nx, then add nan at the end of row
        if len(row) < nx:
            row.extend([np.nan] * (nx - len(row)))
        new.append(row)
    Vpppout.append(new)


Vpppttttout = []
for i in Depth_all:
    filtered_data_dep = data[data['ZZ'] == i]
    new = []
    for j in lat:
        filtered_data_lat = filtered_data_dep[filtered_data_dep['YY']==j]
        row = list(filtered_data_lat.Vpt_ori)
        # If the length of row is smaller than nx, then add nan at the end of row
        if len(row) < nx:
            row.extend([np.nan] * (nx - len(row)))
        new.append(row)
    Vpppttttout.append(new)


Mtttout = []
for i in Depth_all:
    filtered_data_dep = data[data['ZZ'] == i]
    new = []
    for j in lat:
        filtered_data_lat = filtered_data_dep[filtered_data_dep['YY']==j]
        row = list(filtered_data_lat.MT_ori)
        # If the length of row is smaller than nx, then add nan at the end of row
        if len(row) < nx:
            row.extend([np.nan] * (nx - len(row)))
        new.append(row)
    Mtttout.append(new)

Clusterrrrs = []
for i in Depth_all:
    filtered_data_dep = data[data['ZZ'] == i]
    new = []
    for j in lat:
        filtered_data_lat = filtered_data_dep[filtered_data_dep['YY']==j]
        row = list(filtered_data_lat.Clusters)
        # If the length of row is smaller than nx, then add nan at the end of row
        if len(row) < nx:
            row.extend([np.nan] * (nx - len(row)))
        new.append(row)
    Clusterrrrs.append(new)

ncout = Dataset('../tomo.nc','w','NETCDF3'); # using netCDF3 for output format 
ncout.createDimension('lon',nx);
ncout.createDimension('lat',ny);
ncout.createDimension('depth',ndepth);
lonvar = ncout.createVariable('lon','float32',('lon'));lonvar[:] = lon;
latvar = ncout.createVariable('lat','float32',('lat'));latvar[:] = lat;
depvar = ncout.createVariable('depth','float32',('depth'));depvar[:] = Depth_all;
vp = ncout.createVariable('vp','float32',('depth','lat','lon'));vp.setncattr('units','m/s');vp[:] = Vpppout;
vpttt = ncout.createVariable('vpt','float32',('depth','lat','lon'));vpttt.setncattr('units','%');vpttt[:] = Vpppttttout;
mttt = ncout.createVariable('mt','float32',('depth','lat','lon'));mttt.setncattr('units','ohm-m');mttt[:] = Mtttout;
cluterrrrrr = ncout.createVariable('clusters','float32',('depth','lat','lon'));cluterrrrrr.setncattr('units','piece');cluterrrrrr[:] = Clusterrrrs;
ncout.close()

'''