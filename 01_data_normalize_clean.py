# %%
# This code makes the grid of mt model and vp model to be the same.
# 
'''
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import statistics
import time
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf
from sklearn.impute import KNNImputer
mt_data = pd.read_csv('/home/hmhuang/Research/Hongchailin/MT_Result/Ilan_MT3D_all.csv', sep=',')
vp_data = pd.read_csv('/home/hmhuang/Research/Hongchailin/V19/vpvstomo_1220.dat', delim_whitespace=True, skiprows=1)
rmap_data = pd.read_csv('/home/hmhuang/Research/Hongchailin/V19/vpvsrmap_1220.dat', delim_whitespace=True, skiprows=1)
vporvpt = 'vp'
logornot = 'yes'
ptb_mt_ornot = 'no'

resol_p_lowest = 0.6
interval_interpol = 0.002
x_range = np.arange(121.653, 121.735, interval_interpol)
y_range = np.arange(24.666, 24.725, interval_interpol)
z_target = np.arange(0, 1, interval_interpol) 
# Boundaries and intervals for interpolation
xx, yy, zz = np.meshgrid(x_range, y_range, z_target, indexing='ij')
target_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(float)
# Grids generation

mt_data['Elevation_m'] = mt_data['Elevation_m'] * -0.001
vp_data['resol_p'] = rmap_data.resol_p
VP_new = vp_data[vp_data.resol_p>resol_p_lowest]
vp_data = VP_new

# Vp Model interpolation and plot
print('# ============= Vp Model interpolation and plot ============= #')
x_data, y_data, z_data = np.array([vp_data.lon]), np.array([vp_data.lat]), np.array([vp_data.dep])
values = np.array([vp_data.vp])
x_data_flat = x_data.ravel()
y_data_flat = y_data.ravel()
z_data_flat = z_data.ravel() 
values_flat = values.ravel()
interpolated_values_vp = griddata((x_data_flat, y_data_flat, z_data_flat), values_flat, target_points, method='linear', fill_value=np.nan)

# Vpt Model interpolation and plot
print('# ============= Vpt Model interpolation and plot ============= #')
x_data, y_data, z_data = np.array([vp_data.lon]), np.array([vp_data.lat]), np.array([vp_data.dep])
values = np.array([vp_data.vpt])
x_data_flat = x_data.ravel()
y_data_flat = y_data.ravel()
z_data_flat = z_data.ravel()
values_flat = values.ravel()
interpolated_values_vpt = griddata((x_data_flat, y_data_flat, z_data_flat), values_flat, target_points, method='linear', fill_value=np.nan)

# MT Model interpolation and plot
print('# ============= MT Model interpolation and plot ============= #')
x_data, y_data, z_data = np.array([mt_data.X_84]), np.array([mt_data.Y_84]), np.array([mt_data.Elevation_m])
values_mt = np.array([mt_data.Rho_ohm_m])
x_data_flat = x_data.ravel()
y_data_flat = y_data.ravel()
z_data_flat = z_data.ravel()
values_flat_mt = values_mt.ravel()
interpolated_values_mt = griddata((x_data_flat, y_data_flat, z_data_flat), values_flat_mt, target_points, method='linear', fill_value=np.nan)

# ================== Data Preprocessing for Clustering ================== 
print('# ================== Data Preprocessing for Clustering ================== #')
data = {'Lon': target_points[:,0], 'Lat': target_points[:,1], 'Dep': target_points[:,2], 'Vp': interpolated_values_vp, 'Resis': interpolated_values_mt, 'Vpt': interpolated_values_vpt}
threeD_df = pd.DataFrame(data)

new_vp = threeD_df.Vp
if vporvpt == 'vp':
    new_vpt = threeD_df.Vp
if vporvpt == 'vpt':
    new_vpt = threeD_df.Vpt

new_mt = threeD_df.Resis
new_mt_mean = new_mt - statistics.mean(new_mt)
Y_log = np.log(new_mt)
Y_log_mean = Y_log - statistics.mean(Y_log)

Y = new_mt
X = new_vpt

if logornot == 'yes':
    if ptb_mt_ornot == 'yes':
        Y = Y_log_mean
    else:
        Y = Y_log
elif logornot == 'no':
    if ptb_mt_ornot == 'yes':
        Y = new_mt_mean
    else:
        Y = new_mt
mask = np.isnan(X)
all_nan = np.where(~mask)[0]
X_clearnan = X[all_nan]
Y_clearnan = np.array(Y)[all_nan]
#mask = np.isnan(Z)
#all_nan = np.where(~mask)[0]
#Z_clearnan = X_clearnan.values/Y_clearnan
Z_clearnan = Y_clearnan/X_clearnan.values

x = np.array(threeD_df.Lon)[all_nan]
y = np.array(threeD_df.Lat)[all_nan]
z = np.array(threeD_df.Dep)[all_nan]
new_mt_nona = np.array(threeD_df.Resis)[all_nan]
new_vp_nona = np.array(threeD_df.Vp)[all_nan]
new_vpt_nona = np.array(threeD_df.Vpt)[all_nan]

X_scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(X_clearnan).reshape(-1, 1))
X_normal = X_scaler.transform(np.array(X_clearnan).reshape(-1, 1))

Y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(Y_clearnan).reshape(-1, 1))
Y_normal = Y_scaler.transform(np.array(Y_clearnan).reshape(-1, 1))

Z_scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(Z_clearnan).reshape(-1, 1))
Z_normal = Z_scaler.transform(np.array(Z_clearnan).reshape(-1, 1))

X = X_normal.flatten()
Y = Y_normal.flatten()
Z = Z_normal.flatten()

data_new = pd.DataFrame({'x': X, 'y': Y, 'z': Z, 'x_coor': x, 'y_coor': y, 'z_coor': z, 'Vp_value': new_vp_nona, 'MT_Value': new_mt_nona, 'Vpt_value': new_vpt_nona})
data_nona = data_new.dropna()
data_xy = data_new.drop(['x_coor', 'y_coor', 'z_coor', 'Vp_value', 'MT_Value', 'Vpt_value'], axis=1)
data_nona.to_csv('../data_nona_' + str(interval_interpol) + '.csv', index=False)
data_xy.to_csv('../data_xy_'  + str(interval_interpol) + '.csv', index=False)
# %%
'''


# %%
# This code makes the grid of mt model and vp model to be the same.
# 
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
import math
# Load data
mt_data = pd.read_csv('../../MT_Result/Ilan_MT3D_all.csv', sep=',')
vp_data = pd.read_csv('../../V19/vpvstomo_1220.dat', delim_whitespace=True, skiprows=1)
rmap_data = pd.read_csv('../../V19/vpvsrmap_1220.dat', delim_whitespace=True, skiprows=1)

mt_data['Elevation_m'] = mt_data['Elevation_m'] * -0.001
# Define interpolation grid
resol_p_lowest = 0.6
interval_interpol_hori = 0.002  # degree
interval_interpol_vertical = 0.002  # km

x_range = np.arange(121.653, 121.735, interval_interpol_hori)
y_range = np.arange(24.666, 24.725, interval_interpol_hori)
z_target = np.arange(0, 1, interval_interpol_vertical)

# Extracting x, y, z, and values for interpolation from vp_data with correct column names
x = vp_data['lon']
y = vp_data['lat']
z = vp_data['dep']
values_vpt = vp_data['vpt']
values_vp = vp_data['vp']

# Create meshgrid for interpolation
grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_target, indexing='ij')

# Combine x, y, z into coordinates for griddata
points = np.array([x, y, z]).T

# Interpolate Vpt values onto the new grid
grid_values_vpt = griddata(points, values_vpt, (grid_x, grid_y, grid_z), method='linear', fill_value=np.nan)
# Interpolate Vp values onto the new grid
grid_values_vp = griddata(points, values_vp, (grid_x, grid_y, grid_z), method='linear', fill_value=np.nan)

# Extracting x, y, z, and values for interpolation from mt_data
mt_data_filtered = mt_data[mt_data['Rho_ohm_m'] <= 5000]
x_mt = mt_data_filtered['X_84']
y_mt = mt_data_filtered['Y_84']
z_mt = mt_data_filtered['Elevation_m']
values_mt = np.log10(mt_data_filtered['Rho_ohm_m'])

# Interpolate MT values onto the new grid
points_mt = np.array([x_mt, y_mt, z_mt]).T
grid_values_mt = griddata(points_mt, values_mt, (grid_x, grid_y, grid_z), method='linear', fill_value=np.nan)

# Create DataFrame from interpolated values
target_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
interpolated_values_vp = grid_values_vp.flatten()
interpolated_values_mt = grid_values_mt.flatten()
interpolated_values_vpt = grid_values_vpt.flatten()
interpolated_values_mt_vp = interpolated_values_mt/interpolated_values_vp

# Normalize the interpolated values to 0-1
scaler = MinMaxScaler()

# Stack all the interpolated values together
values_to_normalize = np.vstack([
    interpolated_values_vp,
    interpolated_values_mt,
    interpolated_values_vpt,
    interpolated_values_mt_vp
]).T

# Fit and transform the data
normalized_values = scaler.fit_transform(values_to_normalize)


# Extract the normalized values back into separate arrays
normalized_values_vp = normalized_values[:, 0]
normalized_values_mt = normalized_values[:, 1]
normalized_values_vpt = normalized_values[:, 2]
normalized_values_mt_vp = normalized_values[:, 3]

valid_mask = ~np.isnan(normalized_values_vp) & ~np.isnan(normalized_values_mt) & ~np.isnan(normalized_values_vpt) & ~np.isnan(normalized_values_mt_vp)

data = {
    'Lon': target_points[valid_mask, 0],
    'Lat': target_points[valid_mask, 1],
    'Dep': target_points[valid_mask, 2],
    'Vp': interpolated_values_vp[valid_mask],
    'Resis': interpolated_values_mt[valid_mask],
    'Vpt': interpolated_values_vpt[valid_mask],
    'Vp_norm': normalized_values_vp[valid_mask],
    'Resis_norm': normalized_values_mt[valid_mask],
    'Vpt_norm': normalized_values_vpt[valid_mask],
    'Resis_Vp_norm': normalized_values_mt_vp[valid_mask]
}

threeD_df = pd.DataFrame(data)
#scaler = MinMaxScaler()
#threeD_df[['Vp_norm', 'Resis_norm', 'Vpt_norm']] = scaler.fit_transform(threeD_df[['Vp_norm', 'Resis_norm', 'Vpt_norm']])
threeD_df.to_csv('../data_nona_' + str(interval_interpol_vertical) + '.csv', index=False)



# data_xy = threeD_df.drop(['Lon', 'Lat', 'Dep', 'Vp', 'Resis', 'Vpt', 'Vpt_norm'], axis=1)
# data_xy.to_csv('../data_xy_' + str(interval_interpol_vertical) + '.csv', index=False)
