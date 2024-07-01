# %%
# This code makes the grid of mt model and vp model to be the same.
# 
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import statistics
import time
from sklearn.preprocessing import StandardScaler 

vp_data = pd.read_csv('/home/hmhuang/Work/Seismic_Tomo_result_plot/V19/vpvstomo_1220.dat', delim_whitespace=True, skiprows=1)
mt_data = pd.read_csv('/home/hmhuang/Work/MT_Result/Ilan_MT3D_all.csv', sep=',')
rmap_data = pd.read_csv('/home/hmhuang/Work/Seismic_Tomo_result_plot/V19/vpvsrmap_1220.dat', delim_whitespace=True, skiprows=1)
vporvpt = 'vp'
logornot = 'yes'
ptb_mt_ornot = 'no'

resol_p_lowest = 0.55
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

''' 
scaler = MinMaxScaler()
vp_data['normalized_vpt'] = scaler.fit_transform(vp_data[['vpt']])
vp_data['normalized_vp'] = scaler.fit_transform(vp_data[['vp']])


vp_data.reset_index(drop=True, inplace=True)
vp_data.to_csv('../no_interpo_tomo.csv', index=False)
'''

# ============= Vp Model interpolation and plot ============= #
print('# ============= Vp Model interpolation and plot ============= #')
x_data, y_data, z_data = np.array([vp_data.lon]), np.array([vp_data.lat]), np.array([vp_data.dep])
values = np.array([vp_data.vp])

x_data_flat = x_data.ravel()
y_data_flat = y_data.ravel()
z_data_flat = z_data.ravel() 
values_flat = values.ravel()

interpolated_values_vp = griddata((x_data_flat, y_data_flat, z_data_flat), values_flat, target_points, method='linear')
interpolated_values_vp_reshaped = interpolated_values_vp.reshape(xx.shape)

# ============= Vpt Model interpolation and plot ============= #
print('# ============= Vpt Model interpolation and plot ============= #')
x_data, y_data, z_data = np.array([vp_data.lon]), np.array([vp_data.lat]), np.array([vp_data.dep])
values = np.array([vp_data.vpt])

x_data_flat = x_data.ravel()
y_data_flat = y_data.ravel()
z_data_flat = z_data.ravel()
values_flat = values.ravel()

interpolated_values_vpt = griddata((x_data_flat, y_data_flat, z_data_flat), values_flat, target_points, method='linear')
interpolated_values_vpt_reshaped = interpolated_values_vpt.reshape(xx.shape)

# ============= MT Model interpolation and plot ============= #
print('# ============= MT Model interpolation and plot ============= #')
x_data, y_data, z_data = np.array([mt_data.X_84]), np.array([mt_data.Y_84]), np.array([mt_data.Elevation_m])
values_mt = np.array([mt_data.Rho_ohm_m])
x_data_flat = x_data.ravel()
y_data_flat = y_data.ravel()
z_data_flat = z_data.ravel()
values_flat_mt = values_mt.ravel()
interpolated_values_mt = griddata((x_data_flat, y_data_flat, z_data_flat), values_flat_mt, target_points, method='linear')
interpolated_values_mt_reshaped = interpolated_values_mt.reshape(xx.shape)

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
