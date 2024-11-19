#
# This code makes the grid of mt model and vp model to be the same.
# 
# %%
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt

# Define interpolation grid
interval_interpol_hori = 'original_resol'
'0.002 # degree'
interval_interpol_vertical = 'original_resol'
'0.002  # km'

# Load data
mt_data = pd.read_csv('../../MT_Result/Ilan_MT3D_all.csv', sep=',')
vp_data = pd.read_csv('../../V19/vpvstomo_1220.dat', delim_whitespace=True, skiprows=1)
rmap_data = pd.read_csv('../../V19/vpvsrmap_1220.dat', delim_whitespace=True, skiprows=1)

mt_data['Elevation_m'] = mt_data['Elevation_m'] * -0.001

x_range = vp_data.lon.unique()
y_range = vp_data.lat.unique()
z_target = vp_data.dep.unique()

grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_target, indexing='ij')

mt_data_filtered = mt_data[mt_data['Rho_ohm_m'] <= 10000]
x_mt = mt_data_filtered['X_84']
y_mt = mt_data_filtered['Y_84']
z_mt = mt_data_filtered['Elevation_m']
values_mt = np.log10(mt_data_filtered['Rho_ohm_m'])

points_mt = np.array([x_mt, y_mt, z_mt]).T
grid_values_mt = griddata(points_mt, values_mt, (grid_x, grid_y, grid_z), method='linear', fill_value=np.nan)

all_geophysical_data =  vp_data
all_geophysical_data['interpolated_mt'] = griddata(points_mt, values_mt, (vp_data['lon'], vp_data['lat'], vp_data['dep']), method='linear', fill_value=np.nan)

interpolated_values_vp = all_geophysical_data.vp
interpolated_values_mt = all_geophysical_data.interpolated_mt
interpolated_values_vpt = all_geophysical_data.vpt
interpolated_values_mt_vp = interpolated_values_mt/interpolated_values_vp

'''

# Define interpolation grid
interval_interpol_hori = 0.002 # degree
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
''' 
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
    'Lon': all_geophysical_data.lon[valid_mask],
    'Lat': all_geophysical_data.lat[valid_mask],
    'Dep': all_geophysical_data.dep[valid_mask],
    'Vp': interpolated_values_vp[valid_mask],
    'Resis': interpolated_values_mt[valid_mask],
    'Vpt': interpolated_values_vpt[valid_mask],
    'Vp_norm': normalized_values_vp[valid_mask],
    'Resis_norm': normalized_values_mt[valid_mask],
    'Vpt_norm': normalized_values_vpt[valid_mask],
    'Resis_Vp_norm': normalized_values_mt_vp[valid_mask]
}

threeD_df = pd.DataFrame(data)
threeD_df.to_csv('../data_nona_' + interval_interpol_vertical + '.csv', index=False)
