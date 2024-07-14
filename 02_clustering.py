# ============ Setting  ============ #
import os 
import sys
import glob
import pickle
import numpy as np
import pandas as pd 
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.interpolate import griddata
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Dropout

from tensorflow.keras.losses import Huber


NUM_PARALLEL_EXEC_UNITS=20
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_AFFINITY"]= "granularity=fine, verbose, compact, 1, 0"
try:
    os.remove('../tomo.nc')
except:
    print('No file to be deleted')

# ============ Parameters  ============ #
interval_interpol = '0.002'
depth_cluster_result = 0.4
cluster_method = 'GMM'
gmm_clusters = 6
kmean_clusters = 6
vq_clusters = 8 #要2的指數 （2, 4, 8, 16）
#K-means, GMM, VQ
evaluation_of_num_cluster = 'no'
max_clusters = 50  # 最大分群數量，用來找最佳分群數
autoencoder_epoch = 500
autoencoder_batch = 4096*6
optimizer = 'Adam'
huber_loss = Huber()
loss = huber_loss
#'mse' #huber_loss
train_autoencoder = 'load encoded data'
# yes, no, load encoded data
plot_the_input_and_output = 'no'

# ============ Data Loading  ============ #
data_xy = pd.read_csv('../data_xy_' + interval_interpol + '.csv')
data_nona = pd.read_csv('../data_nona_' + interval_interpol + '.csv')
data_dict = data_nona
df = pd.DataFrame(data_dict)
df = df.drop(columns=['z', 'x_coor', 'y_coor', 'z_coor', 'Vp_value', 'MT_Value' , 'Vpt_value'])
data = df.values

# ============ Autoencoder  ============ #
if train_autoencoder == 'yes':
    print(' ============== Input Data ==============')
    print(data)
    print(' ========================================== ')
    # 原本的輸入特徵數
    input_dim = data.shape[1]  

    # Neural Network
    autoencoder = Sequential([
        # Encoder
        Dense(input_dim, activation='tanh', input_shape=(input_dim,), name='input_layer'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='tanh'),  
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='tanh'),  
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='tanh'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(8, activation='tanh'),  
        BatchNormalization(),
        Dropout(0.2),
        
        # Latent Layer
        Dense(4, activation='tanh', name='latent_layer'), 
        BatchNormalization(),
        Dropout(0.2),
        
        # Decoder
        Dense(8, activation='tanh'),  
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='tanh'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='tanh'), 
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='tanh'),  
        BatchNormalization(),
        Dropout(0.2),
        Dense(input_dim, activation='tanh')  
    ])

    # Compile
    autoencoder.compile(optimizer=optimizer, loss=loss)

    # Structure
    #autoencoder.summary()

    # Self-supervised learning
    history = autoencoder.fit(data, data, epochs=autoencoder_epoch, batch_size=autoencoder_batch)
    
    # Save the history
    with open('autoencoder_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    # Plot the loss curve
    matplotlib.rcParams['font.family'] = 'Nimbus Sans'
    matplotlib.rcParams['font.size'] = 15
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', c='gray', linewidth = 2)
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('../Fig/loss_encoder.png', dpi=300)

    # Model
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('latent_layer').output)

    # Apply the encoder to the data and check if any of the latent space dimensions are almost all near-zeros
    latent_space_output = encoder.predict(data)
    print(' ============== Latent Space ==============')
    print(latent_space_output)
    latent_space_output = latent_space_output[:, np.mean(np.abs(latent_space_output) < 1e-5, axis=0) < 0.9]
    # Remove the columns that are almost all near-zeros
    print(latent_space_output)
    print(' ================================')

    if latent_space_output.shape[1] == 0 or latent_space_output.shape[1] == 1:
        print('The latent space is empty or only has one dimension. Please re-run the code.')
        sys.exit()

    df = pd.DataFrame(latent_space_output)
    df.to_csv('../latent_space_output.csv', index=False)
if train_autoencoder == 'no':
    latent_space_output = data

else:
    with open('autoencoder_history.pkl', 'rb') as file:
        history = pickle.load(file)
    # Plot the loss curve
    matplotlib.rcParams['font.family'] = 'Nimbus Sans'
    matplotlib.rcParams['font.size'] = 15
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', c='gray', linewidth = 2)
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('../Fig/loss_encoder.png', dpi=300)
    latent_space_output = pd.read_csv('../latent_space_output.csv').values

if plot_the_input_and_output == 'yes': 
    #Plot the input in map view
    latent_space_output = pd.read_csv('../latent_space_output.csv').values
    matplotlib.rcParams['font.family'] = 'Liberation Sans'
    matplotlib.rcParams['font.size'] = 15
    x = data_nona.x_coor[data_nona.z_coor==depth_cluster_result]
    y = data_nona.y_coor[data_nona.z_coor==depth_cluster_result]
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

    z = data_nona.x[data_nona.z_coor==depth_cluster_result]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    plt.figure(1, figsize=(10, 6), dpi=300)
    plt.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='jet_r')
    plt.title('Normalized Values (Vpt)')
    plt.savefig('../Fig/Normalized_Values_Vpt.png', dpi=300)    

    z = data_nona.y[data_nona.z_coor==depth_cluster_result]
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    plt.figure(2, figsize=(10, 6), dpi=300)
    plt.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='jet_r')
    plt.title('Normalized Values (Resist.)')
    plt.savefig('../Fig/Normalized_Values_Resis.png', dpi=300)

    #Plot the latent number in map view
    encoding_dim = latent_space_output.shape[1]
    df_latent_space = pd.DataFrame(latent_space_output, columns=[f'zz{i}' for i in range(encoding_dim)])
    latent_values = [f'zz{i}' for i in range(encoding_dim)]
    titles = [f'Latent Value {i} ' for i in range(encoding_dim)]
    fig_numbers = [i+3 for i in range(encoding_dim)]
    output_files = [f'../Fig/Latent_Value_{i}.png' for i in range(encoding_dim)]
    print(fig_numbers)
    os.remove('../Fig/Latent_Value_*')
    for k, latent_value in enumerate(latent_values):
        plt.figure(fig_numbers[k], figsize=(10, 6), dpi=300)
        z = df_latent_space[latent_value][np.where(data_nona.z_coor == depth_cluster_result)[0]]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        
        plt.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='jet_r')
        plt.title(titles[k])
        plt.savefig(output_files[k], dpi=300)
else:
    print('The input and output will not be plotted.')

# ============ Clustering  ============ #
clustering_input = latent_space_output
encoding_dim = latent_space_output.shape[1]
df_latent_space = pd.DataFrame(latent_space_output, columns=[f'zz{i}' for i in range(encoding_dim)])
encoded_data_xy = df_latent_space
print(' ==================== Clustering Inputs ==================== ')
print(encoded_data_xy)
print(' =========================================================== ')

if cluster_method == 'K-means':
    # K-means
    from sklearn.cluster import KMeans
    if evaluation_of_num_cluster == 'yes':
        wcss = []
        for i in range(1, max_clusters+1):
            print(str(i) + '/' + str(max_clusters))
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(encoded_data_xy)
            wcss.append(kmeans.inertia_)
        # 繪製WCSS曲線
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.plot(range(1, max_clusters+1), wcss, marker='o')
        plt.title('WCSS for K-means with different number of clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('../Fig/kmeans_num_cluster.png', dpi=300)
        print('finish the evaluation of clusters number.')
    kmeans = KMeans(n_clusters=kmean_clusters)
    kmeans.fit(encoded_data_xy)
    labels = kmeans.predict(encoded_data_xy)
 
if cluster_method == 'GMM':
    # GMM
    from sklearn.mixture import GaussianMixture
    if evaluation_of_num_cluster == 'yes':
        bic = []
        aic = []
        for i in range(1, max_clusters+1):
            print(str(i) + '/' + str(max_clusters))
            gmm = GaussianMixture(n_components=i)
            gmm.fit(encoded_data_xy)
            bic.append(gmm.bic(encoded_data_xy))
            aic.append(gmm.aic(encoded_data_xy))

        # 繪製BIC曲線
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.plot(range(1, max_clusters+1), bic, marker='o')
        plt.title('BIC for GMM with different number of clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('BIC')
        plt.savefig('../Fig/gmm_bic_num_cluster.png', dpi=300)

        # 繪製AIC曲線
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.plot(range(1, max_clusters+1), aic, marker='o')
        plt.title('AIC for GMM with different number of clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('AIC')
        plt.savefig('../Fig/gmm_aic_num_cluster.png', dpi=300)
        print('finish the evaluation of clusters number.')
    gmm = GaussianMixture(n_components=gmm_clusters)
    gmm.fit(encoded_data_xy)
    labels = gmm.predict(encoded_data_xy)

if cluster_method == 'VQ':
    def lbg_algorithm(data, num_clusters, epsilon=1e-6):
        # Initialize the centroids
        data = data.values  # Convert to NumPy array
        print(data)
        centroids = np.mean(data, axis=0).reshape(1, -1)
        print(f"Initial centroid: {centroids}")
        while centroids.shape[0] < num_clusters:
            # Split centroids
            
            centroids = np.vstack([centroids * (1 + epsilon), centroids * (1 - epsilon)])
            print(f"Centroids after splitting: {centroids}")
            prev_error = float('inf')
            
            while True:
                # Assign clusters
                distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
                closest_centroids = np.argmin(distances, axis=1)
                new_centroids = np.array([data[closest_centroids == k].mean(axis=0) for k in range(centroids.shape[0])])
                
                # Compute error
                error = np.linalg.norm(new_centroids - centroids)
                print(f"Error: {error}")
                
                if abs(prev_error - error) < epsilon:
                    break
                
                centroids = new_centroids
                prev_error = error
            print(centroids.shape[0])
        return centroids, closest_centroids

    centroids, labels = lbg_algorithm(encoded_data_xy, vq_clusters)
    print(f"Final centroids: {centroids}")
    print(f"Labels: {labels}")

print('Clustering Completed')
# ============ Clustering Result Visualization ============ #
matplotlib.rcParams['font.family'] = 'Liberation Sans'
matplotlib.rcParams['font.size'] = 15
colors_cluster_all = cm.Set3.colors
colors_cluster = colors_cluster_all[0:vq_clusters]
cmap_cluster = mcolors.ListedColormap(colors_cluster)
z = labels[np.where(data_nona.z_coor==depth_cluster_result)[0]]
x = data_nona.x_coor[data_nona.z_coor==depth_cluster_result]
y = data_nona.y_coor[data_nona.z_coor==depth_cluster_result]
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
plt.figure(figsize=(10, 6), dpi=300)
plt.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap=cmap_cluster)
plt.title('Clustering Result at the depth of ' + str(depth_cluster_result) + ' km ')
plt.savefig('../Fig/Clustering_Result_' + str(depth_cluster_result) + '.png', dpi=300)
print('Figure saved')

# ============ Data Clean  ============ #
data = {
    'Vp': data_xy.x,
    'Ohm': data_xy.y,
    'Ohm_Vp_ratio': data_xy.z,
    'Clusters': labels,
    'YY': data_nona.y_coor,
    'XX': data_nona.x_coor,
    'ZZ': data_nona.z_coor,
    'Vp_ori': data_nona.Vp_value,
    'MT_ori': data_nona.MT_Value,
    'Vpt_ori': data_nona.Vpt_value,
    'clustering_method': np.array([cluster_method] * len(data_nona))

}
df = pd.DataFrame(data)
df.to_csv('../cluster_results.csv', index=False)

# ========================================================== #
# =================== NC File Generation =================== # 
# ========================================================== #
# Here the code transfers the output file of TomoFlex (e.g., vpvstommo.dat) to NetCDF file #
data = df
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
        # 如果 row 的長度小於 nx，則在其末尾添加 nan
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
        # 如果 row 的長度小於 nx，則在其末尾添加 nan
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
        # 如果 row 的長度小於 nx，則在其末尾添加 nan
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
        # 如果 row 的長度小於 nx，則在其末尾添加 nan
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
ncout.close();

# %%
