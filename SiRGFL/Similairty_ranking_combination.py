# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-01-24 10:28:47
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-07-04 16:12:40
# -*- coding = utf-8 -*-
# @time:04/08/2023 10:59
# Author:Yunbo Long
# @File:Similairty_ranking_combination.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @time:28/07/2023 16:37
# Author:Yunbo Long
# @File:ISMM_matrix_kmeans.py
# @Software:PyCharm
import warnings
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import sigmoid_kernel
# Hiding the warnings
warnings.filterwarnings('ignore')
# Hiding the warnings
warnings.filterwarnings('ignore')


# # Initialize Weights and Biases
# wandb.init(project="CNN", name=f"Sheet_{random_sheet_name}")


import wandb

import h5py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import svm, metrics, tree, preprocessing, linear_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, recall_score, confusion_matrix, f1_score, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Hiding the warnings
warnings.filterwarnings('ignore')

import h5py
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import numpy as np
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from scipy.spatial import distance

# Set the number of clients, rounds, and epochs
# sheet_name = ['0', '1', '2']
sheet_name = ['0', '1', '2', '3', '5', '6', '7', '9', '10', '12','14','16','17','22']
num_rounds = 3
num_epochs = 10

region_map = {
    0: 'Southeast Asia',
    1: 'South Asia',
    2: 'Oceania',
    3: 'Eastern Asia',
    4: 'West Asia',
    5: 'West of USA',
    6: 'US Center',
    7: 'West Africa',
    8: 'Central Africa',
    9: 'North Africa',
    10: 'Western Europe',
    11: 'Northern Europe',
    12: 'Central America',
    13: 'Caribbean',
    14: 'South America',
    15: 'East Africa',
    16: 'Southern Europe',
    17: 'East of USA',
    18: 'Canada',
    19: 'Southern Africa',
    20: 'Central Asia',
    21: 'Eastern Europe',
    22: 'South of USA'
}

# Step 3: Compute Pairwise Similarity using Sigmoid Kernel
import numpy as np

# Load the similarity matrices
similarity_matrix_total1 = np.load('similarity_matrix_total1_5rounds.npy')
similarity_matrix_total2 = np.load('similarity_matrix_total2_5rounds.npy')
similarity_matrix_total3 = np.load('similarity_matrix_total3_5rounds.npy')

# print(similarity_matrix_total1)
# print(similarity_matrix_total2)
# print(similarity_matrix_total3)

import numpy as np

# Assuming similarity_matrix_total1, similarity_matrix_total2, and similarity_matrix_total3 are the three similarity matrices

# Stack the three similarity matrices along the third dimension to get a single 3D array
stacked_matrices = np.stack([similarity_matrix_total1, similarity_matrix_total2, similarity_matrix_total3], axis=-1)

# Define the projection vector
projection_vector = np.array([1, 1, 1])

# Calculate the projection values for each point in the 14x14 map
projection_values = np.dot(stacked_matrices, projection_vector)

# Get the new 14x14 matrix showing the projection values
new_matrix = projection_values.reshape(14, 14)

# Print the new matrix
print(new_matrix)


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(10, 8))

font_size = 12

cbar_kws = {"shrink": .14, 'label': 'Projection Values'}  # size of colorbar
cbar_kws['labelsize'] = font_size


# Assuming new_matrix is the 14x14 matrix obtained from the previous code
# Set the diagonal values to 0
np.fill_diagonal(new_matrix, 0)
# Define the region names as column and row names following sheet_name
regions_sheet_name = [f'Sheet_{name}' for name in sheet_name]

# Map the sheet names to their corresponding real names using region_map
regions_real_name = [region_map[int(name.split('_')[1])] for name in regions_sheet_name]

# Create a DataFrame with the new_matrix and real region names
df_heatmap = pd.DataFrame(new_matrix, columns=regions_real_name, index=regions_real_name)

# Create the heatmap using seaborn
plt.figure(figsize=(16, 16))
annot_kws = {"size": font_size}
ax = sns.heatmap(df_heatmap, annot=True, cmap='YlGnBu', center=0, fmt=".3f", cbar_kws={'label': 'Projection Values'},annot_kws = annot_kws)
# set font size of the tick labels
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
ax.set_title('Projection of Similarity Matrices onto (1, 1, 1)', fontsize=font_size)  # set font size for title

# set font size for tick labels
ax.set_xticklabels(ax.get_xticklabels(), fontsize=font_size)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)

# Adjust colorbar labels
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=font_size)

# Adjust the position of the heatmap upwards
plt.subplots_adjust(top=0.97, bottom=0.2, left=0.12, right=0.95)

# Show the plot
plt.show()







# # Step 1: Cluster the regions using KMeans clustering algorithm
# num_clusters = 3  # Choose the number of clusters based on the desired size of the smaller matrix
# kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# clusters = kmeans.fit_predict(new_matrix)
#
# # Step 2: Calculate the average similarity value for each cluster
# cluster_avg_similarity = []
# for cluster_idx in range(num_clusters):
#     cluster_regions = np.where(clusters == cluster_idx)[0]
#     avg_similarity = np.mean(new_matrix[cluster_regions, :][:, cluster_regions])
#     cluster_avg_similarity.append(avg_similarity)
#
# # Step 3: Select the cluster with the highest average similarity value
# selected_cluster_idx = np.argmax(cluster_avg_similarity)
# selected_cluster_regions = np.where(clusters == selected_cluster_idx)[0]
#
# # Step 4: Create the smaller heat map matrix using the selected cluster
# smaller_heatmap_matrix = new_matrix[selected_cluster_regions, :][:, selected_cluster_regions]
#
# # Visualize the smaller heat map matrix
# selected_regions_real_name = [regions_real_name[idx] for idx in selected_cluster_regions]
# df_smaller_heatmap = pd.DataFrame(smaller_heatmap_matrix, columns=selected_regions_real_name, index=selected_regions_real_name)
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(df_smaller_heatmap, annot=True, cmap='YlGnBu', center=0, fmt=".2f", cbar_kws={'label': 'Projection Values'})
# plt.xlabel('Regions')
# plt.ylabel('Regions')
# plt.title('Smaller Heat Map Matrix with Highest Average Similarity')
#
# # Adjust the position of the heatmap upwards
# plt.subplots_adjust(top=0.95, bottom=0.15, left=0.12, right=0.95)
#
# # Show the plot
# plt.show()

#
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming new_matrix is the 14x14 matrix obtained from the previous code

# Set the diagonal values to 0
np.fill_diagonal(new_matrix, 0)
#
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Assuming new_matrix is the 14x14 matrix obtained from the previous code
#
# # Define the region names as column and row names following sheet_name
# regions_sheet_name = [f'Sheet_{name}' for name in sheet_name]
#
# # Map the sheet names to their corresponding real names using region_map
# regions_real_name = [region_map[int(name.split('_')[1])] for name in regions_sheet_name]
#
# # Create a DataFrame with the new_matrix and real region names
# df_heatmap = pd.DataFrame(new_matrix, columns=regions_real_name, index=regions_real_name)
#
# # Set the diagonal values to 0
# np.fill_diagonal(df_heatmap.values, 0)
#
# # Get unique values from the new_matrix
# unique_values = np.unique(df_heatmap.values)
#
# # Sort the unique values in descending order
# unique_values_sorted = np.sort(unique_values)[::-1]
#
# # Plot the matrices one by one from highest to lowest, skipping matrices with only one value
# for idx, value in enumerate(unique_values_sorted):
#     # Get the indices of the matrix with the current value
#     rows, cols = np.where(df_heatmap.values == value)
#
#     # If the matrix has only one value, skip it
#     if len(rows) == 1:
#         continue
#
#     # Create a DataFrame with the current matrix
#     current_df = pd.DataFrame(df_heatmap.values[rows][:, cols], columns=[regions_real_name[c] for c in cols],
#                               index=[regions_real_name[r] for r in rows])
#
#     # Plot the current matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(current_df, annot=True, cmap='YlGnBu', center=0, fmt=".2f", cbar_kws={'label': 'Projection Values'})
#     plt.xlabel('Regions')
#     plt.ylabel('Regions')
#     plt.title(f'Projection of Similarity Matrix onto (1, 1, 1) for Value {value}')
#     plt.show()








#
#
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from itertools import combinations
#
# # Assuming new_matrix is the 14x14 matrix obtained from the previous code
#
# # Define the region names as column and row names following sheet_name
# regions_sheet_name = [f'Sheet_{name}' for name in sheet_name]
#
# # Map the sheet names to their corresponding real names using region_map
# regions_real_name = [region_map[int(name.split('_')[1])] for name in regions_sheet_name]
#
# # Create a DataFrame with the new_matrix and real region names
# df_heatmap = pd.DataFrame(new_matrix, columns=regions_real_name, index=regions_real_name)
#
# # Set the diagonal values to 0
# np.fill_diagonal(df_heatmap.values, 0)
#
# # Get unique values from the new_matrix excluding 0
# unique_values = np.unique(df_heatmap.values[df_heatmap.values != 0])
#
# # Sort the unique values in descending order
# unique_values_sorted = np.sort(unique_values)[::-1]
#
# # Initialize a list to store the selected heat maps and their average values
# selected_heatmaps = []
#
# # Calculate all the possibilities of the smaller matrix and store them in the selected_heatmaps list
# for num_regions in range(2, len(regions_real_name) + 1):
#     # Get all combinations of regions with the current number of regions
#     region_combinations = combinations(range(len(regions_real_name)), num_regions)
#
#     # Calculate the average value for each combination and store it along with the combination itself
#     for combination in region_combinations:
#         # Exclude 0 values when calculating the average
#         avg_value = np.mean(df_heatmap.values[np.ix_(combination, combination)][df_heatmap.values[np.ix_(combination, combination)] != 0])
#         selected_heatmaps.append((combination, avg_value))
#
# # Sort the selected heatmaps based on average value in descending order
# selected_heatmaps.sort(key=lambda x: x[1], reverse=True)
#
# # Plot the top 10 selected heat maps
# for idx, (combination, avg_value) in enumerate(selected_heatmaps[:50]):
#     # Create a DataFrame with the current combination
#     current_df = pd.DataFrame(df_heatmap.values[np.ix_(combination, combination)], columns=[regions_real_name[c] for c in combination],
#                               index=[regions_real_name[r] for r in combination])
#
#     font_size = 12
#
#     cbar_kws = {"shrink": .14, 'label': 'Projection Values'}  # size of colorbar
#     cbar_kws['labelsize'] = font_size
#
#     # Create the heatmap using seaborn
#     plt.figure(figsize=(8, 8))
#     annot_kws = {"size": font_size}
#
#     ax = sns.heatmap(current_df, annot=True, cmap='YlGnBu', center=0, fmt=".2f", cbar_kws={'label': 'Projection Values'},annot_kws=annot_kws)
#
#     # set font size of the tick labels
#     ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
#     ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
#     ax.set_title(f'Projection of Similarity Matrix onto (1, 1, 1) for Combination {combination} (Average Value: {avg_value:.2f})', fontsize=font_size)  # set font size for title
#
#     # set font size for tick labels
#     ax.set_xticklabels(ax.get_xticklabels(), fontsize=font_size)
#     ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size,rotation=0)
#
#     # Adjust colorbar labels
#     cbar = ax.collections[0].colorbar
#     cbar.ax.tick_params(labelsize=font_size)
#
#     # Adjust the position of the heatmap upwards
#     plt.subplots_adjust(top=0.9, bottom=0.2, left=0.12, right=0.95)
#
#
#
#     plt.show()


#
#
#
#
#
import numpy as np
from itertools import combinations

# Assuming new_matrix is the 14x14 matrix obtained from the previous code

# Define the region names as column and row names following sheet_name
regions_sheet_name = ['Sheet_0', 'Sheet_1', 'Sheet_2', 'Sheet_3', 'Sheet_4', 'Sheet_5', 'Sheet_6', 'Sheet_7',
                      'Sheet_8', 'Sheet_9', 'Sheet_10', 'Sheet_11', 'Sheet_12', 'Sheet_13']

region_map = {
    0: 'Southeast Asia',
    1: 'South Asia',
    2: 'Oceania',
    3: 'Eastern Asia',
    4: 'West of USA',
    5: 'US Center',
    6: 'West Africa',
    7: 'North Africa',
    8: 'Western Europe',

    9: 'Central America',

    10: 'South America',

    11: 'Southern Europe',
    12: 'East of USA',

    13: 'South of USA'
}

# Set the diagonal values to 0
np.fill_diagonal(new_matrix, 0)

def region_name_to_index(region_name):
    for idx, name in enumerate(regions_sheet_name):
        if region_map[int(name.split('_')[1])] == region_name:
            return idx
    raise ValueError("Region name not found in the region map.")

def generate_top_10_smaller_matrices(region_name, matrix, top_k=10):
    region_idx = region_name_to_index(region_name)

    # Get the number of regions
    num_regions = matrix.shape[0]

    # Initialize a list to store the selected heat maps and their average values
    selected_heatmaps = []

    # Calculate all the possibilities of the smaller matrix and store them in the selected_heatmaps list
    for num_regions_in_combination in range(2, num_regions):
        # Get all combinations of regions with the current number of regions
        region_combinations = combinations(range(num_regions), num_regions_in_combination)

        # Calculate the average value for each combination and store it along with the combination itself
        for combination in region_combinations:
            if region_idx not in combination:
                continue
            # Exclude 0 values when calculating the average
            avg_value = np.mean(matrix[np.ix_(combination, combination)][matrix[np.ix_(combination, combination)] != 0])
            selected_heatmaps.append((combination, avg_value))

    # Sort the selected heatmaps based on average value in descending order
    selected_heatmaps.sort(key=lambda x: x[1], reverse=True)

    # Return the top k selected heatmaps
    return selected_heatmaps[:top_k]

# Example usage:
region_name = 'East of USA'
top_10_smaller_matrices = generate_top_10_smaller_matrices(region_name, new_matrix)

for idx, (combination, avg_value) in enumerate(top_10_smaller_matrices):
    # Create a DataFrame with the current combination
    current_df = pd.DataFrame(new_matrix[np.ix_(combination, combination)], columns=[region_map[c] for c in combination],
                              index=[region_map[r] for r in combination])

    font_size = 12

    cbar_kws = {"shrink": .14, 'label': 'Projection Values'}  # size of colorbar
    cbar_kws['labelsize'] = font_size

    # Create the heatmap using seaborn
    plt.figure(figsize=(8, 8))
    annot_kws = {"size": font_size}

    ax = sns.heatmap(current_df, annot=True, cmap='YlGnBu', center=0, fmt=".3f", cbar_kws={'label': 'Projection Values'},annot_kws=annot_kws)

    # set font size of the tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    ax.set_title(
        f'Top {idx+1}: Projection of Similarity Matrix onto (1, 1, 1) for Combination {combination} (Average Value: {avg_value:.2f})',
        fontsize=font_size)  # set font size for title

    # set font size for tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=font_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size, rotation=0)

    # Adjust colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)

    # Adjust the position of the heatmap upwards
    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.12, right=0.95)

    plt.show()


    # plt.figure(figsize=(8, 6))
    # sns.heatmap(current_df, annot=True, cmap='YlGnBu', center=0, fmt=".3f", cbar_kws={'label': 'Projection Values'})
    # plt.xlabel('Regions')
    # plt.ylabel('Regions')
    # plt.title(f'Top {idx+1}: Projection of Similarity Matrix onto (1, 1, 1) for Combination {combination} (Average Value: {avg_value:.2f})')
    # plt.show()


