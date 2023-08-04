# -*- coding = utf-8 -*-
# @time:04/08/2023 11:00
# Author:Yunbo Long
# @File:Similarity_space_representation.py
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
# Define a custom dataset class
class MarketDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        target = self.targets[idx]
        return input_data, target


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

print(similarity_matrix_total1)
print(similarity_matrix_total2)
print(similarity_matrix_total3)

# Get the number of clients
num_clients = len(sheet_name)

# Initialize an array to store the unique data points (client pairs) and their corresponding similarity values
data_points = []
similarity_values = []

# Collect unique data points and their corresponding similarity values
for i in range(num_clients):
    for j in range(i+1, num_clients):
        data_points.append([similarity_matrix_total1[i, j], similarity_matrix_total2[i, j], similarity_matrix_total3[i, j]])
        similarity_values.append(f'{sheet_name[i]} with {sheet_name[j]}')


# Convert the data points and similarity values to numpy arrays
data_points = np.array(data_points)
similarity_values = np.array(similarity_values)


# Filter the data points with x, y, z all bigger than 0.7
filtered_data_points = data_points[(data_points[:, 0] > 0) & (data_points[:, 1] > 0) & (data_points[:, 2] > 0)]
filtered_similarity_values = similarity_values[(data_points[:, 0] > 0) & (data_points[:, 1] > 0) & (data_points[:, 2] > 0)]

# Perform K-means Clustering on the filtered_data_points
num_clusters = 1
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(filtered_data_points)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each data point in the 3D space
for i in range(len(filtered_data_points)):
    ax.scatter(filtered_data_points[i, 0], filtered_data_points[i, 1], filtered_data_points[i, 2], c='C'+str(labels[i]), marker='o', label=filtered_similarity_values[i])

# Add the diagonal line
ax.plot([0, 1], [0, 1], [0, 1], color='red')  # plots line from (0,0,0) to (1,1,1)

# Set axis labels and plot title
ax.set_xlabel('Similarity Matrix 1')
ax.set_ylabel('Similarity Matrix 2')
ax.set_zlabel('Similarity Matrix 3')
ax.set_title('3D projection of Similarity Matrices(Points with x, y, z > 0.7)')


# Set axis limits to start from 0.7 and end at 1
ax.set_xlim(-0.1, 1)
ax.set_ylim(-0.1, 1)
ax.set_zlim(-0.1, 1)


# # Add legend and set location to the right and two columns
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

plt.show()



# Define the points for the diagonal
A = np.array([0, 0, 0])
B = np.array([1, 1, 1])

# Compute the direction of the diagonal line
diagonal_direction = B - A

# Initialize an array to store the distances of the projections from the origin
distances = []
# Compute the projection of each point onto the diagonal line and their distances from the origin
client_projections = {}
for i, point in enumerate(data_points):
    C = np.array(point)
    P = A + np.dot(C - A, diagonal_direction) / np.linalg.norm(diagonal_direction) ** 2 * diagonal_direction
    distance = np.linalg.norm(P - A)

    client_name = similarity_values[i].split(' with ')[0]  # extract client name
    if client_name not in client_projections:
        client_projections[client_name] = []
    client_projections[client_name].append((similarity_values[i], distance))

# Sort the client pairs based on their distances from the origin (along the diagonal)
for client in client_projections:
    client_projections[client] = sorted(client_projections[client], key=lambda x: x[1], reverse=True)

# Print the grouped and sorted clients
for client, projections in client_projections.items():
    print(f"Client {client}:")
    for projection in projections:
        print(f"  {projection[0]}: {projection[1]}")







from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Normalize the data
scaler = StandardScaler()
normalized_data_points = scaler.fit_transform(filtered_data_points)

# Initialize lists to store the evaluation scores
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

# Define the range of number of clusters to try
num_clusters_range = range(2, 40)



from sklearn.cluster import AgglomerativeClustering

# Initialize lists to store the evaluation scores for Hierarchical Clustering
hierarchical_silhouette_scores = []
hierarchical_davies_bouldin_scores = []
hierarchical_calinski_harabasz_scores = []

for num_clusters in num_clusters_range:
    # Perform Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    hierarchical_labels = hierarchical.fit_predict(normalized_data_points)

    # Calculate evaluation scores
    hierarchical_silhouette_score_value = silhouette_score(normalized_data_points, hierarchical_labels)
    hierarchical_davies_bouldin_score_value = davies_bouldin_score(normalized_data_points, hierarchical_labels)
    hierarchical_calinski_harabasz_score_value = calinski_harabasz_score(normalized_data_points, hierarchical_labels)

    # Store the scores in lists
    hierarchical_silhouette_scores.append(hierarchical_silhouette_score_value)
    hierarchical_davies_bouldin_scores.append(hierarchical_davies_bouldin_score_value)
    hierarchical_calinski_harabasz_scores.append(hierarchical_calinski_harabasz_score_value)

    # Print the scores
    print(f"Hierarchical Clustering Silhouette Score (Number of Clusters = {num_clusters}): {hierarchical_silhouette_score_value}")
    print(f"Hierarchical Clustering Davies-Bouldin Score (Number of Clusters = {num_clusters}): {hierarchical_davies_bouldin_score_value}")
    print(f"Hierarchical Clustering Calinski-Harabasz Score (Number of Clusters = {num_clusters}): {hierarchical_calinski_harabasz_score_value}")

# Plot all the scores for different numbers of clusters
plt.figure(figsize=(10, 8))
plt.plot(num_clusters_range, hierarchical_silhouette_scores, marker='o', label='Hierarchical Silhouette Score')
plt.plot(num_clusters_range, hierarchical_davies_bouldin_scores, marker='o', label='Hierarchical Davies-Bouldin Score')
# plt.plot(num_clusters_range, hierarchical_calinski_harabasz_scores, marker='o', label='Hierarchical Calinski-Harabasz Score')

plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Evaluation Scores for Hierarchical Clustering')
plt.xticks(num_clusters_range)
plt.legend()
plt.grid(True)
plt.show()
