# -*- coding = utf-8 -*-
# @time:04/08/2023 10:58
# Author:Yunbo Long
# @File:similarity_calculation.py
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


# Define the neural network model
class Net(nn.Module):
    def __init__(self, input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout):
        super(Net, self).__init__()

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_neurons, neurons_per_layer))
        self.layers.append(nn.ReLU())

        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))

        self.layers.append(nn.Linear(neurons_per_layer, output_neurons))

    def forward(self, x):
        x = x.view(-1, self.input_neurons)
        for layer in self.layers:
            x = layer(x)
        return x


# Set the parameters for the model
input_neurons = 25
output_neurons = 1
hidden_layers = 4
neurons_per_layer = 64
dropout = 0.3

# Initialize a shared global model
global_model = Net(input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout)

# Open the HDF5 file
file = h5py.File('E:\Federated_learning_flower\experiments\Presentation\market_data.h5', 'r')

# Get the number of clients from sheet_name
num_clients = len(sheet_name)



# Set the number of iterations for federated learning
num_iterations = 3

# Initialize an empty similarity matrix to store similarity values for each pair of clients
similarity_matrix_total1 = np.zeros((len(sheet_name), len(sheet_name)))
similarity_matrix_total2 = np.zeros((len(sheet_name), len(sheet_name)))
similarity_matrix_total3 = np.zeros((len(sheet_name), len(sheet_name)))

# Initialize a dictionary to store metrics for each client and each iteration
# Initialize a dictionary to store metrics for each client and each iteration
# Initialize a dictionary to store metrics for each client, round, and iteration
# Initialize an empty similarity matrix to store similarity values for each pair of clients for each iteration
similarity_matrix_total = np.zeros((len(sheet_name), len(sheet_name), num_iterations))

# Initialize a dictionary to store metrics for each client, round, and iteration
metrics = {client: {'r2': [[[] for _ in range(num_rounds)] for _ in range(num_iterations)]} for client in sheet_name}

# Initialize a list to store the feature matrices for each iteration
all_feature_matrices = []

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")

    # Initialize a shared global model
    global_model = Net(input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout)

    # Perform federated learning
    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")

        # Initialize an empty list to store the client models for this round
        client_models = []

        for client in sheet_name:
            # Load the state dict of the global model to the client model
            model = Net(input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout)
            model.load_state_dict(global_model.state_dict())

            dataset = file[client][:]
            dataset = pd.DataFrame(dataset)

            # Read the column names from the attributes
            column_names = file[client].attrs['columns']

            # Assign column names to the dataset
            dataset.columns = column_names

            dataset = dataset.drop(columns=['Region Index'])

            # Preprocess the data
            train_data = dataset
            xs = train_data.drop(['Sales'], axis=1)
            ys = train_data['Sales']
            xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.3, random_state=42)

            # Split training set into training and validation sets
            xs_train, xs_val, ys_train, ys_val = train_test_split(xs_train, ys_train, test_size=0.2, random_state=42)

            # Convert data to tensors
            train_inputs = torch.tensor(xs_train.values, dtype=torch.float32)
            train_targets = torch.tensor(ys_train.values, dtype=torch.float32)
            val_inputs = torch.tensor(xs_val.values, dtype=torch.float32)
            val_targets = torch.tensor(ys_val.values, dtype=torch.float32)
            test_inputs = torch.tensor(xs_test.values, dtype=torch.float32)
            test_targets = torch.tensor(ys_test.values, dtype=torch.float32)

            # Create data loaders
            train_dataset = MarketDataset(train_inputs, train_targets)
            val_dataset = MarketDataset(val_inputs, val_targets)
            test_dataset = MarketDataset(test_inputs, test_targets)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)

            # Define the neural network model
            class Net(nn.Module):
                def __init__(self, input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout):
                    super(Net, self).__init__()

                    self.input_neurons = input_neurons
                    self.output_neurons = output_neurons
                    self.hidden_layers = hidden_layers
                    self.neurons_per_layer = neurons_per_layer
                    self.dropout = dropout

                    self.layers = nn.ModuleList()
                    self.layers.append(nn.Linear(input_neurons, neurons_per_layer))
                    self.layers.append(nn.ReLU())

                    for _ in range(hidden_layers):
                        self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
                        self.layers.append(nn.ReLU())
                        self.layers.append(nn.Dropout(p=dropout))

                    self.layers.append(nn.Linear(neurons_per_layer, output_neurons))

                def forward(self, x):
                    x = x.view(-1, self.input_neurons)
                    for layer in self.layers:
                        x = layer(x)
                    return x


            input_neurons = train_inputs.shape[1]
            output_neurons = 1
            hidden_layers = 4
            neurons_per_layer = 64
            dropout = 0.3
            model = Net(input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout)

            criterion = nn.MSELoss()
            learning_rate = 0.005
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_losses = []
            val_losses = []
            train_rmse_list = []
            val_rmse_list = []
            mae_train_list = []
            rmse_train_list = []
            mape_train_list = []
            mse_train_list = []
            r2_train_list = []
            mae_val_list = []
            rmse_val_list = []
            mape_val_list = []
            mse_val_list = []
            r2_val_list = []
            mae_test_list = []
            rmse_test_list = []
            mape_test_list = []
            mse_test_list = []
            r2_test_list = []

            for epoch in range(num_epochs):
                train_losses = []
                model.train()

                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                epoch_loss = np.mean(train_losses)

            # Use model to generate predictions for the test dataset
            client_models.append(model.state_dict())

            # Use model to generate predictions for the test dataset
            model.eval()
            with torch.no_grad():
                test_preds = model(test_inputs)

            r2 = r2_score(test_targets.numpy(), test_preds.numpy())

            # Save the R2 value for the current round and iteration
            # Save the R2 value for the current round and iteration
            metrics[client]['r2'][iteration][round] = r2




        # Average the weights across all clients after each round
        averaged_weights = {k: sum(d[k] for d in client_models) / num_clients for k in client_models[0].keys()}

        # Update the global model
        global_model.load_state_dict(averaged_weights)

    # Create the feature matrix for the current iteration and all rounds
    feature_matrix = np.array(
        [[metrics[client]['r2'][iteration][r] for r in range(num_rounds)] for client in sheet_name])

    print(feature_matrix)

    # Check if the feature matrix is empty (no valid R2 values)
    if feature_matrix.size == 0:
        print("No valid data in the feature matrix. Skipping this iteration.")
        continue

    # # Transpose the feature_matrix to have shape (num_clients, num_rounds)
    # feature_matrix = feature_matrix.T
    # print("Feature matrix shape:", feature_matrix.shape)

    # Append the feature matrix to the list after adding an additional dimension
    all_feature_matrices.append(np.expand_dims(feature_matrix, axis=2))

    # Concatenate the feature matrices along the third dimension to have shape (num_clients, num_rounds, num_iterations)
    feature_matrix_total = np.concatenate(all_feature_matrices, axis=2)

    # Step 2: Standardize the Data
    scaler = StandardScaler()

    # Flatten the last two dimensions
    flattened_data = feature_matrix_total.reshape(feature_matrix_total.shape[0], -1)
    normalized_data = scaler.fit_transform(flattened_data)


    # Compute Pairwise Similarity using Sigmoid Kernel for the current iteration
    similarity_matrix_total = sigmoid_kernel(normalized_data)
    print(similarity_matrix_total)

    # Save the similarity matrix for this round
    if iteration == 0:
        similarity_matrix_total1 = similarity_matrix_total
    elif iteration == 1:
        similarity_matrix_total2 = similarity_matrix_total
    elif iteration == 2:
        similarity_matrix_total3 = similarity_matrix_total

# Create the feature matrix by stacking the similarity matrices from all iterations
feature_matrix_total = np.stack([similarity_matrix_total1, similarity_matrix_total2, similarity_matrix_total3], axis=-1)

# Reshape the feature matrix into a 2D array with shape 14x15 (14 clients and 15 data points for each)
feature_matrix_total = np.reshape(feature_matrix_total, (len(sheet_name), -1))
print('feature_matrix_total', feature_matrix_total)

# Save the average similarity matrix to a file
np.save('similarity_matrix_total1_5rounds.npy', similarity_matrix_total1)
np.save('similarity_matrix_total2_5rounds.npy', similarity_matrix_total2)
np.save('similarity_matrix_total3_5rounds.npy', similarity_matrix_total3)

import numpy as np
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np


#
# # Step 4: Perform Hierarchical Clustering
# agglomerative = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
# hierarchical_labels = agglomerative.fit_predict(similarity_matrix)
#
#
# agglomerative = AgglomerativeClustering(affinity='precomputed', linkage='average')
# hierarchical_labels2 = agglomerative.fit_predict(similarity_matrix)
#
#
# # Step 5: Perform DBSCAN Clustering
# similarity_matrix1 = similarity_matrix + np.abs(np.min(similarity_matrix))
# dbscan = DBSCAN(eps=0.3, min_samples=5, metric='precomputed')
# dbscan_labels = dbscan.fit_predict(similarity_matrix1)
#
# # Step 6: Perform Spectral Clustering
# spectral = SpectralClustering(n_clusters=3, affinity='precomputed')
# spectral_labels = spectral.fit_predict(similarity_matrix)
#
# # Step 7: Perform K-means Clustering
# kmeans = KMeans(n_clusters=3)
# kmeans_labels = kmeans.fit_predict(similarity_matrix)
#
# # Compute the ranking of similarities for each client
# similarity_rankings = similarity_matrix.argsort(axis=1)[:, ::-1]
#
# # Print the cluster labels and similarity rankings for each client
# for i, client in enumerate(sheet_name):
#     print(f"Client: {client}")
#     print("Similarity Rankings:")
#     for rank, similarity_index in enumerate(similarity_rankings[i]):
#         if similarity_index != i:  # Exclude similarity with itself
#             region_number = sheet_name[similarity_index]
#             region_name = region_map[int(region_number)]
#             similarity_value = similarity_matrix[i, similarity_index]
#             print(f"Rank {rank + 1}: Region {region_number} - {region_name} (Similarity: {similarity_value})")
#     print()
#
# # Print the clusters with corresponding client numbers and names
# for cluster_label in set(hierarchical_labels):
#     cluster_clients = [client for i, client in enumerate(sheet_name) if hierarchical_labels[i] == cluster_label]
#     print(f"Hierarchical Cluster {cluster_label}:")
#     for client in cluster_clients:
#         region_name = region_map[int(client)]
#         print(f"Client: {client}-{region_name}")
#     print()
#
# # Print the clusters with corresponding client numbers and names
# for cluster_label in set(hierarchical_labels2):
#     cluster_clients = [client for i, client in enumerate(sheet_name) if hierarchical_labels2[i] == cluster_label]
#     print(f"Hierarchical Cluster2 {cluster_label}:")
#     for client in cluster_clients:
#         region_name = region_map[int(client)]
#         print(f"Client: {client}-{region_name}")
#     print()
#
#
#
# for cluster_label in set(dbscan_labels):
#     if cluster_label == -1:  # Ignore noise points in DBSCAN
#         continue
#     cluster_clients = [client for i, client in enumerate(sheet_name) if dbscan_labels[i] == cluster_label]
#     print(f"DBSCAN Cluster {cluster_label}:")
#     for client in cluster_clients:
#         region_name = region_map[int(client)]
#         print(f"Client: {client}-{region_name}")
#     print()
#
# for cluster_label in set(spectral_labels):
#     cluster_clients = [client for i, client in enumerate(sheet_name) if spectral_labels[i] == cluster_label]
#     print(f"Spectral Cluster {cluster_label}:")
#     for client in cluster_clients:
#         region_name = region_map[int(client)]
#         print(f"Client: {client}-{region_name}")
#     print()
#
# # for cluster_label in set(kmeans_labels):
# #     cluster_clients = [client for i, client in enumerate(sheet_name) if kmeans_labels[i] == cluster_label]
# #     print(f"K-means Cluster {cluster_label}:")
# #     for client in cluster_clients:
# #         region_name = region_map[int(client)]
# #         print(f"Client: {client}-{region_name}")
# #     print()
#
#
# # # Calculate Silhouette Score for Hierarchical Clustering
# # hierarchical_silhouette = silhouette_score(normalized_data, hierarchical_labels, metric='precomputed')
# # print(f"Hierarchical Clustering Silhouette Score: {hierarchical_silhouette}")
#
# # # Calculate Silhouette Score for DBSCAN Clustering
# # dbscan_silhouette = silhouette_score(similarity_matrix1, dbscan_labels)
# # print(f"DBSCAN Clustering Silhouette Score: {dbscan_silhouette}")
# #
# # # Calculate Silhouette Score for Spectral Clustering
# # spectral_silhouette = silhouette_score(similarity_matrix, spectral_labels, metric='precomputed')
# # print(f"Spectral Clustering Silhouette Score: {spectral_silhouette}")
#
# # Calculate Silhouette Score for K-means Clustering
# kmeans_silhouette = silhouette_score(similarity_matrix, kmeans_labels)
# print(f"K-means Clustering Silhouette Score: {kmeans_silhouette}")
