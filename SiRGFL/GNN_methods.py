# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-01-24 10:28:47
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-07-04 16:12:39
# -*- coding = utf-8 -*-
# @time:04/08/2023 11:03
# Author:Yunbo Long
# @File:GNN_methods.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @time:15/07/2023 15:46
# Author:Yunbo Long
# @File:test_7.15.py
# @Software:PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

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

class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNNModel, self).__init__()
        self.conv1 = gnn.GCNConv(input_size, hidden_size)
        self.conv2 = gnn.GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def train_mlp_model(inputs, targets, device):
    model = Net(input_neurons=inputs.shape[1], output_neurons=1, hidden_layers=4, neurons_per_layer=64, dropout=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    num_epochs = 10
    batch_size = 32
    losses = []

    train_inputs = inputs.to(device)
    train_targets = targets.to(device)

    for epoch in range(num_epochs):
        permutation = torch.randperm(train_inputs.size()[0])
        batch_losses = []

        for i in range(0, train_inputs.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_inputs, batch_targets = train_inputs[indices], train_targets[indices]

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model

def cluster_regions(embeddings):
    kmeans = KMeans(n_clusters=2)
    cluster_labels = kmeans.fit_predict(embeddings.detach().numpy())
    return cluster_labels

# Load the data
file = h5py.File('E:/Federated_learning_flower/experiments/Presentation/market_data.h5', 'r')

region_map = {
    '0': 'Southeast Asia',
    '1': 'South Asia',
    '2': 'Oceania',
    '3': 'Eastern Asia',
    '4': 'West Asia',
    '5': 'West of USA',
    '6': 'US Center',
    '7': 'West Africa',
    '8': 'Central Africa',
    '9': 'North Africa',
    '10': 'Western Europe',
    '11': 'Northern Europe',
    '12': 'Central America',
    '13': 'Caribbean',
    '14': 'South America',
    '15': 'East Africa',
    '16': 'Southern Europe',
    '17': 'East of USA',
    '18': 'Canada',
    '19': 'Southern Africa',
    '20': 'Central Asia',
    '21': 'Eastern Europe',
    '22': 'South of USA'
}

sheet_names = ['0', '1', '2', '3', '5', '6', '7', '9', '10', '12', '14', '16', '17', '22']
# sheet_names = ['0','1']
embeddings = []

for sheet_name in sheet_names:
    value = region_map[sheet_name]

    # Read the dataset using the current sheet name
    dataset = file[sheet_name][:]
    dataset = pd.DataFrame(dataset)

    # Read the column names from the attributes
    column_names = file[sheet_name].attrs['columns']

    # Assign column names to the dataset
    dataset.columns = column_names

    dataset = dataset.drop(columns=['Region Index'])

    # print(dataset.head(30))

    # Preprocess the data
    train_data = dataset  # Drop the last 30 rows
    xs = train_data.drop(['Sales'], axis=1)
    ys = train_data['Sales']  # Use the updated train_data for ys
    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.3, random_state=42)

    scaler = MinMaxScaler()
    xs_train = scaler.fit_transform(xs_train)
    xs_test = scaler.transform(xs_test)

    train_inputs = torch.tensor(xs_train, dtype=torch.float32)
    train_targets = torch.tensor(ys_train.values, dtype=torch.float32)

    # Train MLP model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp_model = train_mlp_model(train_inputs, train_targets, device)

    # Get MLP embeddings
    mlp_embeddings = mlp_model(torch.tensor(xs.values, dtype=torch.float32).to(device)).detach()

    # Save MLP embeddings
    embeddings.append(mlp_embeddings)

# Construct graph data
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

# Concatenate embeddings from all regions
embeddings = torch.cat(embeddings)

def train_gnn_model(embeddings, edge_index, device):
    model = GNNModel(input_size=embeddings.shape[1], hidden_size=32, output_size=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    num_epochs = 10

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(embeddings.to(device), edge_index.to(device))
        loss = criterion(outputs, embeddings.to(device))
        loss.backward()
        optimizer.step()

    return model

# Train GNN model on the embeddings
gnn_model = train_gnn_model(embeddings, edge_index, device)

# Get GNN embeddings
gnn_embeddings = gnn_model(embeddings.to(device), edge_index.to(device)).detach()

# Move the GNN embeddings back to CPU
gnn_embeddings = gnn_embeddings.cpu()


# from sklearn.metrics import silhouette_score
# # Perform clustering on the GNN embeddings
# cluster_labels = cluster_regions(gnn_embeddings)
#
# # Calculate Silhouette Score
# silhouette_avg = silhouette_score(gnn_embeddings, cluster_labels)
# print(f"Silhouette Score: {silhouette_avg:.4f}")
#
# # Visualize the clusters
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(gnn_embeddings[:, 0], gnn_embeddings[:, 1], gnn_embeddings[:, 2], c=cluster_labels)
# ax.set_xlabel('Embedding Dimension 1')
# ax.set_ylabel('Embedding Dimension 2')
# ax.set_zlabel('Embedding Dimension 3')
# plt.show()

from sklearn.metrics import calinski_harabasz_score, silhouette_score

# Perform clustering on the GNN embeddings
cluster_labels = cluster_regions(gnn_embeddings)

# Calculate the Calinski-Harabasz Index
calinski_score = calinski_harabasz_score(gnn_embeddings, cluster_labels)
print(f"Calinski-Harabasz Index: {calinski_score:.4f}")

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(gnn_embeddings, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Visualize the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gnn_embeddings[:, 0], gnn_embeddings[:, 1], gnn_embeddings[:, 2], c=cluster_labels)
ax.set_xlabel('Embedding Dimension 1')
ax.set_ylabel('Embedding Dimension 2')
ax.set_zlabel('Embedding Dimension 3')
plt.show()

# Print the regions assigned to each cluster
for cluster_id in range(2):
    cluster_regions = [region_map[sheet_names[i]] for i, sheet in enumerate(sheet_names) if cluster_labels[i] == cluster_id]
    print(f"Cluster {cluster_id}: {cluster_regions}")
