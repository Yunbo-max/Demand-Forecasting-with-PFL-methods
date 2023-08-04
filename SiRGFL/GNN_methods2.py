# -*- coding = utf-8 -*-
# @time:15/07/2023 02:29
# Author:Yunbo Long
# @File:test2.py
# @Software:PyCharm
# import wandb

import wandb
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x



# Open the HDF5 file
file = h5py.File('E:\Federated_learning_flower\experiments\Presentation\market_data.h5', 'r')

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

import numpy as np


def train_mlp_model(inputs, targets):
    model = Net(input_neurons=inputs.shape[1], output_neurons=1, hidden_layers=2, neurons_per_layer=16, dropout=0.3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 5
    batch_size = 32
    losses = []
    train_inputs = inputs
    train_targets = targets
    evaluation_interval = 1  # Set the evaluation interval (e.g., every 5 epochs)

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

def train_gnn_model(embeddings, edge_index):
    model = GNNModel(input_size=embeddings.shape[1], hidden_size=32, output_size=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    num_epochs = 10

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(embeddings, edge_index)
        loss = criterion(outputs, embeddings)
        loss.backward()
        optimizer.step()

    return model

def cluster_regions(embeddings):
    kmeans = KMeans(n_clusters=3)  # Specify number of clusters
    cluster_labels = kmeans.fit_predict(embeddings.detach().numpy())
    return cluster_labels

# sheet_name = ['0','1', '2', '3', '5', '6', '7', '9', '10', '12', '14', '16', '17', '22']
sheet_name = ['0','1']
embeddings = []




for sheet_names in sheet_name:
    value = region_map[float(sheet_names)]
    # config = {"region": value}
    # with wandb.init(project='GNN_finding', config=config) as run:
    # Read the dataset using the current sheet name
    dataset = file[sheet_names][:]
    dataset = pd.DataFrame(dataset)

    # Read the column names from the attributes
    column_names = file[sheet_names].attrs['columns']

    # Assign column names to the dataset
    dataset.columns = column_names

    dataset = dataset.drop(columns=['Region Index'])

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(dataset.head(30))

    # Preprocess the data
    train_data = dataset  # Drop the last 30 rows
    xs = train_data.drop(['Sales'], axis=1)
    ys = train_data['Sales']  # Use the updated train_data for ys
    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.3, random_state=42)

    # xq=train_data.loc[:, train_data.columns != 'Order Item Quantity']
    # yq=train_data['Order Item Quantity']
    # xq_train, xq_test,yq_train,yq_test = train_test_split(xq,yq,test_size = 0.3, random_state = 42)

    scaler = MinMaxScaler()
    xs_train = scaler.fit_transform(xs_train)
    xs_test = scaler.transform(xs_test)
    # xq_train=scaler.fit_transform(xq_train)
    # xq_test=scaler.transform(xq_test)

    train_inputs = torch.tensor(xs_train, dtype=torch.float32)
    train_targets = torch.tensor(ys_train.values, dtype=torch.float32)
    test_inputs = torch.tensor(xs_test, dtype=torch.float32)
    test_targets = torch.tensor(ys_test.values, dtype=torch.float32)

    print(train_inputs.shape)
    print(train_targets.shape)


    # Train MLP model
    mlp_model = train_mlp_model(train_inputs, train_targets)

    # Get MLP embeddings
    mlp_embeddings = mlp_model(torch.tensor(xs.values, dtype=torch.float32)).detach()

    # Save MLP embeddings
    embeddings.append(mlp_embeddings)

# Construct graph data
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)


# Concatenate embeddings from all regions
embeddings = torch.cat(embeddings)

# Train GNN model on the embeddings
gnn_model = train_gnn_model(embeddings, edge_index)

# Get GNN embeddings
gnn_embeddings = gnn_model(embeddings, edge_index).detach()

# Perform clustering on the GNN embeddings
cluster_labels = cluster_regions(gnn_embeddings)

# Visualize the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gnn_embeddings[:, 0], gnn_embeddings[:, 1], gnn_embeddings[:, 2], c=cluster_labels)
ax.set_xlabel('Embedding Dimension 1')
ax.set_ylabel('Embedding Dimension 2')
ax.set_zlabel('Embedding Dimension 3')
plt.show()


from sklearn.cluster import SpectralClustering

# Perform clustering on the GNN embeddings
clustering = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
cluster_labels = clustering.fit_predict(gnn_embeddings.detach().numpy())

# Print the regions assigned to each cluster
for cluster_id in range(3):
    cluster_regions = [region_map[sheet_name[i]] for i, label in enumerate(cluster_labels) if label == cluster_id]
    print(f"Cluster {cluster_id}: {cluster_regions}")



