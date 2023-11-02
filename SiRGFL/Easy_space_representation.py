# -*- coding = utf-8 -*-
# @time:04/08/2023 11:09
# Author:Yunbo Long
# @File:Easy_space_representation.py
# @Software:PyCharm
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
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn import svm,metrics,tree,preprocessing,linear_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import Ridge,LinearRegression,LogisticRegression,ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingRegressor,BaggingClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,mean_squared_error,recall_score,confusion_matrix,f1_score,roc_curve, auc
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
import wandb
import matplotlib.pyplot as plt

# Hiding the warnings
warnings.filterwarnings('ignore')
# Hiding the warnings
warnings.filterwarnings('ignore')



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
import wandb

# Open the HDF5 file
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import h5py

# Set the number of clients, rounds, and epochs
sheet_name = ['0', '1', '2', '3', '5', '6', '7', '9', '10', '12', '14', '16', '17', '22']
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
# Set the parameters for the model
input_neurons = 25
output_neurons = 1
hidden_layers = 4
neurons_per_layer = 64
dropout = 0.3

# Initialize a shared global model
# Initialize a shared global model
global_model = Net(input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout)

# Open the HDF5 file
file = h5py.File('/Users/yunbo-max/Desktop/Personalised_FL/Demand-Forecasting-with-PFL-methods/Explotary data analysis/Dataset and preprocessing/market_data.h5', 'r')

# Get the number of clients from sheet_name
num_clients = len(sheet_name)

# Initialize empty lists to store client models and weights
client_models = []

# Initialize a dictionary to store R-squared scores
r2_scores = {client: [] for client in sheet_name}

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


        class Net(nn.Module):
            def __init__(self, input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout):
                super().__init__()

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

        # Calculate R-squared score for the client
        r2 = r2_score(test_targets.numpy(), test_preds.numpy())

        # Save the R-squared score
        r2_scores[client].append(r2)

    # Average the weights across all clients after each round
    averaged_weights = {k: sum(d[k] for d in client_models) / num_clients for k in client_models[0].keys()}

    # Update the global model
    global_model.load_state_dict(averaged_weights)



# Print the R-squared scores for all clients
for client, scores in r2_scores.items():
    print(f"Client: {client}, R-squared scores: {scores}")

plt.figure(figsize=(10,6))

# Map client IDs to region names
for client, scores in r2_scores.items():
    region_name = region_map[int(client)]
    plt.plot(range(1, num_rounds+1), scores, label=region_name)

plt.title("R-squared scores for all regions over rounds")
plt.xlabel("Round")
plt.ylabel("R-squared score")
plt.legend(loc='best')  # Add a legend
plt.show()





import numpy as np

r2_array = np.array([r2_scores[client] for client in sheet_name])
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
# Perform clustering on the R-squared scores
kmeans = KMeans(n_clusters=6)  # change K=5
clusters = kmeans.fit_predict(r2_array)

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(r2_array, clusters)
# print this
# Print the Silhouette Score
print("Silhouette Score:", silhouette_avg)

# Assign clusters back to the regions
clustered_regions = {cluster: [] for cluster in set(clusters)}
for client, cluster in zip(r2_scores.keys(), clusters):
    region_name = region_map[int(client)]
    clustered_regions[cluster].append(region_name)

# Print out the clusters and the regions in each cluster
for cluster, regions in clustered_regions.items():
    print(f"Cluster {cluster}: {regions}")

# Create a 3D scatter plot of the R-squared values with different clusters represented by different colors
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']  # add more colors for additional clusters
for cluster, color in zip(set(clusters), colors):
    ax.scatter(r2_array[clusters==cluster, 0], r2_array[clusters==cluster, 1], r2_array[clusters==cluster, 2], c=color, label=f'Cluster {cluster}')
ax.legend()

ax.set_xlabel("Round 1 R-squared")
ax.set_ylabel("Round 2 R-squared")
ax.set_zlabel("Round 3 R-squared")
plt.savefig('3d_plot.png', dpi=300, bbox_inches='tight')
plt.show()




# Create a 3D scatter plot of the R-squared values with different clusters represented by different colors
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')


colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']  # add more colors for additional clusters
for cluster, color in zip(set(clusters), colors):
    region_indices = np.where(clusters == cluster)[0]
    cluster_r2 = r2_array[clusters == cluster]
    cluster_regions = [region_map[int(sheet_name[idx])] for idx in region_indices]
    ax.scatter(cluster_r2[:, 0], cluster_r2[:, 1], cluster_r2[:, 2], c=color, label=f'Cluster {cluster}')
    for region, r2 in zip(cluster_regions, cluster_r2):
        ax.text(r2[0], r2[1], r2[2], region, fontsize=8)

ax.legend()
ax.set_xlabel("Round 1 R-squared")
ax.set_ylabel("Round 2 R-squared")
ax.set_zlabel("Round 3 R-squared")

plt.savefig('3d_plot_text.png', dpi=300, bbox_inches='tight')


plt.show()

#
# import plotly.graph_objects as go
#
# fig = go.Figure(data=[go.Scatter3d(
#     x=cluster_r2[:, 0],
#     y=cluster_r2[:, 1],
#     z=cluster_r2[:, 2],
#     mode='markers',
#     marker=dict(
#         size=3,
#         color=color,
#         opacity=0.8
#     )
# ) for cluster, color in zip(set(clusters), colors)])
#
# fig.update_layout(
#     scene=dict(
#         xaxis_title="Round 1 R-squared",
#         yaxis_title="Round 2 R-squared",
#         zaxis_title="Round 3 R-squared"
#     )
# )
#
# # Save the figure as an interactive HTML file
# fig.write_html('3d_plot.html')
#
