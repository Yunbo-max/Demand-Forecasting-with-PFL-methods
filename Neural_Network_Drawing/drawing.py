# -*- coding = utf-8 -*-
# @time:30/07/2023 01:17
# Author:Yunbo Long
# @File:drawing.py
# @Software:PyCharm
import torch
from torchsummary import summary
from graphviz import Digraph
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import flwr as fl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Load and preprocess your dataset
# Modify this part according to your dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns

import lightgbm as lgb
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import svm, metrics, tree, preprocessing, linear_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, recall_score, confusion_matrix, f1_score, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import wandb
import matplotlib.pyplot as plt

# Hiding the warnings
warnings.filterwarnings('ignore')
# Hiding the warnings
warnings.filterwarnings('ignore')


# # Initialize Weights and Biases
# wandb.init(project="CNN", name=f"Sheet_{random_sheet_name}")





import h5py


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

# Print the value for a given key
key = 3
value = region_map[key]
print(value)

sheet_name = '8'

value = region_map[float(sheet_name)]
config = {"region": value}

# Read the dataset using the current sheet name
dataset = file[sheet_name][:]
dataset = pd.DataFrame(dataset)

# Read the column names from the attributes
column_names = file[sheet_name].attrs['columns']

# Assign column names to the dataset
dataset.columns = column_names

dataset = dataset.drop(columns=['Region Index'])

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(dataset.head(30))

# Preprocess the data
train_data = dataset
xs = train_data.drop(['Sales'], axis=1)
ys = dataset['Sales']
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
xs_train = scaler.fit_transform(xs_train)
xs_test = scaler.transform(xs_test)

train_inputs = torch.tensor(xs_train, dtype=torch.float32)
train_targets = torch.tensor(ys_train.values, dtype=torch.float32)
test_inputs = torch.tensor(xs_test, dtype=torch.float32)
test_targets = torch.tensor(ys_test.values, dtype=torch.float32)

print(train_inputs.shape)
print(train_targets.shape)


# Define the dataset class
class RegressionDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# Create the dataset objects
train_dataset = RegressionDataset(train_inputs, train_targets)
test_dataset = RegressionDataset(test_inputs, test_targets)

# Create the data loaders
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)



# Create the model
# Create the model
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


# Create the model and move it to the device (GPU if available, otherwise CPU)
model = Net(input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
learning_rate = 0.005
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a dummy input tensor and move it to the same device as the model
dummy_input = torch.randn(1, input_neurons).to(device)

# Print model summary
summary(model, input_size=(input_neurons,))

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchviz import make_dot
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchviz import make_dot
from graphviz import Digraph


# Create a dummy input tensor to determine the summary
dummy_input = torch.randn(1, input_neurons)

# Print model summary
summary(model, input_size=(input_neurons,))

# Create a dummy model with the same architecture as the original model
# This dummy model is just for visualization purposes, it will not be trained
dummy_model = Net(input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout)

# Create a graph of the model
graph = make_dot(dummy_model(dummy_input), params=dict(dummy_model.named_parameters()))

# Customize the appearance of the graph
graph.format = 'png'
graph.attr(bgcolor='white')  # Set the background color to white

# Save the graph to a file
graph.render("neural_network_structure_white_bg", cleanup=True)




# Create the model (assuming the Net class is defined)
model = Net(input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout)

# Move the model to the same device as the dummy input tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a dummy input tensor to determine the summary
dummy_input = torch.randn(1, input_neurons).to(device)

# Print model summary
summary(model, input_size=(input_neurons,), device=device)
