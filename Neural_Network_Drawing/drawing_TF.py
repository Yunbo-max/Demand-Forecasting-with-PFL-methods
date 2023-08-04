# -*- coding = utf-8 -*-
# @time:30/07/2023 01:17
# Author:Yunbo Long
# @File:drawing.py
# @Software:PyCharm
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
import time
import wandb
import matplotlib.pyplot as plt

# Hiding the warnings
warnings.filterwarnings('ignore')
# Hiding the warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import warnings
from sklearn import metrics
import wandb
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
import wandb
from sklearn.metrics import r2_score
import seaborn as sns



# Define the Transformer model for regression
class SalesPredictionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(SalesPredictionTransformer, self).__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers
        )
        # self.fc = nn.Linear(hidden_dim, output_dim * 32)  # Adjust output dimension to output a sequence
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.fc(x)

        # Reshape the output to match the batch size
        batch_size = x.size(0)
        x = x.view(batch_size, -1, output_dim)

        return x


import random
import h5py
import pandas as pd

import wandb

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

sheet_name = '0'

value = region_map[float(sheet_name)]
config = {"region": value}


# Read the dataset using the current sheet name
dataset = file[sheet_name][:]
dataset = pd.DataFrame(dataset)

# Read the column names from the attributes
column_names = file[sheet_name].attrs['columns']

# Assign column names to the dataset
dataset.columns = column_names

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(dataset.head(30))

# Preprocess the data
# Preprocess the data
train_data = dataset # Drop the last 30 rows
xs = train_data.drop(['Sales'], axis=1)
ys = train_data['Sales']  # Use the updated train_data for ys
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.3, random_state=42)


# Scale the input features
scaler = MinMaxScaler()
xs_train = scaler.fit_transform(xs_train)
xs_test = scaler.transform(xs_test)

# Convert the data to tensors
train_inputs = torch.tensor(xs_train, dtype=torch.float32)
train_targets = torch.tensor(ys_train.values, dtype=torch.float32)
test_inputs = torch.tensor(xs_test, dtype=torch.float32)
test_targets = torch.tensor(ys_test.values, dtype=torch.float32)

# Adjust the batch size
batch_size = 32

# Create DataLoader for batch processing
train_dataset = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_inputs, test_targets)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(train_inputs.shape)
print(train_targets.shape)
print(test_inputs.shape)
print(test_targets.shape)


# Define the hyperparameters
# Define the hyperparameters
input_dim = train_inputs.size(1)
hidden_dim = 32  # Adjust the hidden dimension as desired
output_dim = 1
num_layers = 3
num_heads = 4  # Adjust the number of heads as desired
dropout = 0.5
learning_rate = 0.001
num_epochs = 50

# Initialize the transformer model, loss function, and optimizer
# model = SalesPredictionTransformer(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchviz import make_dot
from graphviz import Digraph


import torch
import torch.nn as nn
from torchviz import make_dot


# Assuming you have already initialized the optimizer and criterion


# Dummy input with the shape of a single batch from the train_loader
# batch_idx, (batch_inputs, batch_targets) = next(enumerate(train_loader))
# dummy_input = batch_inputs.to(device)
#
# # Pass the dummy input through the model to get the output
# model_output = model(dummy_input)
#
# # Create a graph of the model
# graph = make_dot(model_output, params=dict(model.named_parameters()))
# # Print the detailed structure of the model
# summary(model, input_size=(dummy_input.size(1), dummy_input.size(2)))  # Adjust the input size to match the 3D input
#
# # Customize the appearance of the graph
# graph.format = 'png'
# graph.attr(bgcolor='white')  # Set the background color to white
#
# # Save the graph to a file
# graph.render("transformer_model_graph", cleanup=True)




# Create the model instance
model = SalesPredictionTransformer(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Assuming you have already initialized the optimizer and criterion




# Dummy input with the shape of a single batch from the train_loader
batch_idx, (batch_inputs, batch_targets) = next(enumerate(train_loader))
dummy_input = batch_inputs.to(device)  # Move the dummy input to the same device as the model

# Define a custom forward function that returns output at each layer
def custom_forward(x):
    x = model.encoder(x)
    layers_output = []
    for layer in model.transformer.layers:
        x = layer(x)
        layers_output.append(x)
    return layers_output

# Pass the dummy input through the custom forward function to get the layers' output
model_outputs = custom_forward(dummy_input)

# Create a graph of the model
graph = make_dot(model_outputs, params=dict(model.named_parameters()))

# Customize the appearance of the graph
graph.format = 'png'
graph.attr(bgcolor='white')  # Set the background color to white

# Save the graph to a file
graph.render("transformer_model_graph", cleanup=True)
