# -*- coding = utf-8 -*-
# @time:05/07/2023 13:07
# Author:Yunbo Long
# @File:CNN_centralised_5_quantity.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @time:05/05/2023 11:20
# Author:Yunbo Long
# @File:CNN.py
# @Software:PyCharm
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



# Hiding the warnings
warnings.filterwarnings('ignore')


import wandb

# Initialize Weights and Biases with project name
wandb.init(project="CNN_centralised")

import h5py

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

import h5py

# Print the value for a given key
key = 3
value = region_map[key]
print(value)

import wandb

# Initialize Weights and Biases with project name
wandb.init(project="MLP_centralised_quantity_with_validation")

import h5py

# Open the HDF5 file
dataset = pd.read_csv('E:\Federated_learning_flower\experiments\Presentation\integrated_train_data_ISMM.csv')
dataset = dataset.drop(columns=['index'])

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(dataset.head(30))

# Preprocess the data
train_data = dataset
xs = train_data.drop(['Order Item Quantity'], axis=1)
ys = train_data['Order Item Quantity']
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

num_epochs = 20
evaluation_interval = 1

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

    model.eval()
    train_outputs = model(train_inputs)
    train_rmse = torch.sqrt(torch.mean((train_outputs.squeeze() - train_targets) ** 2)).item()

    pred_s_train = train_outputs.squeeze().detach().numpy()
    mae_train = mean_absolute_error(train_targets.detach().numpy(), pred_s_train)
    rmse_train = np.sqrt(mean_squared_error(train_targets.detach().numpy(), pred_s_train))
    mape_train = np.mean(
        np.abs((train_targets.detach().numpy() - pred_s_train) / train_targets.detach().numpy())) * 100
    mse_train = mean_squared_error(train_targets.detach().numpy(), pred_s_train)
    r2_train = metrics.r2_score(train_targets.detach().numpy(), pred_s_train)

    train_rmse_list.append(train_rmse)
    mae_train_list.append(mae_train)
    rmse_train_list.append(rmse_train)
    mape_train_list.append(mape_train)
    mse_train_list.append(mse_train)
    r2_train_list.append(r2_train)

    val_losses = []
    model.eval()

    for inputs, targets in val_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        val_losses.append(loss.item())

    epoch_val_loss = np.mean(val_losses)

    val_rmse, mae_val, rmse_val, mape_val, mse_val, r2_val = None, None, None, None, None, None

    if epoch % evaluation_interval == 0:
        val_outputs = model(val_inputs)
        val_rmse = torch.sqrt(torch.mean((val_outputs.squeeze() - val_targets) ** 2)).item()

        pred_s_val = val_outputs.squeeze().detach().numpy()
        mae_val = mean_absolute_error(val_targets.detach().numpy(), pred_s_val)
        rmse_val = np.sqrt(mean_squared_error(val_targets.detach().numpy(), pred_s_val))
        mape_val = np.mean(
            np.abs((val_targets.detach().numpy() - pred_s_val) / val_targets.detach().numpy())) * 100
        mse_val = mean_squared_error(val_targets.detach().numpy(), pred_s_val)
        r2_val = metrics.r2_score(val_targets.detach().numpy(), pred_s_val)

        val_rmse_list.append(val_rmse)
        mae_val_list.append(mae_val)
        rmse_val_list.append(rmse_val)
        mape_val_list.append(mape_val)
        mse_val_list.append(mse_val)
        r2_val_list.append(r2_val)

    test_outputs = model(test_inputs)
    test_loss = criterion(test_outputs, test_targets.unsqueeze(1))
    test_rmse = torch.sqrt(torch.mean((test_outputs.squeeze() - test_targets) ** 2)).item()

    pred_s_test = test_outputs.squeeze().detach().numpy()
    mae_test = mean_absolute_error(test_targets.detach().numpy(), pred_s_test)
    rmse_test = np.sqrt(mean_squared_error(test_targets.detach().numpy(), pred_s_test))
    mape_test = np.mean(
        np.abs((test_targets.detach().numpy() - pred_s_test) / test_targets.detach().numpy())) * 100
    mse_test = mean_squared_error(test_targets.detach().numpy(), pred_s_test)
    r2_test = metrics.r2_score(test_targets.detach().numpy(), pred_s_test)

    mae_test_list.append(mae_test)
    rmse_test_list.append(rmse_test)
    mape_test_list.append(mape_test)
    mse_test_list.append(mse_test)
    r2_test_list.append(r2_test)

    wandb.log({
        'Train Loss': epoch_loss,
        'MAE (train data)': mae_train,
        'RMSE (train data)': rmse_train,
        'MAPE (train data)': mape_train,
        'MSE (train data)': mse_train,
        'R2 (train data)': r2_train,

        'Validation Loss': epoch_val_loss,
        'MAE (validation data)': mae_val,
        'RMSE (validation data)': rmse_val,
        'MAPE (validation data)': mape_val,
        'MSE (validation data)': mse_val,
        'R2 (validation data)': r2_val,

        'Test Loss': test_loss.item(),
        'MAE (test data)': mae_test,
        'RMSE (test data)': rmse_test,
        'MAPE (test data)': mape_test,
        'MSE (test data)': mse_test,
        'R2 (test data)': r2_test
    })



    # Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(test_targets.detach().numpy(), pred_s_test)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Scatter Plot: Actual vs Predicted Sales')

    # Add line of best fit
    coefficients = np.polyfit(test_targets.detach().numpy(), pred_s_test, 1)
    poly_line = np.polyval(coefficients, test_targets.detach().numpy())
    plt.plot(test_targets.detach().numpy(), poly_line, color='red', label='Line of Best Fit')

    plt.legend()
    plt.tight_layout()
    plt.savefig('scatter_plot.png')
    wandb.log({'Scatter Plot': wandb.Image('scatter_plot.png')})
    plt.close()

    # Distribution Plot
    plt.figure(figsize=(8, 6))
    residuals = test_targets.detach().numpy() - pred_s_test
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Distribution of Residuals')
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution_plot.png')
    wandb.log({'Distribution Plot': wandb.Image('distribution_plot.png')})
    plt.close()

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
    print('Train RMSE:', train_rmse)
    print('Validation RMSE:', val_rmse)
    print('Test RMSE:', test_rmse)
    print('MAE Value (train data):', mae_train)
    print('RMSE (train data):', rmse_train)
    print('MAPE (train data):', mape_train)
    print('MSE (train data):', mse_train)
    print('R2 (train data):', r2_train)
    print('MAE Value (validation data):', mae_val)
    print('RMSE (validation data):', rmse_val)
    print('MAPE (validation data):', mape_val)
    print('MSE (validation data):', mse_val)
    print('R2 (validation data):', r2_val)
    print('MAE Value (test data):', mae_test)
    print('RMSE (test data):', rmse_test)
    print('MAPE (test data):', mape_test)
    print('MSE (test data):', mse_test)
    print('R2 (test data):', r2_test)
    print('\n')

print('Train Loss:', epoch_loss)
print('Train RMSE:', train_rmse)
print('MAE Value (train data):', mae_train)
print('RMSE (train data):', rmse_train)
print('MAPE (train data):', mape_train)
print('MSE (train data):', mse_train)
print('R2 (train data):', r2_train)
print('Test Loss:', test_loss.item())
print('MAE Value (test data):', mae_test)
print('RMSE (test data):', rmse_test)
print('MAPE (test data):', mape_test)
print('MSE (test data):', mse_test)
print('R2 (test data):', r2_test)
print('\n')




