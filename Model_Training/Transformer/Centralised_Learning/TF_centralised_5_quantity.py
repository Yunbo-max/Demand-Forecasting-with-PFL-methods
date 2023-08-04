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
import wandb
from sklearn.metrics import r2_score

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

# Hiding the warnings
warnings.filterwarnings('ignore')


import wandb

# Initialize Weights and Biases with project name


import h5py
sheet_name = ['0','1', '2', '3', '5', '6', '7', '9', '10', '12', '14', '16', '17', '22']

# Initialize Weights and Biases with project name
wandb.init(project="Cambridge_TF_centralised_quantity_final")

import h5py

# Open the HDF5 file
dataset = pd.read_csv('E:\Federated_learning_flower\experiments\Presentation\integrated_train_data_ISMM.csv')
specified_regions = [region_map[int(name)] for name in sheet_name]
# Filter rows based on specified region names
dataset = dataset[dataset['index'].isin(specified_regions)]

dataset = dataset.drop(columns=['index'])

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

# xq=train_data.loc[:, train_data.columns != 'Order Item Quantity']
# yq=train_data['Order Item Quantity']
# xq_train, xq_test,yq_train,yq_test = train_test_split(xq,yq,test_size = 0.3, random_state = 42)

scaler=MinMaxScaler()
xs_train=scaler.fit_transform(xs_train)
xs_test=scaler.transform(xs_test)
# xq_train=scaler.fit_transform(xq_train)
# xq_test=scaler.transform(xq_test)


train_inputs = torch.tensor(xs_train, dtype=torch.float32)
train_targets = torch.tensor(ys_train.values, dtype=torch.float32)
test_inputs = torch.tensor(xs_test, dtype=torch.float32)
test_targets = torch.tensor(ys_test.values, dtype=torch.float32)

print(train_inputs.shape)
print(train_targets.shape)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import warnings
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn import svm,metrics,tree,preprocessing,linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

warnings.filterwarnings('ignore')



input_dim = train_inputs.size(1)
hidden_dim = 32  # Adjust the hidden dimension as desired
output_dim = 1
num_layers = 3
num_heads = 4  # Adjust the number of heads as desired
dropout = 0.5
learning_rate = 0.001


# Initialize the transformer model, loss function, and optimizer
model = SalesPredictionTransformer(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Save the initial parameters
torch.save(model.state_dict(), 'initial_parameters.pth')

num_epochs = 50
batch_size = 32

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



def evaluate_model1(net, testloader):
    criterion = torch.nn.MSELoss()  # Use Mean Squared Error (MSE) for regression
    net.eval()
    predictions = []
    targets = []
    loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, batch_targets in testloader:
            batch_size = inputs.size(0)
            num_samples += batch_size

            if batch_size < testloader.batch_size:
                continue  # Skip the last incomplete batch

            outputs = net(inputs)

            batch_loss = criterion(outputs.squeeze(), batch_targets.float())
            loss += batch_loss.item() * batch_size

            if isinstance(outputs, torch.Tensor):  # Check if outputs is a tensor
                outputs = outputs.squeeze().cpu().tolist()  # Convert to list if tensor
            else:
                outputs = [outputs]  # Wrap single float in a list

            # print("Outputs:", outputs)
            # print("Batch Targets:", batch_targets.cpu().tolist())

            predictions.extend(outputs)  # Extend predictions with the batch predictions
            targets.extend(batch_targets.cpu().tolist())  # Extend targets with the batch targets

    predictions = np.array(predictions)
    targets = np.array(targets)
    loss /= num_samples

    return predictions, targets, loss


# Train the transformer model
# Train the transformer model
losses = []
num_batches = len(train_loader)
for epoch in range(num_epochs):
    batch_losses = []
    for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
        # Skip the last incomplete batch
        if batch_idx == num_batches - 1 and len(batch_inputs) < batch_size:
            continue

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs.squeeze(), batch_targets)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    epoch_loss = np.mean(batch_losses)
    losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    wandb.log({'Training Loss': epoch_loss})  # Log the training loss to Wandb

# Evaluate the model
# Evaluation function


    # Evaluate the model on the test set
    predictions, targets, loss = evaluate_model1(model, testloader)


    # Calculate evaluation metrics
    rmse_val = np.sqrt(mean_squared_error(targets, predictions))
    mae_val = mean_absolute_error(targets, predictions)
    mape_val = np.mean(np.abs((targets - predictions) / targets)) * 100
    mse_val = mean_squared_error(targets, predictions)
    r2_val = r2_score(targets, predictions)

    # Print metrics
    print("Performance Metrics:")
    print(f"RMSE (test data): {rmse_val}")
    print(f"MAE (test data): {mae_val}")
    print(f"MAPE (test data): {mape_val}")
    print(f"MSE (test data): {mse_val}")
    print(f"R2 (test data): {r2_val}")
    print(f"loss (test data): {loss}")

    wandb.log({
        'RMSE (test data)': rmse_val,
        'MAE (test data)': mae_val,
        'MAPE (test data)': mape_val,
        'MSE (test data)': mse_val,
        'R2 (test data)': r2_val,
        'Test Loss': loss
    })

    performance_metrics = {
        "rmse": float(rmse_val),
        "r_squared": float(r2_val),
        "mae": float(mae_val),
        "mse": float(mse_val),
        "loss": float(loss)
    }




