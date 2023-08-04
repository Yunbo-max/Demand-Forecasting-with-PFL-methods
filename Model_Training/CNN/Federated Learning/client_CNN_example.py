# -*- coding = utf-8 -*-
# @time:04/07/2023 18:46
# Author:Yunbo Long
# @File:client_CNN_example.py
# @Software:PyCharm
# Import the necessary libraries
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
import xgboost as xgb
import lightgbm as lgb
import warnings
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
from sklearn.metrics import r2_score
import time
import wandb

#Hiding the warnings
warnings.filterwarnings('ignore')
#Hiding the warnings
warnings.filterwarnings('ignore')

# # Initialize Weights and Biases
# wandb.init(project="CNN", name=f"Sheet_{random_sheet_name}")

# Define the CNN architecture
class SalesPredictionCNN(nn.Module):
    def __init__(self):
        super(SalesPredictionCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import random

# List of sheet names
sheet_names = ['0', '1', '2', '3', '4']

# Select a random sheet name
random_sheet_name = sheet_names[1]
# Load and preprocess your dataset
# Modify this part according to your dataset
# Read the Excel dataset
# Initialize Weights and Biases
wandb.init(project="CNN8", name=f"Sheet_{random_sheet_name}")
dataset = pd.read_excel('E:\Python\Dataguan\FL_supply_chain\\result_document\local_train_data_sheets.xlsx', sheet_name=random_sheet_name)

# Preprocess the data
train_data = dataset
xs = train_data.loc[:, train_data.columns != 'Sales']
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


# Create the model
model = SalesPredictionCNN()
criterion = nn.MSELoss()


# Define performance metrics functions
def rmse(targets, predictions):
    return float(np.sqrt(mean_squared_error(targets, predictions)))

def r_squared(targets, predictions):
    return float(r2_score(targets, predictions))

def mae(targets, predictions):
    return float(mean_absolute_error(targets, predictions))

def mse(targets, predictions):
    return float(mean_squared_error(targets, predictions))

# Define the client class
class RegressionClient(fl.client.NumPyClient):
    def __init__(self):
        self.total_communication_time = 0.0

    def get_parameters(self, config=None):
        return [param.detach().numpy().astype('float32') for param in model.parameters()]

    def fit(self, parameters, config):
        for param, new_param in zip(model.parameters(), parameters):
            param.data = torch.from_numpy(new_param).float()  # Convert model parameters to float

        model.train()

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        num_epochs = 5
        batch_size = 32
        losses = []

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):

            # Measure the start time before sending model updates
            iteration_start_time = time.time()

            epoch_loss = 0.0
            batch_losses = []

            for inputs, targets in train_loader:
                inputs = inputs.float()  # Convert inputs to float
                targets = targets.float()  # Convert targets to float

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            epoch_loss = np.mean(batch_losses)
            losses.append(epoch_loss)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            # Code for sending model updates to the server

            # Measure the end time after receiving the aggregated model
            iteration_end_time = time.time()

            # Calculate the communication time for the current iteration
            iteration_communication_time = iteration_end_time - iteration_start_time

            # Add the current iteration's communication time to the total communication time
            self.total_communication_time += iteration_communication_time

        # Log the total communication time for all iterations
        wandb.log({"total_communication_time": self.total_communication_time})

        return [param.detach().numpy().astype('float32') for param in model.parameters()], len(train_dataset), {}

    def evaluate(self, parameters, config=None):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        for param, new_param in zip(model.parameters(), parameters):
            param.data = torch.from_numpy(new_param).float()

        testloader = DataLoader(test_dataset, batch_size=32)

        def evaluate_model(net, testloader):
            criterion = nn.MSELoss()
            loss = 0.0
            predictions = []
            net.eval()
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs = inputs.float()
                    targets = targets.float()

                    outputs = net(inputs)
                    batch_loss = criterion(outputs, targets)
                    loss += batch_loss.item()

                    predictions.append(outputs.numpy())

                predictions = np.concatenate(predictions)
                targets = test_targets.numpy()

                loss /= len(testloader)  # Calculate average loss

                return predictions, targets, loss

        # ...

        # Evaluate the model after all epochs
        predictions, targets, loss = evaluate_model(model, testloader)

        performance_metrics = {
            "rmse": rmse(targets, predictions),
            "r_squared": r_squared(targets, predictions),
            "mae": mae(targets, predictions),
            "mse": mse(targets, predictions),
            "loss": loss
        }

        print("Performance Metrics:")
        for metric, value in performance_metrics.items():
            print(f"{metric}: {value}")

        # Log metrics to Weights and Biases
        wandb.log(performance_metrics)

        return loss, len(test_dataset), performance_metrics


losses = []
train_rmse_list = []
test_rmse_list = []
mae_test_list = []
rmse_test_list = []
mape_test_list = []
mse_test_list = []
r2_test_list = []

# Create the client and start the client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8098",
    client=RegressionClient()
)
