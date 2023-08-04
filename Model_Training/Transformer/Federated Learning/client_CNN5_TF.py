# -*- coding = utf-8 -*-
# @time:05/07/2023 18:11
# Author:Yunbo Long
# @File:client_CNN6.py
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

sheet_name = '5'

value = region_map[float(sheet_name)]
config = {"region": value}
wandb.init(project='Cambridge_sales_TF_FL', config=config)

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
model = SalesPredictionTransformer(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)




# Train the transformer model
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
            param.data.copy_(torch.from_numpy(new_param).float())  # Update local model parameters

        model.train()

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 5

        start_time = time.time()  # Start time for the current epoch
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

        elapsed_time = time.time() - start_time  # Elapsed time for the current epoch

        wandb.log({"Communication Time": elapsed_time})
        # Code for sending model updates to the server

        return [param.detach().numpy().astype('float32') for param in model.parameters()], len(train_dataset), {}

    def evaluate(self, parameters, config=None):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        # Update local model parameters
        for param, new_param in zip(model.parameters(), parameters):
            param.data.copy_(torch.from_numpy(new_param).float())

        testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # Evaluation function
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

        # Evaluate the model on the test set
        predictions, targets, loss = evaluate_model1(model, testloader)

        pred_s_test = predictions.squeeze()



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
            'loss(test data)': loss
        })

        region_value = region_map[float(sheet_name)]

        # Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(targets, pred_s_test, label=region_value)
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title('Scatter Plot: Actual vs Predicted Sales')

        # Add line of best fit
        coefficients = np.polyfit(targets, pred_s_test, 1)
        poly_line = np.polyval(coefficients, targets)
        plt.plot(targets, poly_line, color='red', label='Line of Best Fit')

        plt.legend()
        plt.tight_layout()
        plt.savefig('scatter_plot5.png')
        wandb.log({'Scatter Plot': wandb.Image('scatter_plot5.png')})
        plt.close()

        # Distribution Plot
        plt.figure(figsize=(8, 6))
        residuals = targets - pred_s_test
        sns.histplot(residuals, kde=True, label=region_value)
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Distribution of Residuals')
        plt.legend()
        plt.tight_layout()
        plt.savefig('distribution_plot5.png')
        wandb.log({'Distribution Plot': wandb.Image('distribution_plot5.png')})
        plt.close()

        performance_metrics = {
            "rmse": float(rmse_val),
            "r_squared": float(r2_val),
            "mae": float(mae_val),
            "mse": float(mse_val),
            "loss": float(loss)
        }
        return loss, len(test_dataset), performance_metrics


# Create the client and start the client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8090",
    client=RegressionClient()
)
