# -*- coding = utf-8 -*-
# @time:03/07/2023 23:49
# Author:Yunbo Long
# @File:client_LSTM.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @time:03/07/2023 23:29
# Author:Yunbo Long
# @File:client_CNN.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @time:03/07/2023 23:08
# Author:Yunbo Long
# @File:client_MLP.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @time:03/07/2023 10:51
# Author:Yunbo Long
# @File:client_final.py
# @Software:PyCharm
# Import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import flwr as fl

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


import wandb

#Hiding the warnings
warnings.filterwarnings('ignore')
#Hiding the warnings
warnings.filterwarnings('ignore')




import random

# List of sheet names
sheet_names = ['0', '1', '2', '3', '4']

# Select a random sheet name
random_sheet_name = random.choice(sheet_names)

# Initialize Weights and Biases
wandb.init(project="LSTM", name=f"Sheet_{random_sheet_name}")

# Read the Excel dataset using the randomly selected sheet name
dataset = pd.read_excel('E:\Python\Dataguan\FL_supply_chain\\result_document\local_train_data_sheets.xlsx', sheet_name=random_sheet_name)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(dataset.head(30))

train_data = dataset
xs=train_data.loc[:, train_data.columns != 'Sales']
ys=dataset['Sales']
xs_train, xs_test,ys_train,ys_test = train_test_split(xs,ys,test_size = 0.3, random_state = 42)

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


# Define your LSTM model
class SalesPredictionLSTM(nn.Module):
    def __init__(self, input_size):
        super(SalesPredictionLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 400, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(400 * 2, 300, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(300 * 2, 200, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(200 * 2, 100, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(100 * 2, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)

        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# Define your training and testing datasets
train_dataset = TensorDataset(train_inputs, train_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

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
    def get_parameters(self, config=None):
        return [param.detach().numpy().astype('float32') for param in model.parameters()]

    def fit(self, parameters, config):
        for param, new_param in zip(model.parameters(), parameters):
            param.data = torch.from_numpy(new_param).float()  # Convert model parameters to float

        model.train()

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        for epoch in range(3):
            train_loss = 0.0

            for inputs, targets in train_loader:
                inputs = inputs.float()  # Convert inputs to float
                targets = targets.float()  # Convert targets to float

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print(f"Epoch {epoch + 1}: Training Loss = {train_loss / len(train_loader)}")

        return [param.detach().numpy().astype('float32') for param in model.parameters()], len(train_dataset), {}

    def evaluate(self, parameters, config=None):
        for param, new_param in zip(model.parameters(), parameters):
            param.data = torch.from_numpy(new_param).float()  # Convert model parameters to float

        model.eval()

        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        test_loss = 0.0
        predictions = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.float()  # Convert inputs to float
                targets = targets.float()  # Convert targets to float

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                predictions.append(outputs.numpy())

        predictions = np.concatenate(predictions)
        targets = test_targets.numpy()

        performance_metrics = {
            "rmse": rmse(targets, predictions),
            "r_squared": r_squared(targets, predictions),
            "mae": mae(targets, predictions),
            "mse": mse(targets, predictions),
            "loss": test_loss
        }

        print("Performance Metrics:")
        for metric, value in performance_metrics.items():
            print(f"{metric}: {value}")

            # Log metrics to Weights and Biases
            wandb.log({metric: value})

        return test_loss, len(test_dataset), performance_metrics

# Initialize your LSTM model
model = SalesPredictionLSTM(input_size=train_inputs.shape[1])

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the client and start the client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8083",
    client=RegressionClient()
)
