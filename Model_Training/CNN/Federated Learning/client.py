import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import flwr as fl

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(8, 64)  # Input size is 8 for the California housing dataset
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = RegressionModel()
model = model.float()  # Set the model's parameters to torch.float32

optimizer = optim.Adam(model.parameters())

# Fetch the California housing dataset
data = fetch_california_housing()

# Randomly select a subset of the dataset
subset_size = 1000  # Set the desired subset size
random_indices = np.random.choice(len(data.data), size=subset_size, replace=False)
subset_data = data.data[random_indices]
subset_target = data.target[random_indices]

# Split the subset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(subset_data, subset_target, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
train_data = torch.from_numpy(X_train).float()
train_labels = torch.from_numpy(y_train).float().unsqueeze(1)
test_data = torch.from_numpy(X_test).float()
test_labels = torch.from_numpy(y_test).float().unsqueeze(1)

class HousingClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [param.detach().numpy().astype('float32') for param in model.parameters()]

    def fit(self, parameters, config):
        for param, new_param in zip(model.parameters(), parameters):
            param.data = torch.from_numpy(new_param)

        criterion = nn.MSELoss()
        model.train()

        for epoch in range(20):
            train_loss = 0.0

            optimizer.zero_grad()
            output = model(train_data)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            print(f"Epoch {epoch + 1}: Training Loss = {train_loss}")

        return [param.detach().numpy().astype('float32') for param in model.parameters()], len(train_data), {}

    def evaluate(self, parameters, config):
        for param, new_param in zip(model.parameters(), parameters):
            param.data = torch.from_numpy(new_param)

        criterion = nn.MSELoss()
        model.eval()

        total_loss = 0.0
        with torch.no_grad():
            output = model(test_data)
            loss = criterion(output, test_labels)
            total_loss = loss.item()

        return total_loss, len(test_data), {"mse_loss": total_loss}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8085",
    client=HousingClient()
)