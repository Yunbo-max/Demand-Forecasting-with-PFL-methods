# -*- coding = utf-8 -*-
# @time:03/07/2023 09:32
# Author:Yunbo Long
# @File:server.py
# @Software:PyCharm
import flwr as fl
import argparse
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


from flwr.server.client_proxy import ClientProxy
import wandb

import flwr as fl
import torch

from collections import OrderedDict

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common import (
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,)

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

wandb.init(project="Server_Cambridge_sales_CNN_FL")
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy

import wandb

import numpy as np
import wandb
from flwr.common import Parameters, Scalar
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union
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




model = SalesPredictionCNN()

class SaveModelStrategy(FedAvg):
    def __init__(self, model, last_round: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_round = last_round
        self.model = model  # Add this line

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Parameters]],
        failures: List[Union[Tuple[ClientProxy, Parameters], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from the base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # Convert the List of ndarrays to a dictionary with appropriate keys
            aggregated_ndarrays_dict = {}
            for i, (name, param) in enumerate(self.model.state_dict().items()):  # use self.model here
                aggregated_ndarrays_dict[name] = aggregated_ndarrays[i]

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", **aggregated_ndarrays_dict)

        # # Save the model in the last round
        # if server_round == self.last_round:
        #     wandb.run.summary["model"] = wandb.Artifact("cnn_model", type="model")
        #     with wandb.run.summary["model"].new_file(f"round-{server_round}-weights.npz") as f:
        #         np.savez(f, *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def weighted_average(metrics):
    losses = []
    rmse_values = []
    r_squared_values = []
    mse_values = []
    mae_values = []
    examples = []

    for num_examples, m in metrics:
        if "loss" in m:
            losses.append(num_examples * m["loss"])
        if "rmse" in m:
            rmse_values.append(num_examples * m["rmse"])
        if "r_squared" in m:
            r_squared_values.append(num_examples * m["r_squared"])
        if "mse" in m:
            mse_values.append(num_examples * m["mse"])
        if "mae" in m:
            mae_values.append(num_examples * m["mae"])
        examples.append(num_examples)

    loss = sum(losses) / sum(examples) if losses else None
    rmse = np.sqrt(sum(rmse_values) / sum(examples)) if rmse_values else None
    r_squared = sum(r_squared_values) / sum(examples) if r_squared_values else None
    mse = sum(mse_values) / sum(examples) if mse_values else None
    mae = sum(mae_values) / sum(examples) if mae_values else None

    metrics_dict = {
        "loss": loss,
        "rmse": rmse,
        "r_squared": r_squared,
        "mse": mse,
        "mae": mae,
    }

    wandb.log(metrics_dict)

    return metrics_dict




#
# fl.server.start_server(
#     server_address="0.0.0.0:8085",
#     config=fl.server.ServerConfig(num_rounds=5)
# )

num_rounds = 100

# Create strategy
strategy = SaveModelStrategy(
    # fraction_fit=0.5,
    # fraction_evaluate=0.5,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    last_round=num_rounds,
    model = model,
    # evaluate_fn=get_evaluate_fn(model, args.toy),
    # on_fit_config_fn=fit_config,
    # on_evaluate_config_fn=evaluate_config,
    # initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start Flower server for four rounds of federated learning
fl.server.start_server(
    server_address="0.0.0.0:8089",
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
)
