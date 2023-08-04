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

wandb.init(project="server_ISMM_DP_quantity_FL")

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



# Create strategy
strategy = fl.server.strategy.FedAvg(
    # fraction_fit=0.5,
    # fraction_evaluate=0.5,
    min_fit_clients=14,
    min_evaluate_clients=14,
    min_available_clients=14,
    # evaluate_fn=get_evaluate_fn(model, args.toy),
    # on_fit_config_fn=fit_config,
    # on_evaluate_config_fn=evaluate_config,
    # initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start Flower server for four rounds of federated learning
fl.server.start_server(
    server_address="0.0.0.0:8082",
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
)


