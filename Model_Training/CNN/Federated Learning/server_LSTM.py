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


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from the base class (FedAvg) to aggregate loss
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh metrics of each client by the number of examples used
        metrics = [(r.num_examples, r.metrics) for _, r in results]
        aggregated_metrics = weighted_average(metrics)
        print(f"Round {server_round} metrics aggregated from client results: {aggregated_metrics}")

        # Log metrics to wandb
        wandb.log(aggregated_metrics, step=server_round)

        # Return aggregated loss and metrics
        return aggregated_loss, aggregated_metrics


def weighted_average(metrics):
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    rmse_values = [num_examples * m["rmse"] for num_examples, m in metrics]
    r_squared_values = [num_examples * m["r_squared"] for num_examples, m in metrics]
    mse_values = [num_examples * m["mse"] for num_examples, m in metrics]
    mae_values = [num_examples * m["mae"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    loss = sum(losses) / sum(examples)
    rmse = np.sqrt(sum(rmse_values) / sum(examples))
    r_squared = sum(r_squared_values) / sum(examples)
    mse = sum(mse_values) / sum(examples)
    mae = sum(mae_values) / sum(examples)

    print("Loss:", loss)
    print("RMSE:", rmse)
    print("R-squared:", r_squared)
    print("MSE:", mse)
    print("MAE:", mae)

    return {
        "loss": loss,
        "rmse": rmse,
        "r_squared": r_squared,
        "mse": mse,
        "mae": mae,
    }


#
# fl.server.start_server(
#     server_address="0.0.0.0:8085",
#     config=fl.server.ServerConfig(num_rounds=5)
# )

wandb.init(project="server_LSTM")

# Create strategy
strategy = SaveModelStrategy(
    fraction_fit=0.2,
    fraction_evaluate=0.2,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
    # evaluate_fn=get_evaluate_fn(model, args.toy),
    # on_fit_config_fn=fit_config,
    # on_evaluate_config_fn=evaluate_config,
    # initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start Flower server for four rounds of federated learning
fl.server.start_server(
    server_address="0.0.0.0:8083",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
