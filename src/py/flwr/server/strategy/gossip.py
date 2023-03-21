# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""


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
import random
import numpy as np

from .aggregate import aggregate, weighted_loss_avg
from .strategy import Strategy
from .fedavg import FedAvg

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

# aggregate_gossip goes from List[(ClientProxy, FitRes)] -> Dict[cid, params] (aggregated)
def aggregate_gossip(
    results: List[Tuple[ClientProxy, FitRes]],
    gossip_count: float) -> Dict[str, Parameters]:
    """Create a new set of parameters per client.
    Each client gets averaged with a fraction of the other clients, randomly
    """
    aggregated_params: Dict[str, Parameters] = {}

    inputs = [
        (client.cid, parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for client, fit_res in results
    ]

    communication_cost = 0
    for cid, params, n in inputs:
        # Randomly sample other clients to gossip with
        gossip_clients = [client for client in inputs if client[0] != cid]
        gossip_clients = random.sample(gossip_clients, gossip_count)


        # gossip_clients = gossip_clients[np.random.choice(np.arange(len(gossip_clients)), int(fraction_gossip * len(gossip_clients)), replace=False)] #

        # Aggregate with gossip clients
        # communication_cost += len(str([(client[1], client[2]) for client in gossip_clients])) 
        gossip_params = [(params, n)] + [(client[1], client[2]) for client in gossip_clients]
        aggregated_params[cid] = ndarrays_to_parameters(
            aggregate(gossip_params)
        )
        communication_cost += len(aggregated_params[cid]) * gossip_count # Get size of message (approximately)

    return aggregated_params, communication_cost

def aggregate_gossip_segmented(
    results: List[Tuple[ClientProxy, FitRes]],
    gossip_count: float,
    segments: int) -> Dict[str, Parameters]:
    """Create a new set of parameters per client.
    Each client gets averaged with a fraction of the other clients, randomly
    The gossip protocol is run separately for each segment of the model weights
    """

    # Split each client's parameters into segments
    # Because the parameters are made up of several ndarrays of different shapes, we need to keep track of the shapes

    # 1. Turn each client's parameters into a single flat ndarray, and store a list of the shapes of each ndarray
    inputs = []
    for client, fit_res in results:
        params = parameters_to_ndarrays(fit_res.parameters)
        shapes = [param.shape for param in params]
        flat_params = np.concatenate([param.flatten() for param in params])
        param_size = flat_params.shape[0]
        inputs.append((client.cid, flat_params, fit_res.num_examples))

    segment_size = param_size // segments

    for cid, params, n in inputs:
        for i in range(segments):
            segment_start = i * segment_size
            segment_end = segment_start + segment_size
            if i == segments - 1:
                segment_end = param_size

            # Randomly sample other clients to gossip with for THIS segment
            gossip_clients = [client for client in inputs if client[0] != cid]
            gossip_clients = random.sample(gossip_clients, gossip_count)

            params[segment_start:segment_end] = aggregate([(params[segment_start:segment_end], n)] + [(client[1][segment_start:segment_end], client[2]) for client in gossip_clients])

    # Transform each clients parameters back into a list of ndarrays following shapes
    aggregated_params: Dict[str, Parameters] = {}
    shape_offsets = [0]
    offset = 0
    for shape in shapes:
        offset += np.prod(shape)
        shape_offsets.append(offset)

    communication_cost = 0
    for cid, params, _ in inputs:
        params = ndarrays_to_parameters(
            [params[shape_offsets[i]:shape_offsets[i+1]].reshape(shapes[i]) for i in range(len(shapes))]
        )
        aggregated_params[cid] = params
        communication_cost += len(aggregated_params[cid]) * gossip_count

    return aggregated_params, communication_cost

def aggregate_gossip_pga(
    client_dict: Dict[str, Parameters],
    client_data_counts: Dict[str, int]):
    """Create an 
    """
    params = [(parameters_to_ndarrays(client_dict[cid]), client_data_counts[cid]) for cid in client_dict if client_data_counts.get(cid)]
    total_communication = sum([len(client_dict[cid]) for cid in client_dict if client_data_counts.get(cid)]) * len(client_dict)

    return ndarrays_to_parameters(aggregate(params)), total_communication


    # # TODO: Sort out the edge cases when not all clients participate
    # aggregated_params: Dict[str, Parameters] = {}

    # inputs = [
    #     (client.cid, parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
    #     for client, fit_res in results
    # ]

    # communication_cost = 0
    # for cid, params, n in inputs:
    #     # Randomly sample other clients to gossip with
    #     gossip_clients = [client for client in inputs if client[0] != cid]
    #     gossip_clients = random.sample(gossip_clients, gossip_count)


    #     # gossip_clients = gossip_clients[np.random.choice(np.arange(len(gossip_clients)), int(fraction_gossip * len(gossip_clients)), replace=False)] #

    #     # Aggregate with gossip clients
    #     communication_cost += len(str([(client[1], client[2]) for client in gossip_clients])) # Get size of message (approximately)
    #     gossip_params = [(params, n)] + [(client[1], client[2]) for client in gossip_clients]
    #     aggregated_params[cid] = ndarrays_to_parameters(
    #         aggregate(gossip_params)
    #     )

    # return aggregated_params, communication_cost


# flake8: noqa: E501
class GossipAvg(FedAvg):
    """Configurable GossipAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        gossip_count: int=1,
        pga_frequency: int=0, # TODO
        gossip_segments: int=1, # TODO

        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = False,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
            will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.gossip_count = gossip_count
        self.pga_frequency = pga_frequency
        self.gossip_segments = gossip_segments

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        self.preserved_clients = {} # Even when clients don't participate in a round, we want to preserve their parameters
        self.client_data_counts = {} # Keep track of how many data points each client has for weighting

    def __repr__(self) -> str:
        rep = f"GossipAvg(gossip_count={self.gossip_count})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is not None:
            raise TypeError("GossipAvg does not support centralised evaluation.")

    def configure_fit(
        self, server_round: int, parameters: Dict[int, Parameters], client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        if isinstance(parameters, Parameters):
            print("[GossipAvg] Received initial parameters from server, using these on all clients.")
            self.initial_parameters = parameters
            parameters = {}

        # Unfreeze clients which didn't participate in the last round
        for cid, params in self.preserved_clients.items():
            if cid not in parameters:
                parameters[cid] = params
        print("[GossipAvg] Preserved clients:", len(self.preserved_clients.keys()))

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # print(clients)

        # Return client/config pairs
        client_dispatch = [(client, FitIns(parameters.get(client.cid, self.initial_parameters), config)) for client in clients]
        # print(client_dispatch)
        return client_dispatch

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        # evaluate_ins = EvaluateIns(parameters, config)

        if isinstance(parameters, Parameters):
            print("[GossipAvg] Received initial parameters from server, using these on all clients.")
            self.initial_parameters = parameters
            parameters = {}

        # Unfreeze clients which didn't participate in the last round
        for cid, params in self.preserved_clients.items():
            if cid not in parameters:
                parameters[cid] = params

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        # return [(client, evaluate_ins) for client in clients]
        return [(client, EvaluateIns(parameters.get(client.cid, self.initial_parameters), config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        # weights_results = [
        #     (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        #     for _, fit_res in results
        # ]

        for client, fit_res in results:
            self.client_data_counts[client.cid] = fit_res.num_examples

        # aggregate_gossip goes from List[(ClientProxy, FitRes)] -> Dict[cid, params] (aggregated)
        print('[GossipAvg] Aggregating fit results, received {} results'.format(len(results)))
        if self.pga_frequency > 0 and server_round % self.pga_frequency == 0:
            print('[GossipAvg] Performing GLOBAL average'.format(len(results)))
            for client, fit_res in results:
                self.preserved_clients[client.cid] = fit_res.parameters

            parameters_dict, communication_cost = aggregate_gossip_pga(self.preserved_clients, self.client_data_counts)

            # Reset preserved clients - this will cause all clients to load the global model on the next round
            self.preserved_clients = {}
            # We will return just a single set of Parameters, which will be set as the default on next iter.
        else:
            if self.gossip_segments > 1:
                parameters_dict, communication_cost = aggregate_gossip_segmented(results, gossip_count=self.gossip_count, segments=self.gossip_segments)
            else:
                parameters_dict, communication_cost = aggregate_gossip(results, gossip_count=self.gossip_count)

            # Preserve client params where necessary
            for cid, params in parameters_dict.items():
                self.preserved_clients[cid] = params

        # parameters_dict, communication_cost = aggregate_gossip_full(results, self.preserved_clients, self.client_data_counts, gossip_count=self.gossip_count)
        print('[GossipAvg] Aggregated fit results, have {} weights. Total communication cost={}'.format(len(parameters_dict) if hasattr(parameters_dict, 'len') else 'GLOBAL', communication_cost))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        # print('fit_metrics:', [res.metrics for _,res in results])
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        metrics_aggregated['communication'] = communication_cost

        return parameters_dict, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        # print('eval_metrics:', [res.metrics for _,res in results])
        if self.evaluate_metrics_aggregation_fn:    
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
