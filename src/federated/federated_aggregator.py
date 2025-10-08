"""
Federated learning aggregation using FedAvg algorithm.
"""

import torch
import copy
from typing import List, Dict
from src.models.node_model import NodeModel


def federated_averaging(node_models: List[NodeModel]) -> Dict:
    """
    FedAvg: Simple weighted averaging of model parameters.

    Args:
        node_models: List of node model instances

    Returns:
        global_weights: Averaged model parameters
    """
    if not node_models:
        raise ValueError("No node models provided for aggregation")

    total_samples = sum(
        len(node.training_data) if node.training_data is not None else 0 for node in node_models
    )

    if total_samples == 0:
        total_samples = len(node_models)
        weights_per_node = [1.0 / len(node_models)] * len(node_models)
    else:
        weights_per_node = [
            len(node.training_data) / total_samples if node.training_data is not None else 0
            for node in node_models
        ]

    first_model = node_models[0].model.state_dict()
    global_weights = {}

    for key in first_model.keys():
        global_weights[key] = torch.zeros_like(first_model[key])

    for node, weight in zip(node_models, weights_per_node):
        local_weights = node.model.state_dict()

        for key in local_weights.keys():
            global_weights[key] += local_weights[key] * weight

    return global_weights


def federated_update_cycle(
    node_models: List[NodeModel], epochs: int = 5, verbose: bool = False
) -> Dict:
    """
    Complete federated learning cycle.

    1. Nodes train locally
    2. Central aggregates weights
    3. Central distributes updated model

    Args:
        node_models: List of node models
        epochs: Number of local training epochs
        verbose: Print progress

    Returns:
        global_weights: Updated global model weights
    """
    if verbose:
        print(f"\nFederated learning update cycle:")
        print(f"  Nodes: {len(node_models)}")
        print(f"  Local epochs: {epochs}")

    for i, node in enumerate(node_models):
        if node.training_data is not None and len(node.training_data) > 100:
            if verbose:
                print(f"  Training node {node.household_id}...", end=" ")

            node.train(
                node.training_data,
                epochs=epochs,
                batch_size=node.config.get("batch_size", 32),
                learning_rate=node.config.get("learning_rate", 0.001),
                verbose=False,
            )

            if verbose:
                print("Done")

    if verbose:
        print("  Aggregating weights...")

    global_weights = federated_averaging(node_models)

    if verbose:
        print("  Distributing updated model...")

    for node in node_models:
        node.update_model_weights(global_weights)

    if verbose:
        print("  Federated update complete")

    return global_weights


def calculate_weight_divergence(node_models: List[NodeModel]) -> float:
    """
    Calculate divergence between node models to assess convergence.

    Args:
        node_models: List of node models

    Returns:
        divergence: Mean parameter divergence across nodes
    """
    if len(node_models) < 2:
        return 0.0

    reference_weights = node_models[0].get_model_weights()
    divergences = []

    for node in node_models[1:]:
        node_weights = node.get_model_weights()

        div = 0.0
        num_params = 0

        for key in reference_weights.keys():
            if "weight" in key or "bias" in key:
                diff = torch.norm(reference_weights[key] - node_weights[key])
                div += diff.item()
                num_params += 1

        if num_params > 0:
            divergences.append(div / num_params)

    return sum(divergences) / len(divergences) if divergences else 0.0


if __name__ == "__main__":
    import sys

    sys.path.append("../..")

    config = {
        "lstm_hidden_size": 64,
        "lstm_num_layers": 2,
        "dropout": 0.2,
        "batch_size": 32,
        "prediction_horizon_intervals": 6,
        "input_sequence_length": 48,
        "learning_rate": 0.001,
    }

    print("Creating node models...")
    nodes = [NodeModel(household_id=i, config=config) for i in range(1, 4)]

    print("Aggregating weights...")
    global_weights = federated_averaging(nodes)

    print(f"Global weights keys: {list(global_weights.keys())[:3]}...")
    print(f"Weight divergence: {calculate_weight_divergence(nodes):.6f}")

    print("\nFederated averaging test complete")
