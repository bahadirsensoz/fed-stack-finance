import os
import random

import numpy as np
import pandas as pd
import torch

from sequence_dataset import build_client_datasets
from client import FederatedClient
from server import FederatedServer


CLIENT_FILES = [
    "client_1_banks.csv",
    "client_2_transport_auto.csv",
    "client_3_defense_glass.csv",
    "client_4_holding_refinery.csv",
    "client_5_steel_retail.csv",
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_clients(device):
    clients = []

    for filename in CLIENT_FILES:
        client_path = os.path.join("data", "clients", filename)

        train_dataset, test_dataset, _ = build_client_datasets(
            client_path,
            sequence_length=60,
        )

        client_id = filename.replace(".csv", "")

        client = FederatedClient(
            client_id=client_id,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=device,
            batch_size=32,
            learning_rate=0.001,
        )

        clients.append(client)

        print(
            f"Loaded {client_id}: "
            f"train={len(train_dataset)}, test={len(test_dataset)}"
        )

    return clients


def save_final_local_weights(last_client_results):
    os.makedirs("results/local_weights", exist_ok=True)

    for result in last_client_results:
        client_id = result["client_id"]
        weights = result["weights"]

        output_path = os.path.join(
            "results",
            "local_weights",
            f"{client_id}_local_weights.pt",
        )

        torch.save(weights, output_path)

        print(f"Saved: {output_path}")


def run_federated_training(num_rounds=30, local_epochs=2):
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    clients = build_clients(device)
    server = FederatedServer(device=device)

    history = []
    last_client_results = None

    for round_idx in range(1, num_rounds + 1):
        print("=" * 70)
        print(f"Federated round {round_idx}/{num_rounds}")

        global_weights = server.get_global_weights()
        client_results = []

        for client in clients:
            client.set_model_weights(global_weights)

            result = client.train_local(epochs=local_epochs)

            client_results.append(result)

            print(
                f"{client.client_id} | "
                f"local_train_loss={result['train_loss']:.6f}"
            )

        last_client_results = client_results

        server.aggregate_weights(client_results)

        round_record = {
            "round": round_idx,
        }

        print("-" * 70)
        print("Global model evaluation:")

        global_eval_values = []

        for client in clients:
            global_eval = server.evaluate_global_on_client(client)
            global_mse = global_eval["global_test_mse"]

            global_eval_values.append(global_mse)

            round_record[f"{client.client_id}_global_mse"] = global_mse

            print(
                f"{client.client_id} | "
                f"global_test_mse={global_mse:.6f}"
            )

        avg_global_mse = sum(global_eval_values) / len(global_eval_values)

        round_record["avg_global_mse"] = avg_global_mse

        print(f"Average global MSE: {avg_global_mse:.6f}")

        history.append(round_record)

    os.makedirs("results", exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv("results/federated_history.csv", index=False)

    torch.save(
        server.get_global_weights(),
        "results/global_model_weights.pt",
    )

    print("Saved: results/global_model_weights.pt")

    if last_client_results is not None:
        save_final_local_weights(last_client_results)

    print("=" * 70)
    print("Federated training completed.")
    print("Saved: results/federated_history.csv")


if __name__ == "__main__":
    run_federated_training(
        num_rounds=30,
        local_epochs=2,
    )