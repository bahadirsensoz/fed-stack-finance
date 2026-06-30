import os
import random

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

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


META_TRAIN_RATIO = 0.7


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_weights(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse_value = rmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse_value,
        "mae": mae,
    }


def safe_improvement_percentage(baseline_rmse, fedstack_rmse):
    if baseline_rmse == 0:
        return 0.0

    return ((baseline_rmse - fedstack_rmse) / baseline_rmse) * 100


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


def load_final_local_weights(clients, device):
    local_weights = {}

    for client in clients:
        path = os.path.join(
            "results",
            "local_weights",
            f"{client.client_id}_local_weights.pt",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Local weights not found for {client.client_id}. "
                "Run federated_train.py first."
            )

        local_weights[client.client_id] = load_weights(path, device)

        print(f"Loaded local weights: {path}")

    return local_weights


def chronological_meta_split_by_ticker(
    features,
    targets,
    ticker_ids,
    train_ratio=META_TRAIN_RATIO,
):
    train_indices = []
    evaluation_indices = []

    for ticker_id in np.unique(ticker_ids):
        ticker_indices = np.where(ticker_ids == ticker_id)[0]
        ticker_indices = np.sort(ticker_indices)

        split_index = int(len(ticker_indices) * train_ratio)

        train_indices.extend(ticker_indices[:split_index])
        evaluation_indices.extend(ticker_indices[split_index:])

    train_indices = np.array(train_indices)
    evaluation_indices = np.array(evaluation_indices)

    x_meta_train = features[train_indices]
    y_meta_train = targets[train_indices]

    x_meta_eval = features[evaluation_indices]
    y_meta_eval = targets[evaluation_indices]

    return x_meta_train, x_meta_eval, y_meta_train, y_meta_eval


def run_fedstack():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    clients = build_clients(device)

    server = FederatedServer(device=device)

    global_weights_path = "results/global_model_weights.pt"

    if not os.path.exists(global_weights_path):
        raise FileNotFoundError(
            "Global model weights not found. "
            "Run federated_train.py first."
        )

    global_weights = load_weights(
        global_weights_path,
        device,
    )

    server.set_global_weights(global_weights)

    local_weights = load_final_local_weights(
        clients,
        device,
    )

    results = []

    for client in clients:
        print("=" * 70)
        print(f"FedStack evaluation for {client.client_id}")

        client.set_model_weights(local_weights[client.client_id])

        local_pred, targets = client.predict_test()

        global_pred, _ = client.predict_with_model_weights(
            server.get_global_weights()
        )

        ticker_ids = client.test_dataset.ticker_ids.detach().cpu().numpy()

        stack_features = np.concatenate(
            [local_pred, global_pred],
            axis=1,
        )

        x_meta_train, x_meta_eval, y_meta_train, y_meta_eval = (
            chronological_meta_split_by_ticker(
                stack_features,
                targets,
                ticker_ids,
                train_ratio=META_TRAIN_RATIO,
            )
        )

        print(
            f"Meta split: train={len(y_meta_train)}, "
            f"evaluation={len(y_meta_eval)}"
        )

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(
            x_meta_train,
            y_meta_train.ravel(),
        )

        fedstack_pred = meta_model.predict(
            x_meta_eval
        ).reshape(-1, 1)

        local_eval_pred = x_meta_eval[:, 0].reshape(-1, 1)
        global_eval_pred = x_meta_eval[:, 1].reshape(-1, 1)

        local_metrics = evaluate_predictions(
            y_meta_eval,
            local_eval_pred,
        )

        global_metrics = evaluate_predictions(
            y_meta_eval,
            global_eval_pred,
        )

        fedstack_metrics = evaluate_predictions(
            y_meta_eval,
            fedstack_pred,
        )

        improvement_over_local = safe_improvement_percentage(
            local_metrics["rmse"],
            fedstack_metrics["rmse"],
        )

        improvement_over_global = safe_improvement_percentage(
            global_metrics["rmse"],
            fedstack_metrics["rmse"],
        )

        print(f"Local RMSE: {local_metrics['rmse']:.6f}")
        print(f"Global RMSE: {global_metrics['rmse']:.6f}")
        print(f"FedStack RMSE: {fedstack_metrics['rmse']:.6f}")

        print(
            f"FedStack improvement over Local RMSE: "
            f"{improvement_over_local:.2f}%"
        )

        print(
            f"FedStack improvement over Global RMSE: "
            f"{improvement_over_global:.2f}%"
        )

        results.append({
            "client_id": client.client_id,

            "local_mse": local_metrics["mse"],
            "local_rmse": local_metrics["rmse"],
            "local_mae": local_metrics["mae"],

            "global_mse": global_metrics["mse"],
            "global_rmse": global_metrics["rmse"],
            "global_mae": global_metrics["mae"],

            "fedstack_mse": fedstack_metrics["mse"],
            "fedstack_rmse": fedstack_metrics["rmse"],
            "fedstack_mae": fedstack_metrics["mae"],

            "fedstack_improvement_over_local_rmse": improvement_over_local,
            "fedstack_improvement_over_global_rmse": improvement_over_global,

            "meta_weight_local": meta_model.coef_[0],
            "meta_weight_global": meta_model.coef_[1],
            "meta_intercept": meta_model.intercept_,
        })

    os.makedirs("results", exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/fedstack_results.csv", index=False)

    print("=" * 70)
    print("FedStack completed.")
    print("Saved: results/fedstack_results.csv")
    print(results_df)


if __name__ == "__main__":
    run_fedstack()