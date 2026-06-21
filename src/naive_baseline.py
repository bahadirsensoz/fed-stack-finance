import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sequence_dataset import build_client_datasets


CLIENT_FILES = [
    "client_1_banks.csv",
    "client_2_transport_auto.csv",
    "client_3_defense_glass.csv",
    "client_4_holding_refinery.csv",
    "client_5_steel_retail.csv",
]


CLOSE_FEATURE_INDEX = 3


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_predictions(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def evaluate_naive_for_client(client_file):
    client_path = os.path.join("data", "clients", client_file)

    _, test_dataset, _ = build_client_datasets(
        client_path,
        sequence_length=60,
    )

    naive_predictions = []
    targets = []

    for sample in test_dataset:
        sequence = sample["sequence"].numpy()
        target = sample["target"].numpy()

        last_close = sequence[-1, CLOSE_FEATURE_INDEX]

        naive_predictions.append([last_close])
        targets.append(target)

    naive_predictions = np.array(naive_predictions)
    targets = np.array(targets).reshape(-1, 1)

    metrics = evaluate_predictions(
        targets,
        naive_predictions,
    )

    return metrics


def main():
    os.makedirs("results", exist_ok=True)

    results = []

    for client_file in CLIENT_FILES:
        client_id = client_file.replace(".csv", "")

        metrics = evaluate_naive_for_client(client_file)

        print("=" * 70)
        print(f"Naive baseline for {client_id}")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")

        results.append({
            "client_id": client_id,
            "naive_mse": metrics["mse"],
            "naive_rmse": metrics["rmse"],
            "naive_mae": metrics["mae"],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/naive_baseline_results.csv", index=False)

    print("=" * 70)
    print("Saved: results/naive_baseline_results.csv")
    print(results_df)


if __name__ == "__main__":
    main()