import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_rmse_comparison(results_df):
    os.makedirs("results/plots", exist_ok=True)

    clients = results_df["client_id"]

    x = range(len(clients))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - width for i in x],
        results_df["local_rmse"],
        width=width,
        label="Local Model",
    )

    plt.bar(
        x,
        results_df["global_rmse"],
        width=width,
        label="Global Federated Model",
    )

    plt.bar(
        [i + width for i in x],
        results_df["fedstack_rmse"],
        width=width,
        label="FedStack",
    )

    plt.xticks(x, clients, rotation=25, ha="right")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison: Local vs Global vs FedStack")
    plt.legend()
    plt.tight_layout()

    output_path = "results/plots/rmse_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_improvement_comparison(results_df):
    os.makedirs("results/plots", exist_ok=True)

    clients = results_df["client_id"]

    x = range(len(clients))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - width / 2 for i in x],
        results_df["fedstack_improvement_over_local_rmse"],
        width=width,
        label="Improvement over Local",
    )

    plt.bar(
        [i + width / 2 for i in x],
        results_df["fedstack_improvement_over_global_rmse"],
        width=width,
        label="Improvement over Global",
    )

    plt.xticks(x, clients, rotation=25, ha="right")
    plt.ylabel("RMSE Improvement (%)")
    plt.title("FedStack Improvement Over Baselines")
    plt.legend()
    plt.tight_layout()

    output_path = "results/plots/fedstack_improvement.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    results_path = "results/fedstack_results.csv"

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            "results/fedstack_results.csv not found. "
            "Run fedstack.py first."
        )

    results_df = pd.read_csv(results_path)

    plot_rmse_comparison(results_df)
    plot_improvement_comparison(results_df)


if __name__ == "__main__":
    main()
