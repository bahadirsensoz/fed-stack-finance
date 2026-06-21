import os
import pandas as pd


def main():
    fedstack_path = "results/fedstack_results.csv"
    naive_path = "results/naive_baseline_results.csv"

    if not os.path.exists(fedstack_path):
        raise FileNotFoundError("Run fedstack.py first.")

    if not os.path.exists(naive_path):
        raise FileNotFoundError("Run naive_baseline.py first.")

    fedstack_df = pd.read_csv(fedstack_path)
    naive_df = pd.read_csv(naive_path)

    merged = fedstack_df.merge(
        naive_df,
        on="client_id",
        how="left",
    )

    selected = merged[
        [
            "client_id",
            "naive_rmse",
            "local_rmse",
            "global_rmse",
            "fedstack_rmse",
            "fedstack_improvement_over_local_rmse",
            "fedstack_improvement_over_global_rmse",
        ]
    ]

    selected.to_csv("results/final_comparison_table.csv", index=False)

    print("Saved: results/final_comparison_table.csv")
    print(selected)


if __name__ == "__main__":
    main()