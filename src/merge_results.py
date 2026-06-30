import os
import pandas as pd


def main():
    fedstack_path = "results/fedstack_results.csv"

    if not os.path.exists(fedstack_path):
        raise FileNotFoundError("Run fedstack.py first.")

    fedstack_df = pd.read_csv(fedstack_path)

    selected = fedstack_df[
        [
            "client_id",
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