import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def run_script(script_path: str) -> None:
    full_path = ROOT_DIR / script_path

    if not full_path.exists():
        raise FileNotFoundError(f"Script not found: {full_path}")

    print("\n" + "=" * 80)
    print(f"Running: {script_path}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=ROOT_DIR,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_path}")

    print(f"Completed: {script_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full FedStack BIST stock prediction pipeline."
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading raw stock data."
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip federated training and use existing saved model weights."
    )

    parser.add_argument(
        "--only-plots",
        action="store_true",
        help="Only regenerate merged results and plots."
    )

    args = parser.parse_args()

    try:
        if args.only_plots:
            run_script("src/merge_results.py")
            run_script("src/plot_results.py")
            print("\nOnly plots/results were regenerated successfully.")
            return

        if not args.skip_download:
            run_script("src/download_data.py")

        run_script("src/dataset.py")
        run_script("src/client_split.py")

        if not args.skip_training:
            run_script("src/federated_train.py")

        run_script("src/fedstack.py")
        run_script("src/naive_baseline.py")
        run_script("src/merge_results.py")
        run_script("src/plot_results.py")

        print("\n" + "=" * 80)
        print("Pipeline completed successfully.")
        print("Final comparison table: results/final_comparison_table.csv")
        print("Plots: results/plots/")
        print("=" * 80)

    except Exception as exc:
        print("\nPipeline failed.")
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
