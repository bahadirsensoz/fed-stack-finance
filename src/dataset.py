import os
import pandas as pd
from indicators import add_all_indicators


def build_processed_dataset():
    os.makedirs("data/processed", exist_ok=True)

    raw_path = "data/raw/bist_combined.csv"

    df = pd.read_csv(raw_path, low_memory=False)

    # Olası bozuk kolon/satır problemlerine karşı
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"])

    processed_frames = []

    for ticker, group in df.groupby("Ticker"):
        print(f"Processing {ticker}...")

        processed = add_all_indicators(group)

        if processed.empty:
            print(f"Skipping {ticker}, not enough data after indicators.")
            continue

        processed_frames.append(processed)

    if not processed_frames:
        raise RuntimeError("No processed data created.")

    final_df = pd.concat(processed_frames, ignore_index=True)

    output_path = "data/processed/bist_features.csv"
    final_df.to_csv(output_path, index=False)

    print("Processed dataset created.")
    print(f"Shape: {final_df.shape}")
    print(final_df.head())


if __name__ == "__main__":
    build_processed_dataset()