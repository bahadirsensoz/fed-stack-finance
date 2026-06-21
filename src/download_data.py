import os
import yfinance as yf
import pandas as pd


BIST_TICKERS = [
    "THYAO.IS",
    "GARAN.IS",
    "AKBNK.IS",
    "ASELS.IS",
    "SISE.IS",
    "KCHOL.IS",
    "TUPRS.IS",
    "EREGL.IS",
    "BIMAS.IS",
    "FROTO.IS",
]


def clean_yfinance_df(df, ticker):
    df = df.copy()

    # yfinance bazen MultiIndex column döndürebiliyor
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Sadece ihtiyacımız olan kolonları al
    required_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    existing_columns = [col for col in required_columns if col in df.columns]
    df = df[existing_columns]

    df["Ticker"] = ticker

    # Numeric olması gereken kolonları garantiye al
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def download_stock_data(tickers, start_date="2018-01-01", end_date="2025-12-31"):
    os.makedirs("data/raw", exist_ok=True)

    all_frames = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="column"
        )

        if df.empty:
            print(f"No data found for {ticker}")
            continue

        df = clean_yfinance_df(df, ticker)

        if df.empty:
            print(f"Cleaned data is empty for {ticker}")
            continue

        output_path = f"data/raw/{ticker.replace('.', '_')}.csv"
        df.to_csv(output_path, index=False)

        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No stock data downloaded.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv("data/raw/bist_combined.csv", index=False)

    print("Data download completed.")
    print(f"Combined shape: {combined.shape}")
    print(combined.head())


if __name__ == "__main__":
    download_stock_data(BIST_TICKERS)