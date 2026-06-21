import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


TICKER_TO_ID = {
    "THYAO.IS": 0,
    "GARAN.IS": 1,
    "AKBNK.IS": 2,
    "ASELS.IS": 3,
    "SISE.IS": 4,
    "KCHOL.IS": 5,
    "TUPRS.IS": 6,
    "EREGL.IS": 7,
    "BIMAS.IS": 8,
    "FROTO.IS": 9,
}

NUM_TICKERS = len(TICKER_TO_ID)


SEQUENCE_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "MA7",
    "MA14",
    "MA30",
    "Volatility_7",
    "Volatility_14",
    "RSI",
    "MACD",
    "MACD_Signal",
]


STATIC_FEATURES = [
    "Daily_Return",
    "MA7",
    "MA14",
    "MA30",
    "Volatility_7",
    "Volatility_14",
    "RSI",
    "MACD",
    "MACD_Signal",
]


TARGET_COLUMN = "Target_Close"


class StockSequenceDataset(Dataset):
    def __init__(self, sequences, static_features, ticker_ids, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.static_features = torch.tensor(static_features, dtype=torch.float32)
        self.ticker_ids = torch.tensor(ticker_ids, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            "sequence": self.sequences[index],
            "static": self.static_features[index],
            "ticker_id": self.ticker_ids[index],
            "target": self.targets[index],
        }


def create_sequences_for_ticker(
    ticker_df,
    ticker_id,
    sequence_length=60,
    sequence_scaler=None,
    static_scaler=None,
    target_scaler=None,
    fit_scalers=True,
):
    ticker_df = ticker_df.sort_values("Date").reset_index(drop=True).copy()

    sequence_data = ticker_df[SEQUENCE_FEATURES].values
    static_data = ticker_df[STATIC_FEATURES].values
    target_data = ticker_df[[TARGET_COLUMN]].values

    if fit_scalers:
        sequence_scaler = StandardScaler()
        static_scaler = StandardScaler()
        target_scaler = StandardScaler()

        sequence_data = sequence_scaler.fit_transform(sequence_data)
        static_data = static_scaler.fit_transform(static_data)
        target_data = target_scaler.fit_transform(target_data)
    else:
        sequence_data = sequence_scaler.transform(sequence_data)
        static_data = static_scaler.transform(static_data)
        target_data = target_scaler.transform(target_data)

    sequences = []
    static_features = []
    ticker_ids = []
    targets = []

    for i in range(sequence_length, len(ticker_df)):
        sequence_window = sequence_data[i - sequence_length:i]
        static_vector = static_data[i]
        target = target_data[i]

        sequences.append(sequence_window)
        static_features.append(static_vector)
        ticker_ids.append(ticker_id)
        targets.append(target)

    return (
        np.array(sequences),
        np.array(static_features),
        np.array(ticker_ids),
        np.array(targets),
        sequence_scaler,
        static_scaler,
        target_scaler,
    )


def train_test_split_by_time(df, test_ratio=0.2):
    df = df.sort_values("Date").reset_index(drop=True)

    split_index = int(len(df) * (1 - test_ratio))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def build_client_datasets(client_csv_path, sequence_length=60, test_ratio=0.2):
    df = pd.read_csv(client_csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    all_train_sequences = []
    all_train_static = []
    all_train_ticker_ids = []
    all_train_targets = []

    all_test_sequences = []
    all_test_static = []
    all_test_ticker_ids = []
    all_test_targets = []

    scalers = {}

    for ticker, ticker_df in df.groupby("Ticker"):
        if ticker not in TICKER_TO_ID:
            print(f"Skipping {ticker}: ticker not found in TICKER_TO_ID.")
            continue

        ticker_id = TICKER_TO_ID[ticker]

        ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)

        train_df, test_df = train_test_split_by_time(
            ticker_df,
            test_ratio=test_ratio,
        )

        if len(train_df) <= sequence_length or len(test_df) <= sequence_length:
            print(f"Skipping {ticker}: not enough data.")
            continue

        (
            train_sequences,
            train_static,
            train_ticker_ids,
            train_targets,
            sequence_scaler,
            static_scaler,
            target_scaler,
        ) = create_sequences_for_ticker(
            train_df,
            ticker_id=ticker_id,
            sequence_length=sequence_length,
            fit_scalers=True,
        )

        (
            test_sequences,
            test_static,
            test_ticker_ids,
            test_targets,
            _,
            _,
            _,
        ) = create_sequences_for_ticker(
            test_df,
            ticker_id=ticker_id,
            sequence_length=sequence_length,
            sequence_scaler=sequence_scaler,
            static_scaler=static_scaler,
            target_scaler=target_scaler,
            fit_scalers=False,
        )

        all_train_sequences.append(train_sequences)
        all_train_static.append(train_static)
        all_train_ticker_ids.append(train_ticker_ids)
        all_train_targets.append(train_targets)

        all_test_sequences.append(test_sequences)
        all_test_static.append(test_static)
        all_test_ticker_ids.append(test_ticker_ids)
        all_test_targets.append(test_targets)

        scalers[ticker] = {
            "sequence_scaler": sequence_scaler,
            "static_scaler": static_scaler,
            "target_scaler": target_scaler,
        }

    if not all_train_sequences:
        raise RuntimeError(f"No valid ticker data found in {client_csv_path}")

    train_dataset = StockSequenceDataset(
        np.concatenate(all_train_sequences, axis=0),
        np.concatenate(all_train_static, axis=0),
        np.concatenate(all_train_ticker_ids, axis=0),
        np.concatenate(all_train_targets, axis=0),
    )

    test_dataset = StockSequenceDataset(
        np.concatenate(all_test_sequences, axis=0),
        np.concatenate(all_test_static, axis=0),
        np.concatenate(all_test_ticker_ids, axis=0),
        np.concatenate(all_test_targets, axis=0),
    )

    return train_dataset, test_dataset, scalers


def test_build_all_clients(sequence_length=60):
    clients_dir = "data/clients"

    for filename in os.listdir(clients_dir):
        if not filename.endswith(".csv"):
            continue

        client_path = os.path.join(clients_dir, filename)

        train_dataset, test_dataset, _ = build_client_datasets(
            client_path,
            sequence_length=sequence_length,
        )

        sample = train_dataset[0]

        print("=" * 60)
        print(filename)
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Sequence shape: {sample['sequence'].shape}")
        print(f"Static shape: {sample['static'].shape}")
        print(f"Ticker id shape: {sample['ticker_id'].shape}")
        print(f"Target shape: {sample['target'].shape}")


if __name__ == "__main__":
    test_build_all_clients(sequence_length=60)