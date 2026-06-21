import os
import torch

from sequence_dataset import build_client_datasets
from client import FederatedClient


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    client_path = os.path.join("data", "clients", "client_1_banks.csv")

    train_dataset, test_dataset, _ = build_client_datasets(
        client_path,
        sequence_length=60,
    )

    client = FederatedClient(
        client_id="client_1_banks",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device=device,
        batch_size=32,
        learning_rate=0.001,
    )

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    before_eval = client.evaluate()
    print(f"Before training: {before_eval}")

    result = client.train_local(epochs=1)
    print(f"Train result: client={result['client_id']}, loss={result['train_loss']:.6f}")

    after_eval = client.evaluate()
    print(f"After training: {after_eval}")


if __name__ == "__main__":
    main()