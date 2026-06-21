import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import HybridStockPredictor


class FederatedClient:
    def __init__(
        self,
        client_id,
        train_dataset,
        test_dataset,
        device="cpu",
        batch_size=32,
        learning_rate=0.001,
    ):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = HybridStockPredictor().to(self.device)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.criterion = nn.MSELoss()

    def set_model_weights(self, global_weights):
        self.model.load_state_dict(copy.deepcopy(global_weights))

    def get_model_weights(self):
        return copy.deepcopy(self.model.state_dict())

    def train_local(self, epochs=1):
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        total_loss = 0.0
        total_batches = 0

        for _ in range(epochs):
            for batch in self.train_loader:
                sequence = batch["sequence"].to(self.device)
                static = batch["static"].to(self.device)
                ticker_id = batch["ticker_id"].to(self.device)
                target = batch["target"].to(self.device)

                optimizer.zero_grad()

                prediction = self.model(sequence, static, ticker_id)

                loss = self.criterion(prediction, target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)

        return {
            "client_id": self.client_id,
            "weights": self.get_model_weights(),
            "num_samples": len(self.train_dataset),
            "train_loss": avg_loss,
        }

    def evaluate(self):
        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in self.test_loader:
                sequence = batch["sequence"].to(self.device)
                static = batch["static"].to(self.device)
                ticker_id = batch["ticker_id"].to(self.device)
                target = batch["target"].to(self.device)

                prediction = self.model(sequence, static, ticker_id)

                loss = self.criterion(prediction, target)

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)

        return {
            "client_id": self.client_id,
            "test_mse": avg_loss,
        }

    def predict_test(self):
        self.model.eval()

        predictions = []
        targets = []

        with torch.no_grad():
            for batch in self.test_loader:
                sequence = batch["sequence"].to(self.device)
                static = batch["static"].to(self.device)
                ticker_id = batch["ticker_id"].to(self.device)
                target = batch["target"].to(self.device)

                prediction = self.model(sequence, static, ticker_id)

                predictions.append(prediction.cpu())
                targets.append(target.cpu())

        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        return predictions, targets

    def predict_with_model_weights(self, model_weights):
        original_weights = self.get_model_weights()

        self.set_model_weights(model_weights)
        predictions, targets = self.predict_test()

        self.set_model_weights(original_weights)

        return predictions, targets