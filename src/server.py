import copy
import torch

from model import HybridStockPredictor


class FederatedServer:
    def __init__(self, device="cpu"):
        self.device = device
        self.global_model = HybridStockPredictor().to(self.device)

    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

    def set_global_weights(self, weights):
        self.global_model.load_state_dict(copy.deepcopy(weights))

    def aggregate_weights(self, client_results):
        """
        FedAvg aggregation.

        client_results contains:
        {
            "client_id": ...,
            "weights": ...,
            "num_samples": ...,
            "train_loss": ...
        }
        """

        total_samples = sum(result["num_samples"] for result in client_results)

        new_global_weights = copy.deepcopy(client_results[0]["weights"])

        for key in new_global_weights.keys():
            new_global_weights[key] = torch.zeros_like(new_global_weights[key])

        for result in client_results:
            client_weight = result["num_samples"] / total_samples
            client_state = result["weights"]

            for key in new_global_weights.keys():
                new_global_weights[key] += client_state[key] * client_weight

        self.set_global_weights(new_global_weights)

        return new_global_weights

    def evaluate_global_on_client(self, client):
        original_client_weights = client.get_model_weights()

        client.set_model_weights(self.get_global_weights())
        result = client.evaluate()

        client.set_model_weights(original_client_weights)

        return {
            "client_id": client.client_id,
            "global_test_mse": result["test_mse"],
        }