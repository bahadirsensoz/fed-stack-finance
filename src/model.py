import torch
import torch.nn as nn


class HybridStockPredictor(nn.Module):
    def __init__(
        self,
        sequence_input_size=14,
        static_input_size=9,
        num_tickers=10,
        ticker_embedding_dim=4,
        lstm_hidden_size=128,
        lstm_num_layers=1,
        mlp_hidden_size=64,
        fusion_hidden_size=128,
        dropout=0.2,
    ):
        super().__init__()

        self.ticker_embedding = nn.Embedding(
            num_embeddings=num_tickers,
            embedding_dim=ticker_embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=sequence_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.lstm_branch = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.static_branch = nn.Sequential(
            nn.Linear(static_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 32),
            nn.ReLU(),
        )

        fusion_input_size = 128 + 32 + ticker_embedding_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, 1),
        )

    def forward(self, sequence, static, ticker_id):
        lstm_output, _ = self.lstm(sequence)

        last_hidden = lstm_output[:, -1, :]

        lstm_features = self.lstm_branch(last_hidden)
        static_features = self.static_branch(static)
        ticker_features = self.ticker_embedding(ticker_id)

        combined = torch.cat(
            [lstm_features, static_features, ticker_features],
            dim=1,
        )

        prediction = self.fusion(combined)

        return prediction


def test_model():
    batch_size = 16
    sequence_length = 60
    sequence_features = 14
    static_features = 9
    num_tickers = 10

    model = HybridStockPredictor(
        sequence_input_size=sequence_features,
        static_input_size=static_features,
        num_tickers=num_tickers,
    )

    dummy_sequence = torch.randn(
        batch_size,
        sequence_length,
        sequence_features,
    )

    dummy_static = torch.randn(
        batch_size,
        static_features,
    )

    dummy_ticker_ids = torch.randint(
        low=0,
        high=num_tickers,
        size=(batch_size,),
    )

    output = model(
        dummy_sequence,
        dummy_static,
        dummy_ticker_ids,
    )

    print(model)
    print(f"Input sequence shape: {dummy_sequence.shape}")
    print(f"Input static shape: {dummy_static.shape}")
    print(f"Ticker id shape: {dummy_ticker_ids.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test_model()