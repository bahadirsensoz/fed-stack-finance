````markdown
# Privacy-Preserving Federated Stock Price Prediction on BIST Data Using FedStack

This project implements a privacy-preserving federated learning pipeline for next-day stock closing price prediction on BIST stocks. The system simulates a federated environment with multiple virtual clients, where each client owns a different subset of stock data. Raw client data is not shared with the server. Instead, only model weights are exchanged during federated training.

The project combines a hybrid LSTM-MLP neural network with FedAvg and a FedStack personalization layer. The LSTM branch learns temporal patterns from 60-day stock sequences, while the MLP branch processes static technical indicators. A ticker embedding layer is also used to help the model distinguish stock-specific behavior.

## Project Overview

The main goal is to predict the next-day normalized closing price of selected BIST stocks while keeping client data local.

The pipeline includes:

1. Downloading BIST stock data from Yahoo Finance
2. Generating technical indicators
3. Splitting stocks into non-IID virtual clients
4. Training a hybrid LSTM-MLP model in a federated setting
5. Aggregating local models with FedAvg
6. Applying FedStack with Ridge regression to combine local and global predictions
7. Comparing FedStack against naive, local-only, and global FedAvg baselines
8. Generating result tables and plots

## Method Summary

The project uses four main prediction approaches:

- **Naive persistence baseline:** predicts the next-day closing price as the current closing price.
- **Local model:** each client's final local model after federated training.
- **Global FedAvg model:** the global model obtained by aggregating locally trained client weights.
- **FedStack model:** a personalized model that combines local and global predictions using Ridge regression.

FedStack is the final proposed method. It uses both client-specific knowledge from the local model and global market knowledge from the federated model.

## Dataset

The experiment uses 10 BIST stocks:

- THYAO.IS
- GARAN.IS
- AKBNK.IS
- ASELS.IS
- SISE.IS
- KCHOL.IS
- TUPRS.IS
- EREGL.IS
- BIMAS.IS
- FROTO.IS

These stocks are split into five virtual clients:

| Client | Stocks |
|---|---|
| client_1_banks | GARAN.IS, AKBNK.IS |
| client_2_transport_auto | THYAO.IS, FROTO.IS |
| client_3_defense_glass | ASELS.IS, SISE.IS |
| client_4_holding_refinery | KCHOL.IS, TUPRS.IS |
| client_5_steel_retail | EREGL.IS, BIMAS.IS |

This creates a non-IID federated setting because each client observes a different part of the market.

## Features

The model uses sequential and static technical features.

Sequential features include:

- Open
- High
- Low
- Close
- Volume
- Daily Return
- Moving averages
- Volatility
- RSI
- MACD
- MACD Signal

The sequence length is 60 trading days. The target variable is the next-day closing price.

## Project Structure

```text
fed-stack-finance/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── clients/
│
├── results/
│   ├── local_weights/
│   ├── plots/
│   ├── federated_history.csv
│   ├── global_model_weights.pt
│   ├── fedstack_results.csv
│   ├── naive_baseline_results.csv
│   └── final_comparison_table.csv
│
├── src/
│   ├── download_data.py
│   ├── indicators.py
│   ├── dataset.py
│   ├── client_split.py
│   ├── sequence_dataset.py
│   ├── model.py
│   ├── client.py
│   ├── server.py
│   ├── federated_train.py
│   ├── fedstack.py
│   ├── naive_baseline.py
│   ├── merge_results.py
│   └── plot_results.py
│
├── main.py
├── README.md
└── requirements.txt
````

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
pandas
scikit-learn
torch
matplotlib
yfinance
```

## Running the Project Step by Step

Run the scripts in the following order:

```bash
python src/download_data.py
python src/dataset.py
python src/client_split.py
python src/federated_train.py
python src/fedstack.py
python src/merge_results.py
python src/plot_results.py
```

## Running the Full Pipeline

The whole pipeline can also be executed with:

```bash
python main.py
```

To skip downloading data again:

```bash
python main.py --skip-download
```

To skip federated training and only regenerate evaluation results and plots from existing saved models:

```bash
python main.py --skip-download --skip-training
```

## Results

The final comparison table is saved as:

```text
results/final_comparison_table.csv
```

The main plots are saved as:

```text
results/plots/rmse_comparison.png
results/plots/fedstack_improvement.png
```

The final results show that FedStack outperforms both neural baselines, local-only and global FedAvg, across all five client groups. It also outperforms the naive persistence baseline in four out of five client groups.

## Privacy Note

The project simulates a privacy-preserving federated learning setup. Raw client data remains local, and the server only receives model weights. However, federated learning alone does not provide a complete cryptographic privacy guarantee. Future improvements can include secure aggregation and differential privacy.

