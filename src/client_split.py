import os
import pandas as pd


CLIENT_TICKER_MAP = {
    "client_1_banks": ["GARAN.IS", "AKBNK.IS"],
    "client_2_transport_auto": ["THYAO.IS", "FROTO.IS"],
    "client_3_defense_glass": ["ASELS.IS", "SISE.IS"],
    "client_4_holding_refinery": ["KCHOL.IS", "TUPRS.IS"],
    "client_5_steel_retail": ["EREGL.IS", "BIMAS.IS"],
}


def split_clients():
    input_path = "data/processed/bist_features.csv"
    output_dir = "data/clients"

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    for client_name, tickers in CLIENT_TICKER_MAP.items():
        client_df = df[df["Ticker"].isin(tickers)].copy()
        client_df = client_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        output_path = os.path.join(output_dir, f"{client_name}.csv")
        client_df.to_csv(output_path, index=False)

        print(f"{client_name}: {client_df.shape} -> {tickers}")

    print("Client split completed.")


if __name__ == "__main__":
    split_clients()