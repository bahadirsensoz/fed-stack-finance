import pandas as pd


def ensure_numeric_columns(df):
    df = df.copy()

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def add_moving_averages(df):
    df["MA7"] = df["Close"].rolling(window=7).mean()
    df["MA14"] = df["Close"].rolling(window=14).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    return df


def add_daily_return(df):
    df["Daily_Return"] = df["Close"].pct_change(fill_method=None)
    return df


def add_volatility(df):
    df["Volatility_7"] = df["Daily_Return"].rolling(window=7).std()
    df["Volatility_14"] = df["Daily_Return"].rolling(window=14).std()
    return df


def add_rsi(df, period=14):
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def add_macd(df):
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def add_target(df):
    next_close = df["Close"].shift(-1)

    df["Target_Close"] = next_close
    df["Target_Return"] = (next_close - df["Close"]) / df["Close"]

    return df


def add_all_indicators(df):
    df = ensure_numeric_columns(df)

    df = add_daily_return(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_target(df)

    df = df.dropna().reset_index(drop=True)

    return df