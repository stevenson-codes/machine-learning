import pandas as pd

def load_csv(filename: str) -> pd.DataFrame:
    """Load CSV file into a DataFrame."""
    df = pd.read_csv(filename)
    df.sort_values(['symbol', 'date'], ascending=[True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[DEBUG] Loaded {len(df)} records. Shape: {df.shape}")
    return df

def save_csv(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to a CSV file."""
    df.to_csv(filename, index=False)


def add_ema(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Add Exponential Moving Average (EMA) column to DataFrame."""
    df[f'ema_{span}'] = df.groupby('symbol')['adjClose'].transform(lambda x: x.ewm(span=span, adjust=False).mean())
    return df


def add_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add Relative Strength Index (RSI) column to DataFrame."""
    df['delta'] = df.groupby('symbol')['adjClose'].diff()
    df['gain'] = df['delta'].clip(lower=0)
    df['loss'] = -df['delta'].clip(upper=0)
    df['avg_gain'] = df.groupby('symbol')['gain'].transform(lambda x: x.rolling(window=period).mean())
    df['avg_loss'] = df.groupby('symbol')['loss'].transform(lambda x: x.rolling(window=period).mean())
    rs = df['avg_gain'] / (df['avg_loss'] + 1e-8)
    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    df.drop(columns=['delta', 'gain', 'loss', 'avg_gain', 'avg_loss'], inplace=True)
    return df


def add_macd(df: pd.DataFrame, span_short: int, span_long: int) -> pd.DataFrame:
    """Add unique MACD columns to avoid overwriting existing EMAs."""
    df['macd_short'] = df.groupby('symbol')['adjClose'].transform(lambda x: x.ewm(span=span_short, adjust=False).mean())
    df['macd_long'] = df.groupby('symbol')['adjClose'].transform(lambda x: x.ewm(span=span_long, adjust=False).mean())
    df[f'macd_{span_short}_{span_long}'] = df['macd_short'] - df['macd_long']
    df.drop(columns=['macd_short', 'macd_long'], inplace=True)
    return df


def add_asi(df: pd.DataFrame) -> pd.DataFrame:
    """Add Accumulation Swing Index (ASI) using fast NumPy arrays to prevent KeyErrors."""
    def calculate_asi(group):
        high = group['adjHigh'].to_numpy()
        low = group['adjLow'].to_numpy()
        close = group['adjClose'].to_numpy()
        asi = [0.0]
        for i in range(1, len(group)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if tr == 0:
                asi.append(asi[-1])
            else:
                asi_value = asi[-1] + ((close[i] - close[i-1]) + 0.5 * (close[i] - low[i]) + 0.25 * (close[i-1] - high[i])) / tr * 100
                asi.append(asi_value)
        return pd.Series(asi, index=group.index)

    df['asi'] = df.groupby('symbol', group_keys=False).apply(calculate_asi)
    return df


def add_returns(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Add daily returns column to DataFrame."""
    df['lag'] = df.groupby('symbol')['adjClose'].shift(span)
    df[f'return_{span}d'] = (df['adjClose'] - df['lag']) / (df['lag'] + 1e-8)
    df.drop(columns=['lag'], inplace=True)
    return df


def add_atr(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Add Average True Range (ATR) column to DataFrame using original row-by-row style."""
    df['prev_Close'] = df.groupby('symbol')['adjClose'].shift(1)
    df['tr'] = df[['adjHigh', 'adjLow', 'adjClose', 'prev_Close']].apply(
        lambda x: max(
                x['adjHigh'] - x['adjLow'],
                abs(x['adjHigh'] - x['prev_Close']) if not pd.isna(x['prev_Close']) else x['adjHigh'] - x['adjLow'],
                abs(x['adjLow'] - x['prev_Close']) if not pd.isna(x['prev_Close']) else x['adjHigh'] - x['adjLow']
            ),
        axis=1
    )
    df[f'atr_{span}'] = df.groupby('symbol')['tr'].transform(lambda x: x.rolling(window=span).mean())
    df.drop(columns=['tr', 'prev_Close'], inplace=True)
    return df


def add_rank(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Calculate the cross-sectional relative strength label for each stock."""
    df['future_close'] = df.groupby('symbol')['adjClose'].shift(-span)
    df['forward_return'] = (df['future_close'] - df['adjClose']) / (df['adjClose'] + 1e-8)
    df['rank'] = df.groupby('date')['forward_return'].rank(pct=True)
    df.drop(columns=['future_close'], inplace=True)
    return df

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize specified columns in the DataFrame."""
    columns = [col for col in df.columns if col not in ['date', 'sector', 'adjOpen', 'adjHigh', 'adjLow', 'volume', 'adjClose', 'symbol', 'rank', 'forward_return']]
    for col in columns:
        df[col] = df.groupby('date')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        df[col] = df[col].clip(lower=-3.0, upper=3.0)

    df['sector'] = df['sector'].astype('category').cat.codes

    return df


if __name__ == "__main__":
    df = load_csv("data/raw/data.csv")

    add_ema(df, span=5)
    add_ema(df, span=20)
    add_ema(df, span=50)

    add_rsi(df, period=7)
    add_rsi(df, period=14)

    add_macd(df, span_short=5, span_long=10)
    add_macd(df, span_short=12, span_long=26)

    add_asi(df)
    # add_atr(df, span=14)

    add_returns(df, span=1)
    add_returns(df, span=5)
    add_returns(df, span=20)
    add_returns(df, span=60)

    add_rank(df, span=20)

    features = [col for col in df.columns if col not in ['date', 'sector', 'adjOpen', 'adjHigh', 'adjLow', 'volume', 'adjClose', 'symbol', 'rank', 'forward_return']]

    df.reset_index(drop=True, inplace=True)
    df.dropna(subset=features, inplace=True)
    save_csv(df, "data/processed/data.csv")

    normalize(df)
    df.drop(columns=['adjOpen', 'adjHigh', 'adjLow', 'volume', 'adjClose'], inplace=True)
    save_csv(df, "data/processed/normalized.csv")
