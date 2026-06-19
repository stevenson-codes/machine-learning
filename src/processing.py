import numpy as np
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
    df[f'ema_{span}'] = df.groupby('symbol')[f'ema_{span}'].pct_change()

    df['rolling_mean'] = df.groupby('symbol')[f'ema_{span}'].transform(lambda x: x.rolling(window=252).mean())
    df['rolling_std'] = df.groupby('symbol')[f'ema_{span}'].transform(lambda x: x.rolling(window=252).std())
    df[f'ema_{span}'] = (df[f'ema_{span}'] - df['rolling_mean']) / (df['rolling_std'] + 1e-8)
    df.drop(columns=['rolling_mean', 'rolling_std'], inplace=True)

    return df


def add_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add Relative Strength Index (RSI) column to DataFrame."""
    df['delta'] = df.groupby('symbol')['adjClose'].diff()
    df['gain'] = df['delta'].clip(lower=0)
    df['loss'] = -df['delta'].clip(upper=0)
    df['avg_gain'] = df.groupby('symbol')['gain'].transform(lambda x: x.rolling(window=period).mean())
    df['avg_loss'] = df.groupby('symbol')['loss'].transform(lambda x: x.rolling(window=period).mean())
    rs = df['avg_gain'] / (df['avg_loss'] + 1e-8)
    df[f'rsi_{period}'] = (100 - (100 / (1 + rs))) / 100
    
    df.drop(columns=['delta', 'gain', 'loss', 'avg_gain', 'avg_loss'], inplace=True)
    return df


def add_macd(df: pd.DataFrame, span_short: int, span_long: int) -> pd.DataFrame:
    """Add unique MACD columns to avoid overwriting existing EMAs."""
    macd = f'macd_{span_short}_{span_long}'
    df['macd_short'] = df.groupby('symbol')['adjClose'].transform(lambda x: x.ewm(span=span_short, adjust=False).mean())
    df['macd_long'] = df.groupby('symbol')['adjClose'].transform(lambda x: x.ewm(span=span_long, adjust=False).mean())
    df[macd] = df['macd_short'] - df['macd_long']
    df[macd] = df.groupby('symbol')[macd].pct_change()
    df.drop(columns=['macd_short', 'macd_long'], inplace=True)

    df['rolling_mean'] = df.groupby('symbol')[macd].transform(lambda x: x.rolling(window=252).mean())
    df['rolling_std'] = df.groupby('symbol')[macd].transform(lambda x: x.rolling(window=252).std())
    df[macd] = (df[macd] - df['rolling_mean']) / (df['rolling_std'] + 1e-8)
    df.drop(columns=['rolling_mean', 'rolling_std'], inplace=True)
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
        return pd.Series(asi, index=group.index) / 100

    df['asi'] = df.groupby('symbol', group_keys=False).apply(calculate_asi)
    df['asi'] = df.groupby('symbol')['asi'].transform(lambda x: x.diff())

    df['rolling_mean'] = df.groupby('symbol')['asi'].transform(lambda x: x.rolling(window=252).mean())
    df['rolling_std'] = df.groupby('symbol')['asi'].transform(lambda x: x.rolling(window=252).std())
    df['asi'] = (df['asi'] - df['rolling_mean']) / (df['rolling_std'] + 1e-8)
    df.drop(columns=['rolling_mean', 'rolling_std'], inplace=True)
    return df


def add_returns(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Add daily returns column to DataFrame."""
    df['lag'] = df.groupby('symbol')['adjClose'].shift(span)
    df[f'return_{span}d'] = (df['adjClose'] - df['lag']) / (df['lag'] + 1e-8)
    df.drop(columns=['lag'], inplace=True)

    df['rolling_mean'] = df.groupby('symbol')[f'return_{span}d'].transform(lambda x: x.rolling(window=252).mean())
    df['rolling_std'] = df.groupby('symbol')[f'return_{span}d'].transform(lambda x: x.rolling(window=252).std())
    df[f'return_{span}d'] = (df[f'return_{span}d'] - df['rolling_mean']) / (df['rolling_std'] + 1e-8)
    df.drop(columns=['rolling_mean', 'rolling_std'], inplace=True)
    return df

def add_atr(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Add Average True Range (ATR) column to DataFrame."""
    df['high_low'] = df['adjHigh'] - df['adjLow']
    df['high_close'] = abs(df['adjHigh'] - df.groupby('symbol')['adjClose'].shift())
    df['low_close'] = abs(df['adjLow'] - df.groupby('symbol')['adjClose'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df[f'atr_{span}d'] = df.groupby('symbol')['true_range'].transform(lambda x: x.rolling(window=span).mean())
    df.drop(columns=['high_low', 'high_close', 'low_close', 'true_range'], inplace=True)

    df['rolling_mean'] = df.groupby('symbol')[f'atr_{span}d'].transform(lambda x: x.rolling(window=252).mean())
    df['rolling_std'] = df.groupby('symbol')[f'atr_{span}d'].transform(lambda x: x.rolling(window=252).std())
    df[f'atr_{span}d'] = (df[f'atr_{span}d'] - df['rolling_mean']) / (df['rolling_std'] + 1e-8)
    df.drop(columns=['rolling_mean', 'rolling_std'], inplace=True)
    return df

def add_cmf(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Add Chaikin Money Flow (CMF) column to DataFrame."""
    df['mult'] = (2 * df['adjClose'] - df['adjLow'] - df['adjHigh']) / (df['adjHigh'] - df['adjLow'] + 1e-8)
    df['money_volume'] = df['mult'] * df['volume']

    rolling_money = df.groupby("symbol")["money_volume"].transform(lambda x: x.rolling(window=span).sum())
    rolling_vol = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(window=span).sum())

    df[f"cmf_{span}d"] = rolling_money / rolling_vol
    df.drop(columns=['mult', 'money_volume'], inplace=True)
    return df

def add_target(df: pd.DataFrame, span: int, gamma: float = 3, ) -> pd.DataFrame:
    """Calculate the cross-sectional relative strength label for each stock."""
    df['future_close'] = df.groupby('symbol')['adjClose'].shift(-span)
    df['forward_return'] = (df['future_close'] - df['adjClose']) / (df['adjClose'] + 1e-8)

    df['return'] = df.groupby('symbol')['adjClose'].pct_change()
    df['n_return'] = np.minimum(0, df['return'])

    df['forward_vol'] = df.groupby('symbol')['n_return'].transform(lambda x: x.rolling(window=span).std().shift(-span))

    df['vol_return'] = df['forward_return'] - (gamma * (df['forward_vol'] ** 2))
    df.drop(columns=['future_close', 'return', 'n_return', 'forward_vol'], inplace=True)
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
    add_atr(df, span=14)
    add_cmf(df, span=20)

    add_returns(df, span=1)
    add_returns(df, span=5)
    add_returns(df, span=20)
    add_returns(df, span=60)

    add_target(df, span=20)

    features = [col for col in df.columns if col not in ['date', 'sector', 'adjOpen', 'adjHigh', 'adjLow', 'volume', 'adjClose', 'symbol', 'forward_return', 'vol_return']]

    df.reset_index(drop=True, inplace=True)
    df.dropna(subset=features, inplace=True)
    save_csv(df, "data/processed/data.csv")

    df.drop(columns=['adjOpen', 'adjHigh', 'adjLow', 'volume', 'adjClose'], inplace=True)
    save_csv(df, "data/processed/normalized.csv")
