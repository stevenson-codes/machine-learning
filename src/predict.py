import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from training import MarketLSTM

def build_lookback(df: pd.DataFrame, features: list, date: str, lookback: int = 60) -> list:
    samples = []
    
    print(f"[INFO] Building {lookback}-day lookback windows for all symbols...")
    for symbol, group in df.groupby('symbol'):
        snapshot = group[group['date'] <= date]
        window = snapshot[features].tail(lookback).values.astype(np.float32)
        samples.append({
            'symbol': symbol,
            'features': window
        })
            
    return samples

def apply_trend_filter(results: pd.DataFrame, df_raw: pd.DataFrame, date: str, sma_period: int = 50) -> pd.DataFrame:
    trend_mask = []
    
    for symbol in results['symbol']:
        group = df_raw[df_raw['symbol'] == symbol]
        group_snapshot = group[group['date'] <= date].sort_values('date')
        
        if len(group_snapshot) < sma_period:
            trend_mask.append(False)
            continue
        
        sma = group_snapshot['adjClose'].tail(sma_period).mean()
        current_price = group_snapshot['adjClose'].iloc[-1]
        
        if current_price >= sma:
            trend_mask.append(True)
        else:
            trend_mask.append(False)
    
    filtered = results[trend_mask].reset_index(drop=True)
    filtered_count = len(results) - len(filtered)
    print(f"[INFO] Trend filter removed {filtered_count} stocks below SMA-{sma_period}")
    
    return filtered

def predict(models_dir: str = 'models/best', top_n: int = 20) -> pd.DataFrame:
    df = pd.read_csv('data/processed/normalized.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['symbol', 'date'], inplace=True)
    
    df_raw = pd.read_csv('data/raw/data.csv')
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    
    features = [col for col in df.columns if col not in ['date', 'sector', 'symbol', 'forward_return', 'vol_return']]
    
    date = pd.to_datetime(df['date'].max()) - pd.Timedelta(days=20)
    print(f"[INFO] Making predictions for date: {date}")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    models = sorted(Path(models_dir).glob('*.pth'))
    
    samples = build_lookback(df, features, date, lookback=60)
    print(f"[INFO] Processing {len(samples)} symbols with all {len(models)} models")
    
    all_scores = []
    all_ranks = []
    
    for path in models:
        print(f"[INFO] Loading model: {path.name}")
        
        num_features = len(features)
        model = MarketLSTM(num_features=num_features, hidden_dim=32)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        
        features_tensor = torch.tensor(np.array([s['features'] for s in samples]), dtype=torch.float32).to(device)
        with torch.no_grad():preds = model(features_tensor).cpu().numpy().flatten()
        all_scores.append(preds)
        
        ranks = np.argsort(np.argsort(-preds))
        all_ranks.append(ranks)
    
    avg_ranks = np.mean(all_ranks, axis=0)
    results = pd.DataFrame({
        'symbol': [s['symbol'] for s in samples],
        'avg_rank': avg_ranks
    })
    
    for i, path in enumerate(models):
        results[f'model_{i}_score'] = all_scores[i]
        results[f'model_{i}_rank'] = all_ranks[i]
    
    results = apply_trend_filter(results, df_raw, date, sma_period=200)
    
    top_picks = results.sort_values('avg_rank').head(top_n).reset_index(drop=True)
    print("\n" + "="*60)
    print(f"TOP {top_n} PREDICTIONS (Rank Aggregation + Trend Filter)")
    print("="*60)
    print(top_picks[['symbol', 'avg_rank']].to_string(index=False))
    print("="*60)
    
    top_picks.to_csv('data/processed/picks.csv', index=False)
    print(f"\n[INFO] Predictions saved to data/processed/picks.csv")
    return top_picks

if __name__ == "__main__":
    predict()
