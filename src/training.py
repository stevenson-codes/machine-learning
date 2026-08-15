import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MarketDataset(Dataset):
    """Custom Dataset for market data."""
    def __init__(self, df: pd.DataFrame, lookback: int = 60):
        self.data = df
        self.lookback = lookback

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        groups = df.groupby('symbol')

        features = [col for col in df.columns if col not in ['date', 'sector', 'symbol', 'forward_return', 'vol_return']]

        sequences = []
        targets = []
        for _, group in groups:
            feature_array = group[features].values.astype(np.float32)
            target_array = group['vol_return'].values.astype(np.float32)
            for i in range(len(group) - self.lookback):
                sequences.append(feature_array[i:i + self.lookback])
                targets.append(target_array[i + self.lookback])

        self.sequences = torch.from_numpy(np.array(sequences))
        self.targets = torch.from_numpy(np.array(targets))


    def __len__(self) -> int:
        return len(self.sequences)
    
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]
    
class MarketLSTM(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 64):
        super().__init__()
        
        # LSTM layer processes the sequence of continuous features
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )
        
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(data)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)
    
def train(filename: str, lookback: int = 60, model_size: int = 32) -> None:
    df = pd.read_csv(filename)
    
    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)
    train_end = int(0.8 * n_dates)
    val_end = int(n_dates)
    
    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end + 80:val_end])
    
    date_split = df['date'].map(lambda x: 'train' if x in train_dates else ('val' if x in val_dates else 'test'))
    train_df = df[date_split == 'train'].reset_index(drop=True)
    val_df = df[date_split == 'val'].reset_index(drop=True)
    # test_df = df[date_split == 'test'].reset_index(drop=True)
    
    train_dataset = MarketDataset(train_df, lookback=lookback)
    val_dataset = MarketDataset(val_df, lookback=lookback)
    # test_dataset = MarketDataset(test_df, lookback=lookback)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    num_features = train_dataset.sequences.shape[2]
    model = MarketLSTM(num_features=num_features, hidden_dim=model_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    print(f"[INFO] Starting training on device: {device.upper()}")

    best_return = -np.inf
    best_model = None
    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch).squeeze(-1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/20 - Loss: {avg_loss:.4f}")
        avg_return, _, sharpe_ratio = run_backtest(model, val_loader, val_df, device, top_n=10)
        if avg_return > best_return:
            best_return = avg_return
            torch.save(model.state_dict(), f"models/checkpoint.pth")

    print("\n[INFO] Evaluating on test set...")
    avg_return, std_return, sharpe_ratio = run_backtest(model, val_loader, val_df, device, top_n=10)
    torch.save(model.state_dict(), f"models/{avg_return:.4f}_{std_return:.4f}_{sharpe_ratio:.2f}.pth")
    model.load_state_dict(torch.load("models/checkpoint.pth"))
    avg_return, std_return, sharpe_ratio = run_backtest(model, val_loader, val_df, device, top_n=10)
    torch.save(model.state_dict(), f"models/{avg_return:.4f}_{std_return:.4f}_{sharpe_ratio:.2f}.pth")

def run_backtest(model, loader, df, device, top_n: int = 20):
    model.eval()
    all_preds = []
    
    print("[INFO] Generating predictions for backtest...")
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            
            preds = model(x_batch).squeeze(-1)
            all_preds.extend(preds.cpu().numpy().flatten())
            
    data = df.copy()
    data = data[data.groupby('symbol').cumcount() >= 60].reset_index(drop=True)
    data['predicted_rank'] = all_preds
    
    portfolio_returns = []
    
    print(f"[INFO] Simulating daily portfolio selection (Top {top_n} Stocks)...")
    for _, group in data.groupby('date'):
        group = group.sort_values('predicted_rank')
        long_portfolio = group.tail(top_n)
        
        avg_return = long_portfolio['forward_return'].mean()
        portfolio_returns.append(avg_return)
        
    avg_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns) + 1e-8

    max_drawdown = min(portfolio_returns)
    max_return = max(portfolio_returns)
    print(f"[DEBUG] Portfolio Return Stats - Max Drawdown: {max_drawdown:.4f}, Max Return: {max_return:.4f}")
    
    periods_per_year = 252 / 20
    risk_free_rate = 0.04 / periods_per_year  # 4% annual risk-free rate, converted to per-period
    sharpe_ratio = ((avg_return - risk_free_rate) / std_return) * np.sqrt(periods_per_year)
    
    print("\n" + "="*40)
    print(f"    LONG-ONLY BACKTEST RESULTS (TOP {top_n})   ")
    print("="*40)
    print(f"Average Return per Period: {avg_return * 100:.4f}%")
    print(f"Std Dev per Period:       {std_return * 100:.4f}%")
    print(f"Annualized Sharpe Ratio:  {sharpe_ratio:.2f}")
    print("="*40)
    
    return avg_return, std_return, sharpe_ratio

if __name__ == "__main__":
    for _ in range(30):
        train("data/processed/normalized.csv", lookback=60, model_size=32)
    # train("data/processed/normalized.csv", lookback=60, model_size=32)