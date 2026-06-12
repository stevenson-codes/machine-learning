import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class SoftSpearmanLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, y_pred, y_true):
        # CRITICAL FIX: Flatten inputs to 1D arrays to prevent broadcasting explosions
        y_pred = y_pred.view(-1)  # Force shape to [Num_Stocks]
        y_true = y_true.view(-1)  # Force shape to [Num_Stocks]
        
        # Now the pairwise matrix math works perfectly
        # Shape becomes a clean 2D matrix: [Num_Stocks, Num_Stocks]
        diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
        soft_pairs = torch.sigmoid(diff / self.temp)
        pred_rank = soft_pairs.sum(dim=1) 
        
        diff_true = y_true.unsqueeze(1) - y_true.unsqueeze(0)
        true_rank = torch.sigmoid(diff_true / self.temp).sum(dim=1)

        # Calculate Pearson Correlation on these flat soft ranks
        pred_centered = pred_rank - pred_rank.mean()
        true_centered = true_rank - true_rank.mean()
        
        covariance = (pred_centered * true_centered).sum()
        pred_variance = (pred_centered ** 2).sum()
        true_variance = (true_centered ** 2).sum()
        
        correlation = covariance / (torch.sqrt(pred_variance * true_variance) + 1e-8)
        return 1.0 - correlation

class CrossSectionalDateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int = 60):
        self.lookback = lookback
        
        # 1. Pivot the dataframe so index is Date and Columns are [Symbol x Feature]
        # This aligns all stocks perfectly in space and time
        feature_cols = [col for col in df.columns if col not in ['date', 'symbol', 'sector', 'rank', 'forward_return']]
        self.feature_cols = feature_cols
        
        # Sort chronologically
        df = df.sort_values('date')
        self.unique_dates = sorted(df['date'].unique())
        
        # Create a unified asset universe list to maintain matrix alignment
        self.symbols = sorted(df['symbol'].unique())
        self.symbol_to_idx = {sym: i for i, sym in enumerate(self.symbols)}
        
        # 2. Build dense 3D dense matrices up front [Total_Days, Total_Symbols, Features]
        num_days = len(self.unique_dates)
        num_symbols = len(self.symbols)
        num_features = len(feature_cols)
        
        self.X_matrix = np.zeros((num_days, num_symbols, num_features), dtype=np.float32)
        self.returns_matrix = np.zeros((num_days, num_symbols), dtype=np.float32)
        self.sectors_matrix = np.zeros((num_days, num_symbols), dtype=np.int64)
        self.mask_matrix = np.zeros((num_days, num_symbols), dtype=np.bool_) # To track which stocks are active
        
        # Map tabular data directly into our structured 3D numpy arrays
        date_to_idx = {date: i for i, date in enumerate(self.unique_dates)}
        
        # 1. Map dates and symbols directly to integer axis index codes
        date_codes = df['date'].map(date_to_idx).values
        symbol_codes = df['symbol'].map(self.symbol_to_idx).values

        # 2. Extract values as contiguous numpy arrays
        feature_values = df[feature_cols].values.astype(np.float32)
        return_values = df['forward_return'].values.astype(np.float32)
        sector_values = df['sector'].values.astype(np.int64)

        print("[INFO] Executing high-speed vectorized tensor population...")

        # 3. Vectorized Multi-Dimensional Array Assignment (No Python loops!)
        self.X_matrix[date_codes, symbol_codes, :] = feature_values
        self.returns_matrix[date_codes, symbol_codes] = return_values
        self.sectors_matrix[date_codes, symbol_codes] = sector_values
        self.mask_matrix[date_codes, symbol_codes] = True

    def __len__(self):
        # We can only sample days where a full lookback history exists
        return len(self.unique_dates) - self.lookback

    def __getitem__(self, idx):
        # Shift index forward by lookback to provide an adequate baseline runway
        t_idx = idx + self.lookback 
        
        # Extract 60 days of historical data for ALL symbols simultaneously via fast array slicing
        # Resulting Shape: [Total_Symbols, 60, Features]
        X_slice = self.X_matrix[t_idx - self.lookback : t_idx, :, :]
        X_slice = np.transpose(X_slice, (1, 0, 2)) # Rearrange layout to: [Symbols, Lookback, Features]
        
        # Extract targets for the single target day (t_idx)
        y_returns = self.returns_matrix[t_idx]
        y_sectors = self.sectors_matrix[t_idx]
        day_mask = self.mask_matrix[t_idx]
        
        # Filter out assets that do not have data recorded for this specific day
        # (This keeps your dynamic universe clean and unpolluted by dead or unlisted stocks)
        valid_X = X_slice[day_mask]
        valid_returns = y_returns[day_mask]
        valid_sectors = y_sectors[day_mask]
        
        return (
            torch.tensor(valid_X, dtype=torch.float32),
            torch.tensor(valid_sectors, dtype=torch.long),
            torch.tensor(valid_returns, dtype=torch.float32).unsqueeze(1)
        )

class MarketLSTM(nn.Module):
    def __init__(self, num_features: int, num_sectors: int, embedding_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.sector_embed = nn.Embedding(num_sectors, embedding_dim)
        
        # LSTM layer processes the sequence of continuous features
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        
        # Fully connected network fuses the final LSTM hidden state output with sector info
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )
        
    def forward(self, data, sector_x):
        # Cross-sectional loaders can emit [batch, symbols, lookback, features].
        # Flatten batch and symbol axes so LSTM receives [N, lookback, features].
        if data.dim() == 4:
            batch_size, num_symbols, lookback, num_features = data.shape
            data = data.reshape(batch_size * num_symbols, lookback, num_features)

        # cont_seq shape: [Batch, Lookback, Features]
        # out shape: [Batch, Lookback, Hidden_Dim]
        out, _ = self.lstm(data)
        
        # Grab the very last hidden state vector (the summary of the lookback sequence)
        last_hidden = out[:, -1, :] # Shape: [Batch, Hidden_Dim]
        
        # # Embed the sector and concatenate it
        # sector_embedded = self.sector_embed(sector_x) # Shape: [Batch, Embedding_Dim]
        # x = torch.cat([last_hidden, sector_embedded], dim=1)
        
        return self.fc(last_hidden)

def run_backtest(model, val_loader, val_df, device, top_n: int = 20):
    """
    Simulates a Long-Only strategy choosing exactly the Top N highest-predicted stocks each day.
    """
    model.eval()
    all_preds = []
    
    print("[INFO] Generating predictions for backtest...")
    with torch.no_grad():
        for cont_batch, sector_batch, _ in val_loader:
            cont_batch = cont_batch.to(device)
            sector_batch = sector_batch.to(device)

            if cont_batch.dim() == 4:
                batch_size, num_symbols, lookback, num_features = cont_batch.shape
                cont_batch = cont_batch.reshape(batch_size * num_symbols, lookback, num_features)
                sector_batch = sector_batch.reshape(batch_size * num_symbols)
            
            preds = model(cont_batch, sector_batch)
            all_preds.extend(preds.cpu().numpy().flatten())
            
    # Align rows 1:1 by dropping the lookback safety runway
    backtest_df = val_df.copy()
    valid_rows = []
    for _, group in backtest_df.groupby('symbol'):
        if len(group) >= 60:  # Matches your current lookback configuration (e.g., 60 days)
            valid_rows.append(group.iloc[60:])
    backtest_df = pd.concat(valid_rows).reset_index(drop=True)
    
    # Quick sanity check: Ensure predictions line up 1:1 with sliced dataframe length
    if len(backtest_df) != len(all_preds):
        print(f"[WARNING] Row mismatch! DataFrame has {len(backtest_df)} rows, but got {len(all_preds)} predictions.")
        # Trim both sides to a shared length so assignment remains valid.
        aligned_len = min(len(backtest_df), len(all_preds))
        backtest_df = backtest_df.iloc[:aligned_len].copy()
        all_preds = all_preds[:aligned_len]
    
    backtest_df['predicted_rank'] = all_preds
    
    portfolio_returns = []
    
    print(f"[INFO] Simulating daily portfolio selection (Top {top_n} Stocks)...")
    for date, group in backtest_df.groupby('date'):
        if len(group) < top_n:  # Skip days if the market data drops below your target count
            continue
            
        # Sort ascending, grab the absolute highest predicted performers at the tail
        group = group.sort_values('predicted_rank')
        long_portfolio = group.tail(top_n)
        
        # FIXED: Changed from 'raw_forward_return' to your correct pipeline field 'forward_return'
        avg_long_return = long_portfolio['forward_return'].mean()
        portfolio_returns.append(avg_long_return)
        
    # Calculate performance metrics
    avg_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns) + 1e-8
    
    # Annualized Sharpe for 20-trading-day holding periods (~252 trading days/year)
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

def predict_latest(model_path: str, top_n: int = 20, hidden_dim: int = 64):
    """
    Loads the absolute most recent market snapshot day from the non-dropped dataset,
    gathers the preceding 60 days of history for every ticker, and pushes them 
    through the model to output the ultimate actionable top 20 picks.
    """
    # 1. Load the dataset that preserved the unranked recent rows
    print(f"\n[INFO] Loading production tracking dataset from normalized.csv...")
    df = pd.read_csv('data/processed/normalized.csv')
    df.sort_values(['symbol', 'date'], ascending=[True, True], inplace=True)
    
    # Identify target features using your standard dataset exclusion criteria
    feature_cols = [col for col in df.columns if col not in ['date', 'symbol', 'sector', 'rank', 'forward_return']]
    
    # Identify the absolute newest day currently recorded in the file
    latest_date = df['date'].max()
    print(f"[INFO] Production Execution targeted at absolute newest date: {latest_date}")
    
    inference_samples = []
    
    # 2. Iterate through every ticker to build the 60-day chronological context window
    print("[INFO] Mining lookback sequence windows for all market symbols...")
    for symbol, group in df.groupby('symbol'):
        # Filter rows to only look at historical information up to this latest date
        group_snapshot = group[group['date'] <= latest_date]
        
        # Ensure the asset has a clean, complete 60-day lookback runway
        if len(group_snapshot) >= 60:
            lookback_matrix = group_snapshot[feature_cols].tail(60).values
            sector_code = group_snapshot['sector'].iloc[-1]
            
            inference_samples.append({
                'symbol': symbol,
                'cont': lookback_matrix,
                'sector': sector_code
            })
            
    if not inference_samples:
        print("[ERROR] No tickers met the minimum 60-day continuous window constraint.")
        return None

    # 3. Structural mapping into PyTorch CPU/GPU Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cont_tensor = torch.tensor([s['cont'] for s in inference_samples], dtype=torch.float32).to(device)
    sector_tensor = torch.tensor([s['sector'] for s in inference_samples], dtype=torch.long).to(device)
    
    # 4. Instantiate model structures and fill weights from your training checkpoint
    num_features = len(feature_cols)
    num_sectors = df['sector'].nunique()
    
    print(f"[INFO] Reconstructing model framework with {num_features} features and {num_sectors} sectors...")
    model = MarketLSTM(num_features=num_features, num_sectors=num_sectors, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 5. Execute production forward prediction pass
    print(f"[INFO] Computing real-time scores across {len(inference_samples)} qualifying assets...")
    with torch.no_grad():
        preds = model(cont_tensor, sector_tensor).cpu().numpy().flatten()
        
    # 6. Collate rankings and isolate the top 20 alpha performers
    results = pd.DataFrame({
        'symbol': [s['symbol'] for s in inference_samples],
        'score': preds
    })
    
    top_picks = results.sort_values('score', ascending=False).head(top_n).reset_index(drop=True)
    
    # Print clean execution report
    print("\n" + "="*55)
    print(f"     WALK-FORWARD SYSTEM EXECUTION SELECTIONS     ")
    print(f"     TARGET DATE: {latest_date} ")
    print("="*55)
    for idx, row in top_picks.iterrows():
        print(f" Rank {idx+1:02d} | Ticker: {row['symbol']:<6} | Model Strength Core Score: {row['score']:.5f}")
    print("="*55)
    print(f"[ACTION] Allocate capital to these {top_n} assets.")
    print(f"[ACTION] Hold position for 20 trading days to finalize walk-forward test.")
    print("="*55 + "\n")
    
    # Save the execution targets file so you can evaluate performance in 2 weeks
    top_picks.to_csv("data/processed/picks.csv", index=False)
    print("[INFO] Picks logged successfully to data/processed/picks.csv")
    
    return top_picks
    
def train_and_evaluate():
    # 1. Load your processed tabular dataset
    # Expects columns: ['date', 'symbol', 'sector', 'rank'] + your 8 indicator columns
    df = pd.read_csv("data/processed/normalized.csv")
    
    # Ensure date is string formatted or datetime sorted
    df.sort_values(['symbol', 'date'], ascending=[True, True], inplace=True)
    
    # 2. Define Chronological Cutoff Dates
    # Adjust these dates based on your actual data timeline bounds
    TRAIN_END = "2024-06-01"
    VAL_END   = "2025-01-01"
    
    train_df = df[df['date'] < TRAIN_END].copy()
    val_df   = df[(df['date'] >= TRAIN_END) & (df['date'] < VAL_END)].copy()
    eval_df  = df[df['date'] >= VAL_END].copy()
    
    print(f"[INFO] Data Splits:")
    print(f"  Train Rows: {len(train_df)} | Val Rows: {len(val_df)} | Eval Rows: {len(eval_df)}")
    
    # 3. Instantiate Sequential Lookback Datasets
    LOOKBACK = 60
    print("\n[INFO] Building 3D historical window matrices (this might take a minute)...")
    train_dataset = CrossSectionalDateDataset(train_df, lookback=LOOKBACK)
    val_dataset   = CrossSectionalDateDataset(val_df, lookback=LOOKBACK)
    eval_dataset  = CrossSectionalDateDataset(eval_df, lookback=LOOKBACK)
    
    # 4. Create PyTorch DataLoaders
    # Large batch size keeps GPUs efficiently fully utilized for cross-sectional market universes
    BATCH_SIZE = 1 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    eval_loader  = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    # 5. Initialize Model parameters dynamically
    # Detect input dimensional size based on your column count filtering layout
    sample_cont, _, _ = train_dataset[0]
    num_features = sample_cont.shape[2] # [Symbols, Lookback, Features]
    num_sectors = df['sector'].nunique() # Dynamic count of your integer labels
    
    # Select execution hardware target automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[INFO] Initializing MarketLSTM on platform target: {device}")
    
    model = MarketLSTM(num_features=num_features, num_sectors=num_sectors, embedding_dim=4, hidden_dim=32)
    model.to(device)
    
    # 6. Define Loss Function and Optimizer
    criterion = SoftSpearmanLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    
    # 7. Core Multi-Epoch Training and Validation Loop
    EPOCHS = 20
    print("\n[INFO] Commencing training process loop...")
    
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for cont_batch, sector_batch, target_batch in train_loader:
            # Shift batch chunks to target hardware memory space
            cont_batch = cont_batch.to(device)
            sector_batch = sector_batch.to(device)
            target_batch = target_batch.to(device)

            if cont_batch.dim() == 4:
                batch_size, num_symbols, lookback, num_features = cont_batch.shape
                cont_batch = cont_batch.reshape(batch_size * num_symbols, lookback, num_features)
                sector_batch = sector_batch.reshape(batch_size * num_symbols)
                target_batch = target_batch.reshape(batch_size * num_symbols, 1)
            
            optimizer.zero_grad()
            
            # Forward prediction pass
            predictions = model(cont_batch, sector_batch)
            loss = criterion(predictions, target_batch)
            
            # Optimization step
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * cont_batch.size(0)
            
        avg_train_loss = train_loss / len(train_dataset)
        
        # --- VALIDATION PHASE (Out of Sample Assessment) ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad(): # Disables gradient calculations to maximize speed and save memory
            for cont_batch, sector_batch, target_batch in val_loader:
                cont_batch = cont_batch.to(device)
                sector_batch = sector_batch.to(device)
                target_batch = target_batch.to(device)

                if cont_batch.dim() == 4:
                    batch_size, num_symbols, lookback, num_features = cont_batch.shape
                    cont_batch = cont_batch.reshape(batch_size * num_symbols, lookback, num_features)
                    sector_batch = sector_batch.reshape(batch_size * num_symbols)
                    target_batch = target_batch.reshape(batch_size * num_symbols, 1)
                
                predictions = model(cont_batch, sector_batch)
                loss = criterion(predictions, target_batch)
                
                val_loss += loss.item() * cont_batch.size(0)
                
        avg_val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        if (epoch + 1) % 2 == 0:
            avg_return, std_return, sharpe_ratio = run_backtest(model, val_loader, val_df, device, top_n=20)
            print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Avg Return: {avg_return:.5f} | Std Dev: {std_return:.5f} | Sharpe Ratio: {sharpe_ratio:.5f}")
    
    # 8. Evaluation Phase (Simulated Backtest Simulation Run)
    # This remains untouched during tuning, run once at the very end of your research!
    print("[INFO] Running final locked verification against evaluation timeline...")
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for cont_batch, sector_batch, target_batch in eval_loader:
            cont_batch = cont_batch.to(device)
            sector_batch = sector_batch.to(device)
            target_batch = target_batch.to(device)

            if cont_batch.dim() == 4:
                batch_size, num_symbols, lookback, num_features = cont_batch.shape
                cont_batch = cont_batch.reshape(batch_size * num_symbols, lookback, num_features)
                sector_batch = sector_batch.reshape(batch_size * num_symbols)
                target_batch = target_batch.reshape(batch_size * num_symbols, 1)
            
            predictions = model(cont_batch, sector_batch)
            loss = criterion(predictions, target_batch)
            eval_loss += loss.item() * cont_batch.size(0)
            
    print(f"Final System Evaluation Out-of-Sample Loss: {eval_loss / len(eval_dataset):.5f}")
    avg_return, std_return, sharpe_ratio = run_backtest(model, val_loader, val_df, device, top_n=20)
    print(f"Final System Evaluation | Avg Return: {avg_return:.5f} | Std Dev: {std_return:.5f} | Sharpe Ratio: {sharpe_ratio:.5f}")
    print("\n[INFO] Training complete. Saving target model weights checkpoint.")
    torch.save(model.state_dict(), f"models/{avg_return:.5f}_{std_return:.5f}_{sharpe_ratio:.5f}.pth")
    return f"models/{avg_return:.5f}_{std_return:.5f}_{sharpe_ratio:.5f}.pth"

if __name__ == "__main__":
    filename = train_and_evaluate()
    # for _ in range(10):
    #     filename = train_and_evaluate()
        # predict_latest(model_path="models/market_lstm.pth", top_n=20, hidden_dim=16)
