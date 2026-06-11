import os
import re
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://financialmodelingprep.com/stable/"
API_KEY = os.getenv("API_KEY", "")
REQUEST_TIMEOUT_SECONDS = 30


def get_universe() -> list[str]:
    """Fetch 3500 largest cap stocks from FMP stock screener."""
    limit = 3500

    response = requests.get(
        API_URL + "company-screener",
        params={
            "limit": limit,
            "isEtf": "false",
            "isFund": "false",
            "isActivelyTrading": "true",
            "apikey": API_KEY,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    print(f"[DEBUG] Status code: {response.status_code}")
    data = response.json()
    print(f"[DEBUG] Response type: {type(data)}, length: {len(data) if isinstance(data, list) else 'N/A'}")

    if response.status_code != 200:
        return []
    
    reg = re.compile(r"^[A-Z]+$|^[A-Z]+-[A-Z]$")
    symbols = {x.get("symbol"): x.get("sector") for x in data if x.get("symbol") and reg.match(x.get("symbol"))}
    print(f"[DEBUG] Extracted {len(symbols)} symbols matching pattern.")
    return symbols


def load_symbol(symbol: str) -> Any:
    """Fetch and return JSON data for a symbol."""
    print(f"\n[DEBUG] Fetching {symbol}...")
    print(f"[DEBUG] API_KEY set: {bool(API_KEY)}")
    response = requests.get(
        API_URL + "historical-price-eod/dividend-adjusted",
        params={"symbol": symbol, "apikey": API_KEY},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    print(f"[DEBUG] Status code: {response.status_code}")
    data = response.json()
    print(f"[DEBUG] Response type: {type(data)}, length: {len(data) if isinstance(data, list) else 'N/A'}")

    if response.status_code != 200:
        return []
    return data


def load_symbols(
    symbols: dict[str, str],
) -> pd.DataFrame:
    """Load symbols one by one and return as a DataFrame."""
    all_records: list[dict[str, Any]] = []
    for symbol, sector in symbols.items():
        records = load_symbol(symbol)
        if len(records) < 1254:
            print(f"[DEBUG] Not enough data for {symbol}, skipping.")
            continue
        for record in records:
            record["sector"] = sector
        all_records.extend(records)
    print(f"[DEBUG] Total records loaded: {len(all_records)}")

    return pd.DataFrame(all_records)


def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to a CSV file."""
    df.to_csv(filename, index=False)


def save_universe(symbols: dict[str, str], filename: str) -> None:
    """Save universe symbols to a file."""
    df = pd.DataFrame({"symbol": list(symbols.keys()), "sector": list(symbols.values())})
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    universe = get_universe()
    save_universe(universe, "data/raw/universe.csv")
    df = load_symbols(universe)
    print(df.head())
    save_to_csv(df, "data/raw/data.csv")