import os
import requests
import pandas as pd
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

DHAN_BASE_URL = "https://api.dhan.co"
HEADERS = {
    "accept": "application/json",
    "access-token": os.getenv("DHAN_ACCESS_TOKEN"),
    "client-id": os.getenv("DHAN_CLIENT_ID")
}

def get_fno_index_token(index_name):
    try:
        # Load CSV from Dhan's URL
        url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        response = requests.get(url)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))

        # Check required columns
        required_cols = ['sm_symbol_name', 'sem_smst_security_id', 'sem_segment', 'sem_exch_instrument_type']
        if not all(col in df.columns for col in required_cols):
            print("❌ Required columns missing in CSV!")
            return None

        # Normalize column values
        df = df.dropna(subset=['sm_symbol_name'])  # Drop rows with NaN in symbol name
        df['sm_symbol_name'] = df['sm_symbol_name'].str.upper()
        df['sem_segment'] = df['sem_segment'].astype(str).str.upper()
        df['sem_exch_instrument_type'] = df['sem_exch_instrument_type'].astype(str).str.upper()

        # Clean input
        index_name = index_name.upper().strip()

        # Filter only for index derivatives (like BANKNIFTY, SENSEX)
        fno_df = df[
            (df['sem_segment'] == 'D') &  # Derivatives segment
            (df['sem_exch_instrument_type'] == 'IDX') &  # Index
            (df['sm_symbol_name'].str.contains(index_name, na=False))
        ]

        if fno_df.empty:
            print(f"❌ No token found for {index_name}")
            return None

        print(f"✅ Found token for {index_name}:")
        print(fno_df[['sm_symbol_name', 'sem_smst_security_id']].head())

        return fno_df.iloc[0]['sem_smst_security_id']

    except Exception as e:
        print(f"❌ Exception while fetching token: {e}")
        return None


# ✅ Candle fetch (mock or skip if not used now)
def fetch_candle_data(security_id, limit=30):
    try:
        url = f"{DHAN_BASE_URL}/charts/india/advanced-candle"
        params = {
            "securityId": security_id,
            "exchangeSegment": "NSE_FNO",
            "instrument": "FUTIDX",
            "interval": "5m",
            "limit": limit
        }
        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()
        candles = data.get("candles", [])
        if not candles:
            print(f"⚠️ No candles returned for: {security_id}")
        return candles
    except Exception as e:
        print("⚠️ Candle fetch error:", e)
        return []


# ✅ Basic strategy logic
def analyze_strategy(index_price):
    if index_price > 80000:
        return "BULLISH: Consider CALL strategy"
    elif index_price < 78000:
        return "BEARISH: Consider PUT strategy"
    else:
        return "NEUTRAL: Avoid trades"


# ✅ Fetch 5-minute Candles (last N)
def fetch_candle_data(security_id, limit=30):
    try:
        url = f"{DHAN_BASE_URL}/charts/india/advanced-candle"
        params = {
            "securityId": security_id,
            "exchangeSegment": "NSE_FNO",
            "instrument": "FUTIDX",
            "interval": "5m",
            "limit": limit
        }
        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()
        candles = data.get("candles", [])
        if not candles:
            print(f"⚠️ No candles returned for: {security_id}")
        return candles
    except Exception as e:
        print("⚠️ Candle fetch error:", e)
        return []
