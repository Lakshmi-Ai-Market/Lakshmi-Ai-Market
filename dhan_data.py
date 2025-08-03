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

def get_fno_index_token(symbol):
    try:
        url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        response = requests.get(url)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), low_memory=False)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        print("📊 Available columns:", df.columns.tolist())  # Optional debug

        # Filter for FNO Index Instruments
        match = df[
            df['exchange_segment'].str.upper().str.contains("NSE_FNO", na=False) &
            df['trading_symbol'].str.upper().str.contains(symbol.upper(), na=False)
        ]

        if not match.empty:
            token = match.iloc[0]['scrip_code']  # ✅ use scrip_code instead
            print(f"✅ Found scrip_code for {symbol}: {token}")
            return token
        else:
            print(f"⚠️ No token found for {symbol}")
            return None

    except Exception as e:
        print(f"❌ Error fetching instrument from Dhan CSV: {e}")
        return None


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
