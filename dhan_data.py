import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DHAN_BASE_URL = "https://api.dhan.co"
HEADERS = {
    "accept": "application/json",
    "access-token": os.getenv("DHAN_ACCESS_TOKEN"),
    "client-id": os.getenv("DHAN_CLIENT_ID")
}

# ✅ Get NSE_FNO Index Future Token
def get_fno_index_token(symbol):
    try:
        url = "https://images.dhan.co/api-data/FNO.csv"
        df = pd.read_csv(url)

        # Filter rows
        df = df[
            (df['instrument_type'] == 'FUTIDX') &
            (df['exchange_segment'] == 'NSE_FNO') &
            (df['trading_symbol'].str.contains(symbol.upper()))
        ]

        if df.empty:
            print(f"❌ No matching F&O Index found for: {symbol}")
            return None

        security_id = df.iloc[0]['security_id']
        print(f"✅ Found {symbol}: Security ID = {security_id}")
        return security_id

    except Exception as e:
        print("❌ Error fetching instrument from CSV:", str(e))
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
