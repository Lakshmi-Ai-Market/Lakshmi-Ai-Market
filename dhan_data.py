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

# Load CSV once during app start (ensure correct path)
FNO_CSV_PATH = "fno-indices.csv"
try:
    fno_data = pd.read_csv(FNO_CSV_PATH)
    print("✅ fno-indices.csv loaded with columns:", fno_data.columns.tolist())
except Exception as e:
    print(f"❌ Failed to load CSV: {e}")
    fno_data = pd.DataFrame()

def get_index_token(index_name):
    """
    Returns the security_id (token) for a given index (BANKNIFTY, SENSEX) from the CSV.
    """
    try:
        row = fno_data[fno_data['dhan_symbol'].str.upper() == index_name.upper()]
        if not row.empty:
            token = int(row.iloc[0]['security_id'])
            print(f"✅ Token for {index_name}: {token}")
            return token
        else:
            print(f"❌ {index_name} not found in dhan_symbol column")
            return None
    except Exception as e:
        print(f"❌ Error fetching token for {index_name}: {e}")
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
