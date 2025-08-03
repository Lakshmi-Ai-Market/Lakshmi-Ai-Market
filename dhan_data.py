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
        url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        response = requests.get(url)
        response.raise_for_status()

        # Load CSV with proper dtype handling
        df = pd.read_csv(StringIO(response.text), low_memory=False)

        # Debug: Print column names
        print("📊 Columns found:", df.columns.tolist())

        # Normalize search
        index_name = index_name.upper().strip()
        match = df[df['SM_SYMBOL_NAME'].str.upper().str.contains(index_name)]

        if match.empty:
            print(f"❌ No match found for {index_name}")
            return None

        # Debug: Show top matches
        print(f"✅ Match found for {index_name}:\n", match[['SM_SYMBOL_NAME', 'SEM_SMST_SECURITY_ID']].head())

        # Return token ID
        return match.iloc[0]['SEM_SMST_SECURITY_ID']

    except Exception as e:
        print(f"❌ Error fetching instrument token: {e}")
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
