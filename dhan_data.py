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

# ✅ Path to Dhan's actual CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), 'api-scrip-master.csv')

# ✅ Load the CSV once
try:
    dhan_df = pd.read_csv(CSV_PATH)
    print("✅ Dhan CSV loaded.")
    print("📊 Available columns:", list(dhan_df.columns))
except Exception as e:
    print("❌ Failed to load CSV:", e)
    dhan_df = pd.DataFrame()

# ✅ Function to get token using 'sm_symbol_name'
def get_fno_index_token(symbol_name):
    if dhan_df.empty:
        return None

    try:
        row = dhan_df[
            (dhan_df['sem_exch_instrument_type'] == 'INDEX') &
            (dhan_df['sm_symbol_name'].str.upper().str.strip() == symbol_name.upper().strip())
        ].iloc[0]

        return str(row['sem_smst_security_id'])
    except Exception as e:
        print(f"❌ Error fetching token for {symbol_name}:", e)
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
