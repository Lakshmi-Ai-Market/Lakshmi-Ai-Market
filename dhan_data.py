import requests
import os

DHAN_BASE_URL = "https://api.dhan.co"

HEADERS = {
    "accept": "application/json",
    "access-token": os.getenv("DHAN_ACCESS_TOKEN"),
    "client-id": os.getenv("DHAN_CLIENT_ID")
}

# Mapping common symbols to Dhan security IDs for F&O
INSTRUMENT_IDS = {
    "NIFTY": "1330",       # NIFTY 50 Index Futures
    "BANKNIFTY": "26009",  # BANKNIFTY Index Futures
    "SENSEX": "256265"     # SENSEX Index Futures (if supported)
}

# ✅ LTP Fetcher
def fetch_dhan_price(symbol):
    instrument_id = INSTRUMENT_IDS.get(symbol.upper())
    if not instrument_id:
        print(f"❌ Unknown symbol: {symbol}")
        return None

    url = f"{DHAN_BASE_URL}/market/feed/iex/{instrument_id}"
    try:
        response = requests.get(url, headers=HEADERS)
        data = response.json()
        return float(data['last_traded_price']) if 'last_traded_price' in data else None
    except Exception as e:
        print("⚠️ Error in fetch_dhan_price:", e)
        return None

# ✅ Candle Fetcher (last 15x 5-minute candles)
def fetch_candle_data(symbol):
    instrument_id = INSTRUMENT_IDS.get(symbol.upper())
    if not instrument_id:
        print(f"❌ Unknown symbol: {symbol}")
        return []

    url = f"{DHAN_BASE_URL}/charts/india/advanced-candle"
    params = {
        "securityId": instrument_id,
        "exchangeSegment": "NSE_FNO",
        "instrument": "FUTIDX",
        "interval": "5m",
        "limit": 15
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        candles = response.json().get("candles", [])
        return candles
    except Exception as e:
        print("⚠️ Error in fetch_candle_data:", e)
        return []
