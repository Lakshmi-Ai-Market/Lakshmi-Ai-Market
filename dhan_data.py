import requests
import os

DHAN_BASE_URL = "https://api.dhan.co"

HEADERS = {
    "accept": "application/json",
    "access-token": os.getenv("DHAN_ACCESS_TOKEN"),
    "client-id": os.getenv("DHAN_CLIENT_ID")
}

# ✅ Live LTP fetcher
def fetch_dhan_price(symbol):
    url = f"{DHAN_BASE_URL}/market/feed/iex/{symbol}"
    try:
        response = requests.get(url, headers=HEADERS)
        data = response.json()
        return float(data['last_traded_price']) if 'last_traded_price' in data else None
    except Exception as e:
        print("⚠️ Error in fetch_dhan_price:", e)
        return None

# ✅ OHLC Candle Fetcher (last 15 candles, 5min)
def fetch_candle_data(symbol):
    url = f"{DHAN_BASE_URL}/charts/india/advanced-candle"
    params = {
        "securityId": symbol,
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
