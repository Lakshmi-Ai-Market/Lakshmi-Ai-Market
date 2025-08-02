import requests
import os

DHAN_BASE_URL = "https://api.dhan.co"

HEADERS = {
    "accept": "application/json",
    "access-token": os.getenv("DHAN_ACCESS_TOKEN"),
    "client-id": os.getenv("DHAN_CLIENT_ID")
}

def fetch_dhan_price(symbol):
    try:
        url = f"{DHAN_BASE_URL}/market/feed/iex/{symbol}"
        response = requests.get(url, headers=HEADERS)
        data = response.json()
        return float(data['last_traded_price']) if 'last_traded_price' in data else None
    except Exception as e:
        print("⚠️ LTP Error:", e)
        return None

def fetch_latest_data(symbol):
    token = os.getenv("DHAN_ACCESS_TOKEN")
    client_id = os.getenv("DHAN_CLIENT_ID")
    headers = {
        "access-token": token,
        "client-id": client_id
    }

    url = f"https://api.dhan.co/market/quotes/intraday/{symbol}"
    response = requests.get(url, headers=headers)
    data = response.json()
    
    try:
        return float(data["lastTradedPrice"])
    except:
        return 0.0

def fetch_candle_data(symbol):
    try:
        url = f"{DHAN_BASE_URL}/charts/india/advanced-candle"
        params = {
            "securityId": symbol,
            "exchangeSegment": "NSE_FNO",
            "instrument": "FUTIDX",
            "interval": "5m",
            "limit": 15
        }
        response = requests.get(url, headers=HEADERS, params=params)
        return response.json().get("candles", [])
    except Exception as e:
        print("⚠️ Candle Error:", e)
        return []
