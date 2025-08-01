import requests
import os

def fetch_dhan_price(symbol):
    instrument_map = {
        "NIFTY": "1330",
        "BANKNIFTY": "26009",
        "SENSEX": "256265"
    }
    instrument_id = instrument_map.get(symbol.upper())
    if not instrument_id:
        return None

    url = f"https://api.dhan.co/market/feed/iex/{instrument_id}"
    headers = {
        "accept": "application/json",
        "access-token": os.getenv("DHAN_ACCESS_TOKEN"),
        "client-id": os.getenv("DHAN_CLIENT_ID")
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        return float(data['last_traded_price']) if 'last_traded_price' in data else None
    except Exception as e:
        print("⚠️ Error fetching Dhan price:", e)
        return None
