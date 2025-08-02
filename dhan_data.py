import os
import requests
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
        url = f"{DHAN_BASE_URL}/instruments/fno"
        response = requests.get(url, headers=HEADERS)
        
        print("🔁 Response status:", response.status_code)
        print("📦 Raw response:", response.text)

        instruments = response.json()

        if not isinstance(instruments, list):
            print("❌ Unexpected API format: instruments is not a list.")
            return None

        for item in instruments:
            if not isinstance(item, dict):
                continue
            if (
                symbol.upper() in item.get('trading_symbol', '') and
                item.get('instrument_type') == 'FUTIDX' and
                item.get('exchange_segment') == 'NSE_FNO'
            ):
                print(f"✅ Found: {item['trading_symbol']} -> {item['security_id']}")
                return item['security_id']

        print(f"❌ No matching F&O Index found for: {symbol}")
        return None

    except Exception as e:
        print("❌ Error fetching instrument:", str(e))
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
