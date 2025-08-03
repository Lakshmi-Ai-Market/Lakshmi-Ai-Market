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

        df = pd.read_csv(StringIO(response.text), low_memory=False)

        print("\U0001F4CA Columns found:", df.columns.tolist())

        index_name = index_name.upper().strip()
        df = df[df['SM_SYMBOL_NAME'].notnull()]
        df['SM_SYMBOL_NAME'] = df['SM_SYMBOL_NAME'].astype(str)

        # Exact match
        exact_match = df[df['SM_SYMBOL_NAME'].str.upper().str.strip() == index_name]
        if not exact_match.empty:
            print(f"✅ Exact match for {index_name}:", exact_match[['SM_SYMBOL_NAME', 'SEM_SMST_SECURITY_ID']].head())
            return exact_match.iloc[0]['SEM_SMST_SECURITY_ID']

        # Partial match
        contains_match = df[df['SM_SYMBOL_NAME'].str.upper().str.contains(index_name, na=False)]
        if not contains_match.empty:
            print(f"🔍 Partial match for {index_name}:", contains_match[['SM_SYMBOL_NAME', 'SEM_SMST_SECURITY_ID']].head())
            return contains_match.iloc[0]['SEM_SMST_SECURITY_ID']

        print(f"❌ No match found for {index_name}")
        return None

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

def analyze_nifty(ltp):
    if ltp < 24000:
        return {
            "symbol": "NIFTY",
            "bias": "Bullish",
            "confidence": 78.2,
            "summary": "NIFTY is near major support. Buying interest expected."
        }
    elif ltp > 24700:
        return {
            "symbol": "NIFTY",
            "bias": "Bearish",
            "confidence": 70.5,
            "summary": "NIFTY is in overbought zone. Caution advised."
        }
    else:
        return {
            "symbol": "NIFTY",
            "bias": "Neutral",
            "confidence": 50.0,
            "summary": "NIFTY is in range. No clear bias."
        }


def analyze_banknifty(ltp):
    if ltp < 55000:
        return {
            "symbol": "BANKNIFTY",
            "bias": "Bullish",
            "confidence": 81.1,
            "summary": "BANKNIFTY is close to support. Potential upward move."
        }
    elif ltp > 56500:
        return {
            "symbol": "BANKNIFTY",
            "bias": "Bearish",
            "confidence": 74.3,
            "summary": "BANKNIFTY is at resistance. Downside risk increasing."
        }
    else:
        return {
            "symbol": "BANKNIFTY",
            "bias": "Neutral",
            "confidence": 49.0,
            "summary": "BANKNIFTY showing no strong directional trend."
        }


def analyze_sensex(ltp):
    if ltp < 80000:
        return {
            "symbol": "SENSEX",
            "bias": "Bullish",
            "confidence": 75.6,
            "summary": "SENSEX below 80K is historically strong buying zone."
        }
    elif ltp > 81200:
        return {
            "symbol": "SENSEX",
            "bias": "Bearish",
            "confidence": 69.7,
            "summary": "SENSEX appears overextended. Pullback possible."
        }
    else:
        return {
            "symbol": "SENSEX",
            "bias": "Neutral",
            "confidence": 51.0,
            "summary": "SENSEX is consolidating in a narrow range."
        }


def analyze_finnifty(ltp):
    if ltp < 20500:
        return {
            "symbol": "FINNIFTY",
            "bias": "Bullish",
            "confidence": 77.9,
            "summary": "FINNIFTY approaching demand zone. Bullish bias."
        }
    elif ltp > 21000:
        return {
            "symbol": "FINNIFTY",
            "bias": "Bearish",
            "confidence": 73.4,
            "summary": "FINNIFTY hitting resistance. Possible decline."
        }
    else:
        return {
            "symbol": "FINNIFTY",
            "bias": "Neutral",
            "confidence": 48.8,
            "summary": "FINNIFTY in no-trade zone. Wait for breakout."
        }


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
