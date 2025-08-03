import os
import requests
from datetime import datetime, timedelta
from dhan_data import get_fno_index_token, fetch_candle_data
from strategies import strategy_rsi, strategy_ema_crossover, strategy_price_action
import re

def extract_symbol_and_price(text):
    pattern = r"(NIFTY|BANKNIFTY|SENSEX|FINNIFTY)\s+([\d.]+)"
    matches = re.findall(pattern, text.upper())
    return {symbol: float(price) for symbol, price in matches}

# ✅ Correct usage of datetime
today = datetime.today()
expiry_date = today + timedelta(days=7)
expiry_str = expiry_date.strftime('%d%b').upper()  # e.g., '25AUG'

def fetch_dhan_price(symbol):
    url = f"https://api.dhan.co/market/feed/indices/{symbol}"
    headers = {
        "accept": "application/json",
        "access-token": os.getenv("DHAN_ACCESS_TOKEN"),
        "client-id": os.getenv("DHAN_CLIENT_ID")
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    return float(data.get("indexFeedDetails", {}).get("lastTradedPrice", 0))

def extract_symbol_and_price(user_input):
    matches = re.findall(r'(Sensex|BankNifty)\s*([\d.]+)', user_input, re.IGNORECASE)
    result = {}
    for match in matches:
        index = match[0].capitalize()
        price = float(match[1])
        result[index] = price
    return result

def generate_option_symbol(price):
    rounded_price = int(round(price / 100.0) * 100)
    return f"BANKNIFTY{expiry_str}{rounded_price}CE"

def analyze_all_strategies(user_input):
    try:
        # Extract symbols and prices from user input
        index_prices = extract_symbol_and_price(user_input)

        sensex_price = index_prices.get("SENSEX", 0)
        banknifty_price = index_prices.get("BANKNIFTY", 0)
        nifty_price = index_prices.get("NIFTY", 0)
        finnifty_price = index_prices.get("FINNIFTY", 0)

        response = ""

        # 🔍 Individual index analysis (basic logic from your above code)
        if sensex_price:
            result = analyze_sensex(sensex_price)
            response += f"📊 {result['symbol']} - {result['bias']} ({result['confidence']}%): {result['summary']}\n"
        if banknifty_price:
            result = analyze_banknifty(banknifty_price)
            response += f"📊 {result['symbol']} - {result['bias']} ({result['confidence']}%): {result['summary']}\n"
        if nifty_price:
            result = analyze_nifty(nifty_price)
            response += f"📊 {result['symbol']} - {result['bias']} ({result['confidence']}%): {result['summary']}\n"
        if finnifty_price:
            result = analyze_finnifty(finnifty_price)
            response += f"📊 {result['symbol']} - {result['bias']} ({result['confidence']}%): {result['summary']}\n"

        if banknifty_price:
            # 🔑 Symbol generation
            symbol = generate_option_symbol(banknifty_price)
            token = get_fno_index_token(symbol)

            if not token:
                return response + "\n⚠️ Token not found for symbol."

            candles = fetch_candle_data(symbol, token)

            if not candles or len(candles) < 20:
                return response + "\n📉 Not enough candle data for strategy engine."

            # 📈 Run strategies
            result_rsi = strategy_rsi(candles)
            result_ema = strategy_ema_crossover(candles)
            result_price = strategy_price_action(candles)

            response += "\n🧠 Strategy Engine Output:\n"
            if result_rsi:
                response += f"🟢 RSI Strategy: {result_rsi}\n"
            if result_ema:
                response += f"🔵 EMA Crossover: {result_ema}\n"
            if result_price:
                response += f"🟣 Price Action: {result_price}\n"

            if "🟢" not in response and "🔵" not in response and "🟣" not in response:
                response += "😢 No strategy triggered.\n"

        if response.strip() == "":
            return "❌ No valid indices found in input."
        return response

    except Exception as e:
        return f"❌ Error occurred: {str(e)}"
