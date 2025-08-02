import os
import re
import requests
from datetime import datetime, timedelta
from dhan_data import get_fno_index_token, fetch_candle_data
from strategies import strategy_rsi, strategy_ema_crossover, strategy_price_action

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
    index_prices = extract_symbol_and_price(user_input)
    sensex_price = index_prices.get("Sensex", 0)
    banknifty_price = index_prices.get("Banknifty", 0)

    print(f"📊 Sensex: {sensex_price} BankNifty: {banknifty_price}")

    if not banknifty_price:
        return "❌ BankNifty price missing in input."

    symbol = generate_option_symbol(banknifty_price)
    print(f"✅ Generated symbol: {symbol}")

    token = get_fno_index_token(symbol)
    candles = fetch_candle_data(symbol, token)

    if not candles or len(candles) < 20:
        return "📉 Not enough data to run strategy."

    # Run strategy analysis
    result_rsi = strategy_rsi(candles)
    result_ema = strategy_ema_crossover(candles)
    result_price = strategy_price_action(candles)

    result = "📈 Strategy Results:\n"

    if result_rsi:
        result += f"🟢 RSI Strategy: {result_rsi}\n"
    if result_ema:
        result += f"🔵 EMA Crossover: {result_ema}\n"
    if result_price:
        result += f"🟣 Price Action: {result_price}\n"

    if result == "📈 Strategy Results:\n":
        return "😢 No strategy found"
    return result
