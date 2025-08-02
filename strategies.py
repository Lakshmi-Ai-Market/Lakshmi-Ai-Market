# strategies.py

from dhan_data import fetch_dhan_price, fetch_candle_data
import numpy as np

# === Candle Fetch ===
def fetch_dhan_candles(symbol, interval="1m", limit=20):
    candles = fetch_candle_data(symbol)
    if not candles:
        print(f"⚠️ No candle data found for symbol: {symbol}")
        return []
    return candles[-limit:]  # Latest 'limit' candles only

# === Strategy 1: EMA Crossover ===
def strategy_ema_crossover(symbol):
    candles = fetch_dhan_candles(symbol, limit=50)
    if len(candles) < 20:
        return ""

    closes = [float(c['close']) for c in candles]
    ema_9 = np.mean(closes[-9:])
    ema_21 = np.mean(closes[-21:])

    if ema_9 > ema_21:
        return f"📊 EMA Crossover: Strong Bullish on {symbol}"
    elif ema_9 < ema_21:
        return f"📉 EMA Crossover: Bearish on {symbol}"
    return ""

# === Strategy 2: RSI ===
def strategy_rsi(symbol):
    candles = fetch_dhan_candles(symbol, limit=20)
    if len(candles) < 15:
        return ""

    closes = [float(c['close']) for c in candles]
    deltas = np.diff(closes)
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    if losses == 0:
        rsi = 100
    else:
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))

    if rsi > 70:
        return f"⚠️ RSI: Overbought ({rsi:.1f}) on {symbol}"
    elif rsi < 30:
        return f"✅ RSI: Oversold ({rsi:.1f}) on {symbol}"
    return ""

# === Strategy 3: Price Action (Dummy) ===
def strategy_price_action(symbol):
    price = fetch_dhan_price(symbol)
    if not price:
        return ""
    return f"💰 Price Action: Current price of {symbol} is {price}"
