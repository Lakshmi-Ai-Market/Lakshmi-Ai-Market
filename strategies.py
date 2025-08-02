# strategies.py

from dhan_data import fetch_candle_data

def fetch_dhan_candles(symbol, interval="5m", limit=30):
    candles = fetch_candle_data(symbol)
    if not candles or len(candles) < limit:
        print(f"⚠️ Not enough candles ({len(candles) if candles else 0}) for symbol: {symbol}")
        return []
    print(f"✅ Fetched {len(candles)} candles for {symbol}")
    return candles[-limit:]  # Return only latest candles

# === Strategy 1: RSI Trend Analyzer ===
def strategy_rsi(symbol):
    candles = fetch_dhan_candles(symbol, limit=20)
    if len(candles) < 15:
        return "Not enough data for RSI"

    closes = [float(c[4]) for c in candles[-15:]]  # Last 15 closing prices
    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)

    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14 if sum(losses) != 0 else 0.0001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    trend = "Bullish" if rsi > 60 else "Bearish" if rsi < 40 else "Neutral"
    return f"RSI = {rsi:.2f} ➜ {trend}"

# === Strategy 2: EMA Crossover ===
def strategy_ema_crossover(symbol):
    candles = fetch_dhan_candles(symbol, limit=40)
    if len(candles) < 30:
        return "Not enough data for EMA"

    closes = [float(c[4]) for c in candles]

    def ema(data, period):
        if len(data) < period:
            return []
        k = 2 / (period + 1)
        ema_values = [sum(data[:period]) / period]
        for price in data[period:]:
            ema_values.append(price * k + ema_values[-1] * (1 - k))
        return ema_values

    short_ema = ema(closes, 9)
    long_ema = ema(closes, 21)

    if not short_ema or not long_ema or len(short_ema) < 1 or len(long_ema) < 1:
        return "EMA calculation error"

    if short_ema[-1] > long_ema[-1]:
        return "EMA crossover ➜ Bullish"
    elif short_ema[-1] < long_ema[-1]:
        return "EMA crossover ➜ Bearish"
    else:
        return "EMA crossover ➜ Sideways"

# === Strategy 3: Price Action (Engulfing Patterns) ===
def strategy_price_action(symbol):
    candles = fetch_dhan_candles(symbol, limit=5)
    if len(candles) < 3:
        return "Not enough data for price action"

    last = candles[-1]
    prev = candles[-2]

    last_close = float(last[4])
    prev_close = float(prev[4])
    last_open = float(last[1])
    prev_open = float(prev[1])

    if last_close > prev_open and last_open < prev_close:
        return "Price Action ➜ Bullish Engulfing"
    elif last_close < prev_open and last_open > prev_close:
        return "Price Action ➜ Bearish Engulfing"
    elif last_close > prev_close:
        return "Price Action ➜ Mild Bullish"
    elif last_close < prev_close:
        return "Price Action ➜ Mild Bearish"
    else:
        return "Price Action ➜ Sideways"
