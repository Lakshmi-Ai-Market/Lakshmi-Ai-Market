from dhan_data import fetch_candle_data, get_fno_index_token

# === Core Fetch for Strategies ===
def fetch_dhan_candles(symbol, limit=40):
    token = get_fno_index_token(symbol.upper())
    if not token:
        print(f"❌ Could not fetch token for symbol: {symbol}")
        return []
    candles = fetch_candle_data(token, limit)
    if not candles or len(candles) < limit:
        print(f"⚠️ Not enough candles ({len(candles) if candles else 0}) for {symbol}")
        return []
    return candles[-limit:]

# === Strategy 1: RSI Trend Analyzer ===
def strategy_rsi(symbol):
    candles = fetch_dhan_candles(symbol, limit=20)
    if len(candles) < 15:
        return "RSI ➜ Not enough data"
    closes = [float(c[4]) for c in candles[-15:]]

    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(0, delta))
        losses.append(max(0, -delta))

    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14 if sum(losses) != 0 else 0.0001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    if rsi > 60:
        trend = "Bullish"
    elif rsi < 40:
        trend = "Bearish"
    else:
        trend = "Neutral"
    return f"RSI = {rsi:.2f} ➜ {trend}"

# === Strategy 2: EMA Crossover ===
def strategy_ema_crossover(symbol):
    candles = fetch_dhan_candles(symbol, limit=40)
    if len(candles) < 30:
        return "EMA ➜ Not enough data"
    closes = [float(c[4]) for c in candles]

    def ema(data, period):
        k = 2 / (period + 1)
        ema_values = [sum(data[:period]) / period]
        for price in data[period:]:
            ema_values.append(price * k + ema_values[-1] * (1 - k))
        return ema_values

    short_ema = ema(closes, 9)
    long_ema = ema(closes, 21)

    if short_ema[-1] > long_ema[-1]:
        return "EMA ➜ Bullish Crossover"
    elif short_ema[-1] < long_ema[-1]:
        return "EMA ➜ Bearish Crossover"
    else:
        return "EMA ➜ Sideways"

# === Strategy 3: Price Action (Engulfing + Direction) ===
def strategy_price_action(symbol):
    candles = fetch_dhan_candles(symbol, limit=5)
    if len(candles) < 3:
        return "Price Action ➜ Not enough data"

    last = candles[-1]
    prev = candles[-2]
    last_open = float(last[1])
    last_close = float(last[4])
    prev_open = float(prev[1])
    prev_close = float(prev[4])

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

# === Trend Summary Combiner ===
def combined_trend(symbol):
    rsi_result = strategy_rsi(symbol)
    ema_result = strategy_ema_crossover(symbol)
    price_action_result = strategy_price_action(symbol)

    signals = f"{rsi_result}\n{ema_result}\n{price_action_result}"

    if "Bullish" in signals and "Bearish" not in signals:
        overall = "📈 Strong Bullish Bias"
    elif "Bearish" in signals and "Bullish" not in signals:
        overall = "📉 Strong Bearish Bias"
    elif "Neutral" in signals or "Sideways" in signals:
        overall = "🔍 Sideways / Neutral"
    else:
        overall = "⚖️ Mixed Signals"

    return f"✅ Trend Summary: {overall}\n\n{signals}"
