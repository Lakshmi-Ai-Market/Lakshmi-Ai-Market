from dhan_data import fetch_candle_data

def fetch_dhan_candles(symbol, interval="5m", limit=15):
    candles = fetch_candle_data(symbol)
    if not candles:
        print(f"⚠️ No candle data found for symbol: {symbol}")
        return []
    return candles[-limit:]  # Last N candles

def strategy_rsi(symbol):
    candles = fetch_dhan_candles(symbol)
    if len(candles) < 14:
        return "Not enough data for RSI"

    closes = [float(c[4]) for c in candles]  # Close prices
    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
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

def strategy_ema_crossover(symbol):
    candles = fetch_dhan_candles(symbol)
    if len(candles) < 20:
        return "Not enough data for EMA"

    closes = [float(c[4]) for c in candles]
    
    def ema(data, period):
        k = 2 / (period + 1)
        ema_vals = [sum(data[:period]) / period]
        for price in data[period:]:
            ema_vals.append(price * k + ema_vals[-1] * (1 - k))
        return ema_vals

    short_ema = ema(closes, 9)
    long_ema = ema(closes, 21)

    if short_ema[-1] > long_ema[-1]:
        return "EMA crossover ➜ Bullish"
    else:
        return "EMA crossover ➜ Bearish"

def strategy_price_action(symbol):
    candles = fetch_dhan_candles(symbol)
    if len(candles) < 3:
        return "Not enough data for price action"

    last = candles[-1]
    prev = candles[-2]

    if float(last[4]) > float(prev[4]):
        return "Price Action ➜ Bullish Engulfing"
    elif float(last[4]) < float(prev[4]):
        return "Price Action ➜ Bearish Engulfing"
    else:
        return "Price Action ➜ Sideways"
