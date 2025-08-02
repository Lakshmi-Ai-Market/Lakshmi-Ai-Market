import datetime from dhan_data import fetch_dhan_price, fetch_candle_data

🔧 Correct function to fetch candles consistently

def fetch_dhan_candles(symbol, interval, limit): candles = fetch_candle_data(symbol) if not candles: print(f"⚠️ No candle data found for symbol: {symbol}") return [] return candles[-limit:]  # Take latest 'limit' candles

=== Strategy Implementations ===

def strategy_rsi(candles): if len(candles) < 14: return None

closes = [c[4] for c in candles]  # Close price
gains = [max(0, closes[i] - closes[i - 1]) for i in range(1, len(closes))]
losses = [max(0, closes[i - 1] - closes[i]) for i in range(1, len(closes))]

avg_gain = sum(gains[-14:]) / 14
avg_loss = sum(losses[-14:]) / 14
rs = avg_gain / avg_loss if avg_loss != 0 else 0
rsi = 100 - (100 / (1 + rs))

if rsi > 70:
    return {"strategy": "RSI", "bias": "Bearish", "confidence": round(rsi, 2)}
elif rsi < 30:
    return {"strategy": "RSI", "bias": "Bullish", "confidence": round(100 - rsi, 2)}
return None

def strategy_trend(candles): closes = [c[4] for c in candles] if closes[-1] > closes[0]: return {"strategy": "Trend", "bias": "Bullish", "confidence": 70} elif closes[-1] < closes[0]: return {"strategy": "Trend", "bias": "Bearish", "confidence": 70} return None

def strategy_candle_size(candles): bullish = sum(1 for c in candles if c[4] > c[1]) bearish = sum(1 for c in candles if c[4] < c[1]) total = bullish + bearish if total == 0: return None bias = "Bullish" if bullish > bearish else "Bearish" confidence = round((max(bullish, bearish) / total) * 100, 2) return {"strategy": "Candle Strength", "bias": bias, "confidence": confidence}

def strategy_trend_bias(symbol): data = fetch_dhan_candles(symbol, "5m", 20) if not data or len(data) < 5: return {"bias": "Neutral", "confidence": 0, "reason": "Insufficient candle data"}

closes = [c[4] for c in data]
average = sum(closes) / len(closes)
current_price = closes[-1]
previous_price = closes[-2]

if current_price > average and current_price > previous_price:
    return {"bias": "Bullish", "confidence": 75, "reason": "Price above average and rising"}
elif current_price < average and current_price < previous_price:
    return {"bias": "Bearish", "confidence": 75, "reason": "Price below average and falling"}
else:
    return {"bias": "Neutral", "confidence": 50, "reason": "Price around average"}

def strategy_breakout(symbol): candles = fetch_dhan_candles(symbol, "15m", 10) if not candles or len(candles) < 5: return {"bias": "Neutral", "confidence": 0, "reason": "Not enough 15m candles"}

highs = [c[2] for c in candles]  # high
lows = [c[3] for c in candles]   # low
current = candles[-1][4]         # close

resistance = max(highs[:-1])
support = min(lows[:-1])

if current > resistance:
    return {"bias": "Bullish", "confidence": 85, "reason": "Breakout above resistance"}
elif current < support:
    return {"bias": "Bearish", "confidence": 85, "reason": "Breakdown below support"}
else:
    return {"bias": "Neutral", "confidence": 45, "reason": "Still in range"}

def strategy_momentum(symbol): candles = fetch_dhan_candles(symbol, "1m", 15) if not candles or len(candles) < 10: return {"bias": "Neutral", "confidence": 0, "reason": "Insufficient 1m candles"}

closes = [c[4] for c in candles]
gains = sum([closes[i] - closes[i - 1] for i in range(1, len(closes)) if closes[i] > closes[i - 1]])
losses = sum([closes[i - 1] - closes[i] for i in range(1, len(closes)) if closes[i] < closes[i - 1]])

if gains > losses * 1.5:
    return {"bias": "Bullish", "confidence": 70, "reason": "Strong upward momentum"}
elif losses > gains * 1.5:
    return {"bias": "Bearish", "confidence": 70, "reason": "Strong downward momentum"}
else:
    return {"bias": "Neutral", "confidence": 50, "reason": "Weak or mixed momentum"}

def strategy_volume_spike(symbol): candles = fetch_dhan_candles(symbol, "5m", 10) if not candles or len(candles) < 5: return {"bias": "Neutral", "confidence": 0, "reason": "Volume data unavailable"}

volumes = [c[5] for c in candles]  # volume
avg_vol = sum(volumes[:-1]) / (len(volumes) - 1)
current_vol = volumes[-1]

if current_vol > 1.8 * avg_vol:
    price_movement = candles[-1][4] - candles[-1][1]  # close - open
    if price_movement > 0:
        return {"bias": "Bullish", "confidence": 80, "reason": "Volume spike on green candle"}
    elif price_movement < 0:
        return {"bias": "Bearish", "confidence": 80, "reason": "Volume spike on red candle"}

return {"bias": "Neutral", "confidence": 40, "reason": "No volume spike detected"}

def strategy_price_action(symbol): candles = fetch_dhan_candles(symbol, "5m", 7) if not candles or len(candles) < 5: return {"bias": "Neutral", "confidence": 0, "reason": "Not enough candle data"}

last = candles[-1]
prev = candles[-2]

if last[4] > last[1] and prev[4] < prev[1]:
    return {"bias": "Bullish", "confidence": 65, "reason": "Bullish engulfing"}
elif last[4] < last[1] and prev[4] > prev[1]:
    return {"bias": "Bearish", "confidence": 65, "reason": "Bearish engulfing"}
return {"bias": "Neutral", "confidence": 50, "reason": "No clear price pattern"}

=== Analyzer ===

def analyze_all_strategies(symbol="BANKNIFTY"): strategies = [ strategy_trend_bias, strategy_breakout, strategy_momentum, strategy_volume_spike, strategy_price_action ]

results = []
for strat in strategies:
    try:
        result = strat(symbol)
        results.append(result)
    except Exception as e:
        results.append({"bias": "Error", "confidence": 0, "reason": f"{strat.__name__} failed: {e}"})

bullish = sum(r["confidence"] for r in results if r["bias"] == "Bullish")
bearish = sum(r["confidence"] for r in results if r["bias"] == "Bearish")

if bullish > bearish:
    overall = "Bullish"
    confidence = round(bullish / (bullish + bearish + 1) * 100)
elif bearish > bullish:
    overall = "Bearish"
    confidence = round(bearish / (bullish + bearish + 1) * 100)
else:
    overall = "Neutral"
    confidence = 50

return {
    "symbol": symbol,
    "summary": overall,
    "confidence": confidence,
    "details": results,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

