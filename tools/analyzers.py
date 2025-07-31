# tools/analyzers.py

def detect_ema_crossover(candles):
    if len(candles) < 20:
        return None

    closes = [c['close'] for c in candles]
    ema5 = sum(closes[-5:]) / 5
    ema20 = sum(closes[-20:]) / 20

    if ema5 > ema20:
        return {
            "strategy": "EMA Bullish Crossover Detected 💞",
            "confidence": 85,
            "entry": closes[-1],
            "sl": closes[-1] - 50,
            "target": closes[-1] + 120
        }

def detect_rsi_reversal(candles):
    if len(candles) < 15:
        return None

    closes = [c['close'] for c in candles]
    gains, losses = [], []

    for i in range(-14, 0):
        change = closes[i] - closes[i - 1]
        if change >= 0:
            gains.append(change)
        else:
            losses.append(abs(change))

    if not gains or not losses:
        return None

    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    if rsi < 30:
        return {
            "strategy": "RSI Reversal Detected 🔁",
            "confidence": 80,
            "entry": closes[-1],
            "sl": closes[-1] - 40,
            "target": closes[-1] + 100
        }

def detect_breakout(candles):
    if len(candles) < 10:
        return None

    closes = [c['close'] for c in candles]
    highs = [c['high'] for c in candles]
    resistance = max(highs[:-1])

    if closes[-1] > resistance:
        return {
            "strategy": "Breakout Zone Approaching 💥",
            "confidence": 75,
            "entry": closes[-1],
            "sl": closes[-1] - 60,
            "target": closes[-1] + 90
        }
