import requests

def fetch_candles(symbol):
    try:
        res = requests.post("https://lakshmi-ai-trades.onrender.com/api/candle", json={
            "symbol": symbol.upper(),
            "timeframe": "5m"
        }, timeout=8)
        data = res.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        return []

def ema_crossover(candles):
    try:
        closes = [c['close'] for c in candles]
        if len(closes) < 21: return None
        ema9 = sum(closes[-9:]) / 9
        ema21 = sum(closes[-21:]) / 21
        if ema9 > ema21:
            return {"strategy": "📈 EMA Bullish Crossover", "confidence": 85}
        elif ema9 < ema21:
            return {"strategy": "📉 EMA Bearish Crossover", "confidence": 80}
    except: return None

def rsi_reversal(candles):
    try:
        closes = [c['close'] for c in candles]
        if len(closes) < 15: return None
        gains, losses = [], []
        for i in range(-14, 0):
            change = closes[i] - closes[i - 1]
            (gains if change > 0 else losses).append(abs(change))
        if not gains or not losses: return None
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        if rsi < 30:
            return {"strategy": "🔁 RSI Bullish Reversal (<30)", "confidence": 80}
        elif rsi > 70:
            return {"strategy": "🔄 RSI Bearish Reversal (>70)", "confidence": 75}
    except: return None

def macd_strategy(candles):
    try:
        closes = [c['close'] for c in candles]
        if len(closes) < 35: return None
        ema12 = sum(closes[-12:]) / 12
        ema26 = sum(closes[-26:]) / 26
        macd = ema12 - ema26
        signal = sum(closes[-9:]) / 9
        if macd > signal:
            return {"strategy": "🚀 MACD Bullish Momentum", "confidence": 82}
        elif macd < signal:
            return {"strategy": "🔻 MACD Bearish Momentum", "confidence": 77}
    except: return None

def breakout_strategy(candles):
    try:
        highs = [c['high'] for c in candles]
        closes = [c['close'] for c in candles]
        if len(highs) < 10: return None
        resistance = max(highs[-10:-1])
        if closes[-1] > resistance:
            return {"strategy": "💥 Breakout Above Resistance", "confidence": 79}
    except: return None

def pullback_strategy(candles):
    try:
        closes = [c['close'] for c in candles]
        if len(closes) < 10: return None
        if closes[-3] > closes[-2] < closes[-1]:
            return {"strategy": "🔂 Pullback Entry Signal", "confidence": 74}
    except: return None

def volume_surge(candles):
    try:
        vols = [c['volume'] for c in candles]
        if len(vols) < 10: return None
        avg_vol = sum(vols[-10:-1]) / 9
        if vols[-1] > 1.5 * avg_vol:
            return {"strategy": "📊 Volume Surge Detected", "confidence": 76}
    except: return None

def supertrend(candles):
    try:
        if len(candles) < 10: return None
        last = candles[-1]
        trend = "Bullish" if last['close'] > last['open'] else "Bearish"
        return {
            "strategy": f"🌐 SuperTrend Signal: {trend}",
            "confidence": 78 if trend == "Bullish" else 72
        }
    except: return None

def analyze_all(symbol="BANKNIFTY"):
    candles = fetch_candles(symbol)
    if not candles or not isinstance(candles, list):
        return {"symbol": symbol, "summary": f"❌ No candles found for {symbol.upper()}"}

    strategies = [
        ema_crossover(candles),
        rsi_reversal(candles),
        macd_strategy(candles),
        breakout_strategy(candles),
        pullback_strategy(candles),
        volume_surge(candles),
        supertrend(candles)
    ]
    valid = [s for s in strategies if s is not None]
    if not valid:
        return {"symbol": symbol, "summary": f"⚠️ No strategy signals found for {symbol.upper()}"}

    bullish = sum(1 for s in valid if "Bullish" in s['strategy'] or "Breakout" in s['strategy'])
    bearish = len(valid) - bullish
    bias = "📈 Bullish" if bullish > bearish else "📉 Bearish"
    confidence = round((max(bullish, bearish) / len(valid)) * 100)

    return {
        "symbol": symbol,
        "summary": f"{symbol.upper()} Bias: {bias} ({confidence}% confidence)",
        "strategies": valid
    }
