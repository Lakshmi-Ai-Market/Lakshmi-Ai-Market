import requests import numpy as np

➤ Fetch real candles from your internal candle API

def fetch_candles(): try: res = requests.get("http://localhost:5000/api/candle")  # or your deployed endpoint data = res.json() return data if isinstance(data, list) else [] except Exception as e: print("❌ Error fetching candles:", str(e)) return []

➤ EMA Crossover Strategy

def ema_crossover(candles): closes = [c['close'] for c in candles] if len(closes) < 21: return None ema9 = sum(closes[-9:]) / 9 ema21 = sum(closes[-21:]) / 21 if ema9 > ema21: return {"strategy": "📈 EMA Bullish Crossover", "confidence": 85} elif ema9 < ema21: return {"strategy": "📉 EMA Bearish Crossover", "confidence": 80}

➤ RSI Reversal

def rsi_reversal(candles): closes = [c['close'] for c in candles] if len(closes) < 15: return None gains, losses = [], [] for i in range(-14, 0): change = closes[i] - closes[i - 1] if change > 0: gains.append(change) else: losses.append(abs(change)) if not gains or not losses: return None avg_gain = sum(gains) / 14 avg_loss = sum(losses) / 14 rs = avg_gain / avg_loss if avg_loss != 0 else 0 rsi = 100 - (100 / (1 + rs)) if rsi < 30: return {"strategy": "🔁 RSI Bullish Reversal (<30)", "confidence": 80} elif rsi > 70: return {"strategy": "🔄 RSI Bearish Reversal (>70)", "confidence": 75}

➤ MACD Divergence

def macd_strategy(candles): closes = [c['close'] for c in candles] if len(closes) < 35: return None ema12 = sum(closes[-12:]) / 12 ema26 = sum(closes[-26:]) / 26 macd = ema12 - ema26 signal = sum(closes[-9:]) / 9 if macd > signal: return {"strategy": "🚀 MACD Bullish Momentum", "confidence": 82} elif macd < signal: return {"strategy": "🔻 MACD Bearish Momentum", "confidence": 77}

➤ Breakout Strategy

def breakout_strategy(candles): highs = [c['high'] for c in candles] closes = [c['close'] for c in candles] resistance = max(highs[-10:-1]) if closes[-1] > resistance: return {"strategy": "💥 Breakout Above Resistance", "confidence": 79}

➤ Pullback Strategy

def pullback_strategy(candles): closes = [c['close'] for c in candles] if len(closes) < 10: return None retracement = closes[-3] > closes[-2] < closes[-1] if retracement: return {"strategy": "🔂 Pullback Entry Signal", "confidence": 74}

➤ Volume Surge Strategy

def volume_surge(candles): vols = [c['volume'] for c in candles] if len(vols) < 10: return None avg_vol = sum(vols[-10:-1]) / 9 if vols[-1] > 1.5 * avg_vol: return {"strategy": "📊 Volume Surge Detected", "confidence": 76}

➤ SuperTrend (basic version)

def supertrend(candles): if len(candles) < 10: return None last = candles[-1] trend = "Bullish" if last['close'] > last['open'] else "Bearish" return { "strategy": f"🌐 SuperTrend Signal: {trend}", "confidence": 78 if trend == "Bullish" else 72 }

➤ Combine all strategies

def analyze_all(): candles = fetch_candles() if not candles: return {"error": "❌ Unable to fetch real candle data."}

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
    return {"message": "⚠️ No strong signals detected. Market unclear."}

bullish_count = sum(1 for s in valid if "Bullish" in s['strategy'] or "Breakout" in s['strategy'])
bearish_count = len(valid) - bullish_count
total = len(valid)
bias = "📈 Bullish" if bullish_count > bearish_count else "📉 Bearish"
confidence = round((max(bullish_count, bearish_count) / total) * 100)

return {
    "summary": f"Market Bias: {bias} ({confidence}% confidence)",
    "strategies": valid
}

