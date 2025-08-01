import requests
import re
import time
import os
from datetime import datetime, timedelta

# === Detect F&O symbol from free text ===
def extract_symbol_from_text(user_input):
    input_lower = user_input.lower()
    if re.search(r"\bbank\s?nifty|\bbanknifty|\bbank\b", input_lower):
        return "NSE_INDEX|BANKNIFTY"
    elif re.search(r"\bnifty\b", input_lower):
        return "NSE_INDEX|NIFTY_50"
    elif re.search(r"\bsensex\b|\bsen\b", input_lower):
        return "BSE_INDEX|SENSEX"
    return None

# === Fetch candles from Dhan API ===
def fetch_candles(symbol):
    try:
        token = os.getenv("DHAN_API_TOKEN")
        if not token:
            print("❌ Missing DHAN_API_TOKEN in environment.")
            return []

        exchange = symbol.split("|")[0]
        security = symbol.split("|")[1]

        url = "https://api.dhan.co/market/v1/candles"
        now = datetime.utcnow()
        past = now - timedelta(minutes=60)

        params = {
            "securityId": f"{exchange}|{security}",
            "exchangeSegment": exchange,
            "instrumentType": "INDEX",
            "fromDate": past.strftime("%Y-%m-%dT%H:%M:%S"),
            "toDate": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "interval": "5MIN"
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }

        res = requests.get(url, headers=headers, params=params)
        data = res.json()

        if "candles" not in data:
            print("❌ Dhan error:", data)
            return []

        candles = []
        for i in data["candles"]:
            candles.append({
                "time": int(datetime.strptime(i["startTime"], "%Y-%m-%dT%H:%M:%S").timestamp()),
                "open": i["open"],
                "high": i["high"],
                "low": i["low"],
                "close": i["close"],
                "volume": i["volume"]
            })

        return candles
    except Exception as e:
        print("❌ Error fetching candles:", str(e))
        return []

# === Strategies ===
def ema_crossover(c):
    closes = [x['close'] for x in c]
    if len(closes) < 21:
        return None
    ema9 = sum(closes[-9:]) / 9
    ema21 = sum(closes[-21:]) / 21
    if ema9 > ema21:
        return {"strategy": "📈 EMA Bullish Crossover", "confidence": 85}
    elif ema9 < ema21:
        return {"strategy": "📉 EMA Bearish Crossover", "confidence": 80}

def rsi_reversal(c):
    closes = [x['close'] for x in c]
    if len(closes) < 15: return None
    gains, losses = [], []
    for i in range(-14, 0):
        change = closes[i] - closes[i - 1]
        if change > 0: gains.append(change)
        else: losses.append(abs(change))
    if not gains or not losses: return None
    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    if rsi < 30: return {"strategy": "🔁 RSI Bullish Reversal (<30)", "confidence": 80}
    elif rsi > 70: return {"strategy": "🔄 RSI Bearish Reversal (>70)", "confidence": 75}

def macd_strategy(c):
    closes = [x['close'] for x in c]
    if len(closes) < 35: return None
    ema12 = sum(closes[-12:]) / 12
    ema26 = sum(closes[-26:]) / 26
    macd = ema12 - ema26
    signal = sum(closes[-9:]) / 9
    if macd > signal: return {"strategy": "🚀 MACD Bullish Momentum", "confidence": 82}
    elif macd < signal: return {"strategy": "🔻 MACD Bearish Momentum", "confidence": 77}

def breakout_strategy(c):
    highs = [x['high'] for x in c]
    closes = [x['close'] for x in c]
    if len(highs) < 10: return None
    resistance = max(highs[-10:-1])
    if closes[-1] > resistance:
        return {"strategy": "💥 Breakout Above Resistance", "confidence": 79}

def pullback_strategy(c):
    closes = [x['close'] for x in c]
    if len(closes) < 10: return None
    retracement = closes[-3] > closes[-2] < closes[-1]
    if retracement: return {"strategy": "🔂 Pullback Entry Signal", "confidence": 74}

def volume_surge(c):
    vols = [x['volume'] for x in c]
    if len(vols) < 10: return None
    avg_vol = sum(vols[-10:-1]) / 9
    if vols[-1] > 1.5 * avg_vol: return {"strategy": "📊 Volume Surge Detected", "confidence": 76}

def supertrend(c):
    if len(c) < 10: return None
    last = c[-1]
    trend = "Bullish" if last['close'] > last['open'] else "Bearish"
    return {"strategy": f"🌐 SuperTrend Signal: {trend}", "confidence": 78 if trend == "Bullish" else 72}

def bollinger_band_squeeze(c):
    closes = [x['close'] for x in c]
    if len(closes) < 20: return None
    sma = sum(closes[-20:]) / 20
    stddev = (sum([(x - sma) ** 2 for x in closes[-20:]]) / 20) ** 0.5
    upper = sma + 2 * stddev
    lower = sma - 2 * stddev
    if (upper - lower) / sma < 0.03:
        return {"strategy": "🧨 Bollinger Squeeze", "confidence": 78}

# === Candlestick Patterns ===

def marubozu_bullish(c): last = c[-1]; return {"strategy": "🚩 Bullish Marubozu", "confidence": 79} if last['open'] == min(last['open'], last['close']) and last['close'] == max(last['open'], last['close']) else None
def marubozu_bearish(c): last = c[-1]; return {"strategy": "🚩 Bearish Marubozu", "confidence": 78} if last['open'] == max(last['open'], last['close']) and last['close'] == min(last['open'], last['close']) else None
def bullish_engulfing(c): a,b=c[-2],c[-1]; return {"strategy": "🟩 Bullish Engulfing", "confidence": 80} if a['close']<a['open'] and b['close']>b['open'] and b['close']>a['open'] and b['open']<a['close'] else None
def bearish_engulfing(c): a,b=c[-2],c[-1]; return {"strategy": "🟥 Bearish Engulfing", "confidence": 80} if a['close']>a['open'] and b['close']<b['open'] and b['open']>a['close'] and b['close']<a['open'] else None
def morning_star(c): a,b,d=c[-3],c[-2],c[-1]; return {"strategy": "🌅 Morning Star", "confidence": 82} if a['close']<a['open'] and b['low']<a['close'] and d['close']>((a['open']+a['close'])/2) else None
def evening_star(c): a,b,d=c[-3],c[-2],c[-1]; return {"strategy": "🌇 Evening Star", "confidence": 82} if a['close']>a['open'] and b['high']>a['close'] and d['close']<((a['open']+a['close'])/2) else None
def hammer_candle(c): last=c[-1]; body=abs(last['close']-last['open']); wick=last['open']-last['low'] if last['open']>last['close'] else last['close']-last['low']; return {"strategy":"🔨 Hammer Pattern","confidence":76} if wick > 2 * body else None
def inverted_hammer(c): last=c[-1]; body=abs(last['close']-last['open']); wick=last['high']-max(last['open'],last['close']); return {"strategy":"🔎 Inverted Hammer","confidence":75} if wick > 2 * body else None
def spinning_top(c): last=c[-1]; body=abs(last['close']-last['open']); rng=last['high']-last['low']; return {"strategy":"🎯 Spinning Top","confidence":74} if rng>0 and body/rng<0.3 else None
def harami_bullish(c): a,b=c[-2],c[-1]; return {"strategy":"🈶 Bullish Harami","confidence":77} if a['open']>a['close'] and b['open']<b['close'] and b['open']>a['close'] and b['close']<a['open'] else None
def harami_bearish(c): a,b=c[-2],c[-1]; return {"strategy":"🈚 Bearish Harami","confidence":76} if a['open']<a['close'] and b['open']>b['close'] and b['open']<a['close'] and b['close']>a['open'] else None
def three_white_soldiers(c): a,b,d=c[-3],c[-2],c[-1]; return {"strategy":"🎌 Three White Soldiers","confidence":85} if all(x['close']>x['open'] for x in [a,b,d]) and a['close']<b['close']<d['close'] else None
def three_black_crows(c): a,b,d=c[-3],c[-2],c[-1]; return {"strategy":"☠️ Three Black Crows","confidence":85} if all(x['close']<x['open'] for x in [a,b,d]) and a['close']>b['close']>d['close'] else None
def piercing_pattern(c): a,b=c[-2],c[-1]; mid=(a['open']+a['close'])/2; return {"strategy":"⚡ Piercing Pattern","confidence":79} if a['close']<a['open'] and b['open']<b['close'] and b['close']>mid and b['open']<a['low'] else None
def dark_cloud_cover(c): a,b=c[-2],c[-1]; mid=(a['open']+a['close'])/2; return {"strategy":"🌩️ Dark Cloud Cover","confidence":79} if a['open']<a['close'] and b['open']>b['close'] and b['close']<mid and b['open']>a['high'] else None
def doji_star_bullish(c): a,b,d=c[-3],c[-2],c[-1]; return {"strategy":"✨ Bullish Doji Star","confidence":78} if a['close']<a['open'] and abs(b['open']-b['close'])<0.1 and d['close']>d['open'] else None
def doji_star_bearish(c): a,b,d=c[-3],c[-2],c[-1]; return {"strategy":"💫 Bearish Doji Star","confidence":78} if a['close']>a['open'] and abs(b['open']-b['close'])<0.1 and d['close']<d['open'] else None
def tweezers_bottom(c): a,b=c[-2],c[-1]; return {"strategy":"🍥 Tweezers Bottom","confidence":76} if a['low']==b['low'] and a['close']<a['open'] and b['close']>b['open'] else None
def tweezers_top(c): a,b=c[-2],c[-1]; return {"strategy":"🍡 Tweezers Top","confidence":76} if a['high']==b['high'] and a['close']>a['open'] and b['close']<b['open'] else None

# === Analyzer ===
def analyze_all_strategies(user_input):
    symbol = extract_symbol_from_text(user_input)
    if not symbol:
        return {"error": "❌ Could not detect a valid F&O index (e.g. NIFTY, BANKNIFTY, SENSEX)"}
    
    candles = fetch_candles(symbol)
    if not candles:
        return {"error": "❌ Unable to fetch real candle data."}

    strategies = [
        ema_crossover(candles), rsi_reversal(candles), macd_strategy(candles),
        breakout_strategy(candles), pullback_strategy(candles), volume_surge(candles),
        supertrend(candles), bollinger_band_squeeze(candles),
        marubozu_bullish(candles), marubozu_bearish(candles),
        bullish_engulfing(candles), bearish_engulfing(candles),
        morning_star(candles), evening_star(candles),
        hammer_candle(candles), inverted_hammer(candles), spinning_top(candles),
        harami_bullish(candles), harami_bearish(candles),
        three_white_soldiers(candles), three_black_crows(candles),
        piercing_pattern(candles), dark_cloud_cover(candles),
        doji_star_bullish(candles), doji_star_bearish(candles),
        tweezers_bottom(candles), tweezers_top(candles)
    ]

    valid = [s for s in strategies if s]
    if not valid:
        return {"message": "⚠️ No strong signals detected. Market unclear."}

    bullish = sum(1 for s in valid if "Bullish" in s['strategy'] or "Breakout" in s['strategy'])
    bearish = len(valid) - bullish
    bias = "📈 Bullish" if bullish > bearish else "📉 Bearish"
    confidence = round((max(bullish, bearish) / len(valid)) * 100)

    return {
        "summary": f"Market Bias: {bias} ({confidence}% confidence)",
        "strategies": valid,
        "symbol": symbol
    }
