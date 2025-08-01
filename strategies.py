herefrom dhan_data import fetch_dhan_price
import numpy as np
import pandas as pd

def ema_crossover(candles):
    df = pd.DataFrame(candles)
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

    if df['ema9'].iloc[-2] < df['ema21'].iloc[-2] and df['ema9'].iloc[-1] > df['ema21'].iloc[-1]:
        return {"strategy": "EMA Crossover - Bullish", "confidence": "High"}
    elif df['ema9'].iloc[-2] > df['ema21'].iloc[-2] and df['ema9'].iloc[-1] < df['ema21'].iloc[-1]:
        return {"strategy": "EMA Crossover - Bearish", "confidence": "High"}

def rsi_reversal(candles):
    df = pd.DataFrame(candles)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    rsi = df['rsi'].iloc[-1]

    if rsi < 30:
        return {"strategy": "RSI Oversold Reversal - Bullish", "confidence": "Moderate"}
    elif rsi > 70:
        return {"strategy": "RSI Overbought Reversal - Bearish", "confidence": "Moderate"}

def macd_strategy(candles):
    df = pd.DataFrame(candles)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    if df['macd'].iloc[-2] < df['signal'].iloc[-2] and df['macd'].iloc[-1] > df['signal'].iloc[-1]:
        return {"strategy": "MACD Bullish Crossover", "confidence": "Strong"}
    elif df['macd'].iloc[-2] > df['signal'].iloc[-2] and df['macd'].iloc[-1] < df['signal'].iloc[-1]:
        return {"strategy": "MACD Bearish Crossover", "confidence": "Strong"}

def breakout_strategy(candles):
    df = pd.DataFrame(candles)
    high_range = df['high'][-20:].max()
    low_range = df['low'][-20:].min()
    last_close = df['close'].iloc[-1]

    if last_close > high_range:
        return {"strategy": "Range Breakout - Bullish", "confidence": "High"}
    elif last_close < low_range:
        return {"strategy": "Range Breakdown - Bearish", "confidence": "High"}

def pullback_strategy(candles):
    df = pd.DataFrame(candles)
    df['ema'] = df['close'].ewm(span=20, adjust=False).mean()

    if df['close'].iloc[-1] > df['ema'].iloc[-1] and df['low'].iloc[-1] < df['ema'].iloc[-1]:
        return {"strategy": "Pullback on Uptrend - Buy Opportunity", "confidence": "Moderate"}
    elif df['close'].iloc[-1] < df['ema'].iloc[-1] and df['high'].iloc[-1] > df['ema'].iloc[-1]:
        return {"strategy": "Pullback on Downtrend - Sell Opportunity", "confidence": "Moderate"}

def volume_surge(candles):
    df = pd.DataFrame(candles)
    avg_vol = df['volume'][-20:].mean()
    last_vol = df['volume'].iloc[-1]

    if last_vol > 1.5 * avg_vol:
        return {"strategy": "Volume Surge Detected", "confidence": "Informative"}

def supertrend(candles, period=7, multiplier=3):
    df = pd.DataFrame(candles)
    hl2 = (df['high'] + df['low']) / 2
    atr = (df['high'] - df['low']).rolling(period).mean()
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    if df['close'].iloc[-1] > upper_band.iloc[-1]:
        return {"strategy": "Supertrend Buy Signal", "confidence": "High"}
    elif df['close'].iloc[-1] < lower_band.iloc[-1]:
        return {"strategy": "Supertrend Sell Signal", "confidence": "High"}

def bollinger_band_squeeze(candles):
    df = pd.DataFrame(candles)
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    width = upper - lower

    if width.iloc[-1] < 0.01 * df['close'].iloc[-1]:
        return {"strategy": "Bollinger Band Squeeze (Volatility Likely Ahead)", "confidence": "Alert"}

# More strategies can be added as needed
