# core_strategies.py

import pandas as pd

def strategy_rsi(candles):
    try:
        df = pd.DataFrame(candles)
        df['close'] = df['close'].astype(float)

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        latest_rsi = df['rsi'].iloc[-1]

        if latest_rsi < 30:
            signal = "Buy"
            confidence = 90
        elif latest_rsi > 70:
            signal = "Sell"
            confidence = 90
        else:
            signal = "Hold"
            confidence = 60

        return {
            "strategy": "RSI",
            "signal": signal,
            "confidence": confidence,
            "rsi_value": round(latest_rsi, 2)
        }

    except Exception as e:
        return {"strategy": "RSI", "signal": "Error", "confidence": 0, "error": str(e)}


def strategy_supertrend(candles, period=10, multiplier=3):
    try:
        df = pd.DataFrame(candles)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)

        df['ATR'] = (df['high'] - df['low']).rolling(window=period).mean()
        hl2 = (df['high'] + df['low']) / 2
        df['upper_band'] = hl2 + multiplier * df['ATR']
        df['lower_band'] = hl2 - multiplier * df['ATR']

        df['supertrend'] = df['close']
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['upper_band'].iloc[i - 1]:
                df.at[i, 'supertrend'] = df['lower_band'].iloc[i]
            elif df['close'].iloc[i] < df['lower_band'].iloc[i - 1]:
                df.at[i, 'supertrend'] = df['upper_band'].iloc[i]
            else:
                df.at[i, 'supertrend'] = df['supertrend'].iloc[i - 1]

        if df['close'].iloc[-1] > df['supertrend'].iloc[-1]:
            signal = "Buy"
            confidence = 85
        else:
            signal = "Sell"
            confidence = 85

        return {
            "strategy": "Supertrend",
            "signal": signal,
            "confidence": confidence,
            "supertrend_value": round(df['supertrend'].iloc[-1], 2)
        }

    except Exception as e:
        return {"strategy": "Supertrend", "signal": "Error", "confidence": 0, "error": str(e)}￼Enter
