import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Any, Optional
import logging
import os
import math
from dataclasses import dataclass
from datetime import datetime


# Import indicators correctly from utils package
from utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

@dataclass
class StrategySignal:
    signal: str                    # 'buy'|'sell'|'neutral'
    confidence: int                # 0-100
    reason: str
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None

class StrategyEngine:
    """
    Fully merged StrategyEngine with many real strategies.
    Use:
        engine = StrategyEngine()
        result = engine.run_analysis(market_data, strategy_type='all')
        ai = engine.ai_interpretation(symbol, result, indicators_dict)
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()
        # strategy categories -> list of method suffix names (methods below implemented with _prefix)
        self.strategy_categories = {
            'momentum': [
                'rsi_momentum','macd_crossover','stochastic_oscillator','williams_r',
                'cci','connors_rsi','coppock_curve','trix_indicator','ultimate_oscillator'
            ],
            'trend': [
                'ma_crossover','ema_crossover','adx_trend','ichimoku_cloud','aroon_indicator',
                'supertrend','keltner_channels','atr_trailing_stop','hull_ma'
            ],
            'volume': [
                'vwap','anchored_vwap','on_balance_volume','chaikin_money_flow',
                'chaikin_oscillator','volume_spike','vroc'
            ],
            'volatility': [
                'bollinger_bands','keltner_breakout','atr_breakout','mass_index','bollinger_squeeze'
            ],
            'price_action': [
                'bullish_engulfing','bearish_engulfing','hammer','shooting_star',
                'doji','morning_star','evening_star','piercing','dark_cloud_cover',
                'bullish_harami','bearish_harami'
            ],
            'swing': [
                'pivot_points','fibonacci_retracement','elder_ray','pivot_breakout'
            ],
            'ai': [
                'deepseek_interpretation'
            ]
        }

    # -----------------------
    # Public API
    # -----------------------
    def run_analysis(self, market_data: Dict, strategy_type: str = 'all') -> Dict:
        """
        Run strategies and return full report.
        Expects market_data['chart'] = list of candles dicts with keys open/high/low/close/volume/time
        """
        chart = market_data.get('chart') or []
        if not chart or len(chart) < 5:
            raise ValueError("market_data['chart'] must be provided with at least 5 candles")

        # normalize arrays
        opens = [float(c.get('open', c['close'])) for c in chart]
        highs = [float(c['high']) for c in chart]
        lows = [float(c['low']) for c in chart]
        closes = [float(c['close']) for c in chart]
        volumes = [int(c.get('volume', 0)) for c in chart]

        data = {'opens': opens, 'highs': highs, 'lows': lows, 'closes': closes, 'volumes': volumes, 'chart': chart}

        # choose strategies
        if strategy_type == 'all':
            strategy_list = []
            for cat in self.strategy_categories.values():
                strategy_list.extend(cat)
        else:
            strategy_list = self.strategy_categories.get(strategy_type, [])

        results = {}
        buy = sell = neutral = 0
        conf_total = 0.0

        for name in strategy_list:
            fn = getattr(self, f"_{name}", None)
            if not fn:
                # defensive: try synonyms or fallback map
                fn = self._fallback(name)
            if not fn:
                results[name] = StrategySignal('neutral', 40, 'Not implemented').__dict__
                neutral += 1
                conf_total += 40
                continue
            try:
                out = fn(data)
                # ensure standardized dict
                sig = out.get('signal', 'neutral')
                conf = int(round(float(out.get('confidence', 0))))
                reason = out.get('reason', '')
                entry = out.get('entry', None)
                sl = out.get('stop_loss', None)
                tgt = out.get('target', None)
                results[name] = {'signal': sig, 'confidence': conf, 'reason': reason, 'entry': entry, 'stop_loss': sl, 'target': tgt}
                if sig == 'buy': buy += 1
                elif sig == 'sell': sell += 1
                else: neutral += 1
                conf_total += conf
            except Exception as e:
                logger.exception(f"Strategy {name} failed")
                results[name] = {'signal': 'neutral', 'confidence': 0, 'reason': f'Error: {str(e)}', 'entry': None, 'stop_loss': None, 'target': None}
                neutral += 1

        total = len(results) or 1
        avg_conf = conf_total / total
        if buy > sell:
            consensus = 'BUY'
            strength = (buy / total) * 100
        elif sell > buy:
            consensus = 'SELL'
            strength = (sell / total) * 100
        else:
            consensus = 'NEUTRAL'
            strength = 50.0

        return {
            'strategies': results,
            'consensus': {'signal': consensus, 'strength': round(strength, 2), 'confidence': round(avg_conf, 2)},
            'summary': {'total': total, 'buy': buy, 'sell': sell, 'neutral': neutral},
            'market_conditions': self._market_conditions(data)
        }

    # -----------------------
    # Fallbacks / helpers
    # -----------------------
    def _fallback(self, name: str):
        mapping = {
            'ma_crossover': self._ma_crossover,
            'ema_crossover': self._ema_crossover,
            'cci': self._cci,
            'trix_indicator': self._trix_indicator,
            'vwap': self._vwap,
            'anchored_vwap': self._anchored_vwap,
            'on_balance_volume': self._on_balance_volume,
            'chaikin_money_flow': self._chaikin_money_flow,
            'chaikin_oscillator': self._chaikin_oscillator,
            'bollinger_bands': self._bollinger_bands,
            'keltner_channels': self._keltner_channels,
            'supertrend': self._supertrend,
            'elder_ray': self._elder_ray,
            'pivot_points': self._pivot_points,
            'fibonacci_retracement': self._fibonacci_retracement,
            'coppock_curve': self._coppock_curve
        }
        return mapping.get(name)

    def _market_conditions(self, data: Dict) -> Dict:
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        sma20 = self.indicators.calculate_sma(closes, 20) if len(closes) >= 20 else None
        sma50 = self.indicators.calculate_sma(closes, 50) if len(closes) >= 50 else None
        atr = self.indicators.calculate_atr(highs, lows, closes) if len(closes) >= 14 else 0
        volatility = 'High' if atr > (np.mean(closes) * 0.02) else 'Normal'
        trend = 'Unknown'
        if sma20 is not None and sma50 is not None:
            trend = 'Uptrend' if sma20 > sma50 else 'Downtrend' if sma20 < sma50 else 'Sideways'
        support = min(closes[-20:]) if len(closes) >= 20 else min(closes)
        resistance = max(closes[-20:]) if len(closes) >= 20 else max(closes)
        return {'trend': trend, 'volatility': volatility, 'atr': atr, 'support': support, 'resistance': resistance}

    # -----------------------
    # AI overlay (synchronous; safe fallback)
    # -----------------------
    def ai_interpretation(self, symbol: str, analysis: Dict, indicators: Dict = None) -> Dict:
        """
        Synchronous AI interpretation using DeepSeek v3 via OpenRouter.
        If OPENROUTER_KEY missing, returns explanation rather than failing.
        """
        if not OPENROUTER_KEY:
            return {'error': 'OPENROUTER_API_KEY not set; AI disabled', 'ai': None}

        # Build prompt
        top = {k: v for k, v in list(analysis['strategies'].items())[:40]}
        prompt = (
            "You are an expert quantitative trading analyst.\n"
            f"Symbol: {symbol}\n"
            f"Market conditions: {analysis.get('market_conditions')}\n"
            f"Top strategies (trimmed): {top}\n"
            f"Consensus: {analysis.get('consensus')}\n\n"
            "Provide JSON-like output: summary, bias (Bullish/Bearish/Neutral + confidence), "
            "two trade ideas with entry/stop/target and position sizing guidance, and 3 invalidation scenarios."
        )
        headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek/deepseek-chat-v3",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 600
        }
        try:
            resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=45)
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            return {'ai': content, 'source': 'DeepSeek v3 via OpenRouter'}
        except Exception as e:
            logger.exception("AI call failed")
            return {'error': str(e), 'ai': None}

    # -----------------------
    # Strategy Implementations
    # Each returns dict: signal/confidence/reason/(optional entry/stop/target)
    # -----------------------

    # ---------- Momentum ----------
    def _rsi_momentum(self, data: Dict) -> Dict:
        closes = data['closes']
        rsi = self.indicators.calculate_rsi(closes, period=14)
        # handle both list or scalar returns
        try:
            if isinstance(rsi, (list, np.ndarray, pd.Series)):
                latest = float(rsi[-1])
            else:
                latest = float(rsi)
        except Exception:
            return {'signal': 'neutral', 'confidence': 40, 'reason': 'RSI calc failed'}
        if latest < 30:
            return {'signal': 'buy', 'confidence': 88, 'reason': f'RSI {latest:.2f} oversold'}
        if latest > 70:
            return {'signal': 'sell', 'confidence': 88, 'reason': f'RSI {latest:.2f} overbought'}
        if latest < 45:
            return {'signal': 'buy', 'confidence': 55, 'reason': f'RSI {latest:.2f} mild bullish bias'}
        if latest > 55:
            return {'signal': 'sell', 'confidence': 55, 'reason': f'RSI {latest:.2f} mild bearish bias'}
        return {'signal': 'neutral', 'confidence': 45, 'reason': 'RSI neutral'}

    def _macd_crossover(self, data: Dict) -> Dict:
        closes = data['closes']
        macd, signal, hist = self.indicators.calculate_macd(closes, fast=12, slow=26, signal=9)
        if not macd or not signal or len(macd) < 2:
            return {'signal': 'neutral', 'confidence': 40, 'reason': 'Insufficient MACD data'}
        cur_macd, cur_sig = float(macd[-1]), float(signal[-1])
        prev_macd, prev_sig = float(macd[-2]), float(signal[-2])
        if cur_macd > cur_sig and prev_macd <= prev_sig:
            return {'signal':'buy','confidence':90,'reason':'MACD bullish crossover'}
        if cur_macd < cur_sig and prev_macd >= prev_sig:
            return {'signal':'sell','confidence':90,'reason':'MACD bearish crossover'}
        return {'signal':'buy' if cur_macd > cur_sig else 'sell','confidence':60,'reason':'MACD bias'}

    def _stochastic_oscillator(self, data: Dict) -> Dict:
        highs, lows, closes = data['highs'], data['lows'], data['closes']
        k, d = self.indicators.calculate_stochastic(highs, lows, closes, period=14)
        try:
            kf, df = float(k), float(d)
        except Exception:
            return {'signal':'neutral','confidence':40,'reason':'Stochastic calc failed'}
        if kf < 20 and df < 20:
            return {'signal':'buy','confidence':75,'reason':'Stochastic oversold'}
        if kf > 80 and df > 80:
            return {'signal':'sell','confidence':75,'reason':'Stochastic overbought'}
        if kf > df and (kf - df) > 5:
            return {'signal':'buy','confidence':55,'reason':'Stochastic bullish cross'}
        if df > kf and (df - kf) > 5:
            return {'signal':'sell','confidence':55,'reason':'Stochastic bearish cross'}
        return {'signal':'neutral','confidence':45,'reason':'Stochastic neutral'}

    def _williams_r(self, data: Dict) -> Dict:
        highs, lows, closes = data['highs'], data['lows'], data['closes']
        if len(closes) < 14:
            return {'signal':'neutral','confidence':40,'reason':'Insufficient Williams %R data'}
        highest = max(highs[-14:]); lowest = min(lows[-14:])
        wr = (highest - closes[-1]) / (highest - lowest + 1e-9) * -100
        if wr < -80: return {'signal':'buy','confidence':70,'reason':'Williams %R oversold'}
        if wr > -20: return {'signal':'sell','confidence':70,'reason':'Williams %R overbought'}
        return {'signal':'neutral','confidence':45,'reason':'Williams %R neutral'}

    def _cci(self, data: Dict) -> Dict:
        highs, lows, closes = data['highs'], data['lows'], data['closes']
        period = 20
        if len(closes) < period: return {'signal':'neutral','confidence':40,'reason':'Insufficient CCI data'}
        tp = [(h+l+c)/3 for h,l,c in zip(highs, lows, closes)]
        s = pd.Series(tp)
        sma = s.rolling(period).mean().iloc[-1]
        md = s.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x)))).iloc[-1]
        cci_val = (tp[-1] - sma) / (0.015 * (md + 1e-9))
        if cci_val > 100: return {'signal':'sell','confidence':65,'reason':'CCI overbought'}
        if cci_val < -100: return {'signal':'buy','confidence':65,'reason':'CCI oversold'}
        return {'signal':'neutral','confidence':45,'reason':'CCI neutral'}

    def _connors_rsi(self, data: Dict) -> Dict:
        closes = pd.Series(data['closes'])
        if len(closes) < 10: return {'signal':'neutral','confidence':40,'reason':'Insufficient Connors RSI data'}
        rsi1 = self.indicators.calculate_rsi(closes.tolist(), period=3)
        streak = closes.diff().apply(lambda x: 1 if x>0 else (-1 if x<0 else 0)).tolist()
        streak_rsi = self.indicators.calculate_rsi(streak, period=2)
        pct_rank = closes.pct_change().rank(pct=True).iloc[-1] * 100
        crsi = ((rsi1 if isinstance(rsi1,(int,float)) else rsi1[-1]) + (streak_rsi if isinstance(streak_rsi,(int,float)) else (streak_rsi[-1] if len(streak_rsi)>0 else 50)) + pct_rank) / 3
        if crsi < 20: return {'signal':'buy','confidence':80,'reason':'Connors RSI oversold composite'}
        if crsi > 80: return {'signal':'sell','confidence':80,'reason':'Connors RSI overbought composite'}
        return {'signal':'neutral','confidence':50,'reason':'Connors RSI neutral'}

    def _coppock_curve(self, data: Dict) -> Dict:
        closes = pd.Series(data['closes'])
        if len(closes) < 50: return {'signal':'neutral','confidence':40,'reason':'Insufficient Coppock data'}
        roc1 = closes.pct_change(periods=11)
        roc2 = closes.pct_change(periods=14)
        coppock = (roc1 + roc2).rolling(window=10).sum().iloc[-1]
        if coppock > 0: return {'signal':'buy','confidence':65,'reason':'Coppock positive'}
        if coppock < 0: return {'signal':'sell','confidence':65,'reason':'Coppock negative'}
        return {'signal':'neutral','confidence':45,'reason':'Coppock neutral'}

    def _trix_indicator(self, data: Dict) -> Dict:
        closes = pd.Series(data['closes'])
        if len(closes) < 30: return {'signal':'neutral','confidence':40,'reason':'Insufficient TRIX data'}
        ema1 = closes.ewm(span=15).mean(); ema2 = ema1.ewm(span=15).mean(); ema3 = ema2.ewm(span=15).mean()
        trix = ema3.pct_change().iloc[-1] * 100
        if trix > 0: return {'signal':'buy','confidence':60,'reason':'TRIX positive'}
        if trix < 0: return {'signal':'sell','confidence':60,'reason':'TRIX negative'}
        return {'signal':'neutral','confidence':45,'reason':'TRIX neutral'}

    def _ultimate_oscillator(self, data: Dict) -> Dict:
        highs = np.array(data['highs']); lows = np.array(data['lows']); closes = np.array(data['closes'])
        prior_close = np.roll(closes, 1)
        bp = closes - np.minimum(lows, prior_close)
        tr = np.maximum(highs, prior_close) - np.minimum(lows, prior_close)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg7 = pd.Series(bp).rolling(7).sum() / pd.Series(tr).rolling(7).sum()
            avg14 = pd.Series(bp).rolling(14).sum() / pd.Series(tr).rolling(14).sum()
            avg28 = pd.Series(bp).rolling(28).sum() / pd.Series(tr).rolling(28).sum()
        uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
        val = uo.iloc[-1] if not uo.empty else 50
        if val > 65: return {'signal':'sell','confidence':60,'reason':'Ultimate Oscillator overbought'}
        if val < 35: return {'signal':'buy','confidence':60,'reason':'Ultimate Oscillator oversold'}
        return {'signal':'neutral','confidence':45,'reason':'Ultimate Oscillator neutral'}

    # ---------- Trend ----------
    def _ma_crossover(self, data: Dict) -> Dict:
        closes = data['closes']
        sma20 = self.indicators.calculate_sma(closes, 20)
        sma50 = self.indicators.calculate_sma(closes, 50)
        if sma20 is None or sma50 is None: return {'signal':'neutral','confidence':40,'reason':'Insufficient MA data'}
        if sma20 > sma50: return {'signal':'buy','confidence':70,'reason':'SMA20 > SMA50'}
        if sma20 < sma50: return {'signal':'sell','confidence':70,'reason':'SMA20 < SMA50'}
        return {'signal':'neutral','confidence':45,'reason':'MA neutral'}

    def _ema_crossover(self, data: Dict) -> Dict:
        closes = pd.Series(data['closes'])
        if len(closes) < 26: return {'signal':'neutral','confidence':40,'reason':'Insufficient EMA data'}
        ema12 = closes.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = closes.ewm(span=26, adjust=False).mean().iloc[-1]
        if ema12 > ema26: return {'signal':'buy','confidence':68,'reason':'EMA12 > EMA26'}
        if ema12 < ema26: return {'signal':'sell','confidence':68,'reason':'EMA12 < EMA26'}
        return {'signal':'neutral','confidence':45,'reason':'EMA neutral'}

    def _adx_trend(self, data: Dict) -> Dict:
        highs = np.array(data['highs']); lows = np.array(data['lows']); closes = np.array(data['closes'])
        period = 14
        if len(closes) < period*2: return {'signal':'neutral','confidence':40,'reason':'Insufficient ADX data'}
        up_move = highs[1:] - highs[:-1]; down_move = lows[:-1] - lows[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = np.maximum.reduce([highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])])
        atr = pd.Series(tr).rolling(window=period).mean().iloc[-1] if len(tr) >= period else np.mean(tr)
        plus_di = (np.sum(plus_dm[-period:]) / (atr*period + 1e-9)) * 100
        minus_di = (np.sum(minus_dm[-period:]) / (atr*period + 1e-9)) * 100
        adx = abs(plus_di - minus_di)
        if adx > 25:
            if plus_di > minus_di: return {'signal':'buy','confidence':75,'reason':f'ADX strong uptrend ({adx:.1f})'}
            else: return {'signal':'sell','confidence':75,'reason':f'ADX strong downtrend ({adx:.1f})'}
        return {'signal':'neutral','confidence':45,'reason':'ADX weak'}

    def _ichimoku_cloud(self, data: Dict) -> Dict:
        highs = pd.Series(data['highs']); lows = pd.Series(data['lows']); closes = pd.Series(data['closes'])
        if len(closes) < 52: return {'signal':'neutral','confidence':40,'reason':'Insufficient Ichimoku data'}
        tenkan = (highs.rolling(9).max() + lows.rolling(9).min())/2
        kijun = (highs.rolling(26).max() + lows.rolling(26).min())/2
        span_a = ((tenkan + kijun)/2).shift(26)
        span_b = ((highs.rolling(52).max() + lows.rolling(52).min())/2).shift(26)
        price = closes.iloc[-1]
        if price > span_a.iloc[-1] and price > span_b.iloc[-1]: return {'signal':'buy','confidence':78,'reason':'Price above Ichimoku cloud'}
        if price < span_a.iloc[-1] and price < span_b.iloc[-1]: return {'signal':'sell','confidence':78,'reason':'Price below Ichimoku cloud'}
        return {'signal':'neutral','confidence':45,'reason':'In Ichimoku cloud'}

    def _aroon_indicator(self, data: Dict) -> Dict:
        highs = pd.Series(data['highs']); lows = pd.Series(data['lows']); period = 14
        if len(highs) < period: return {'signal':'neutral','confidence':40,'reason':'Insufficient Aroon data'}
        last = len(highs)
        # calculate days since high/low over period
        days_since_high = period - (highs[-period:].idxmax() - (last - period))
        days_since_low = period - (lows[-period:].idxmin() - (last - period))
        aroon_up = ((period - days_since_high)/period) * 100
        aroon_down = ((period - days_since_low)/period) * 100
        if aroon_up > 70 and aroon_down < 30: return {'signal':'buy','confidence':65,'reason':'Aroon uptrend strong'}
        if aroon_down > 70 and aroon_up < 30: return {'signal':'sell','confidence':65,'reason':'Aroon downtrend strong'}
        return {'signal':'neutral','confidence':45,'reason':'Aroon mixed'}

    def _supertrend(self, data: Dict) -> Dict:
        closes = data['closes']; highs = data['highs']; lows = data['lows']
        period = 10; multiplier = 3.0
        if len(closes) < period: return {'signal':'neutral','confidence':40,'reason':'Insufficient Supertrend data'}
        atr = self.indicators.calculate_atr(highs, lows, closes, period=period)
        ma = np.mean(closes[-period:])
        price = closes[-1]
        if price > ma + multiplier * atr: return {'signal':'buy','confidence':75,'reason':'Supertrend bullish'}
        if price < ma - multiplier * atr: return {'signal':'sell','confidence':75,'reason':'Supertrend bearish'}
        return {'signal':'neutral','confidence':45,'reason':'Supertrend neutral'}

    def _keltner_channels(self, data: Dict) -> Dict:
        closes = pd.Series(data['closes']); period = 20
        if len(closes) < period: return {'signal':'neutral','confidence':40,'reason':'Insufficient Keltner data'}
        ma = closes.rolling(period).mean().iloc[-1]
        atr = self.indicators.calculate_atr(data['highs'], data['lows'], data['closes'], period=10)
        upper = ma + 1.5 * atr; lower = ma - 1.5 * atr
        price = closes.iloc[-1]
        if price > upper: return {'signal':'buy','confidence':72,'reason':'Above Keltner upper'}
        if price < lower: return {'signal':'sell','confidence':72,'reason':'Below Keltner lower'}
        return {'signal':'neutral','confidence':45,'reason':'Keltner neutral'}

    def _atr_trailing_stop(self, data: Dict) -> Dict:
        closes = data['closes']
        if len(closes) < 20: return {'signal':'neutral','confidence':40,'reason':'Insufficient ATR trail data'}
        atr = self.indicators.calculate_atr(data['highs'], data['lows'], closes, period=14)
        ma = np.mean(closes[-20:])
        price = closes[-1]
        if price > ma and atr > 0: return {'signal':'buy','confidence':60,'reason':'ATR trailing supports long trend'}
        if price < ma and atr > 0: return {'signal':'sell','confidence':60,'reason':'ATR trailing supports short trend'}
        return {'signal':'neutral','confidence':45,'reason':'ATR trailing neutral'}

    def _hull_ma(self, data: Dict) -> Dict:
        closes = pd.Series(data['closes'])
        if len(closes) < 30: return {'signal':'neutral','confidence':40,'reason':'Insufficient HMA data'}
        n = 16
        wma_n2 = closes.rolling(int(n/2)).mean()
        wma_n = closes.rolling(n).mean()
        hma = (2 * wma_n2 - wma_n).rolling(int(math.sqrt(n))).mean().iloc[-1]
        price = closes.iloc[-1]
        if price > hma: return {'signal':'buy','confidence':60,'reason':'Price above Hull MA'}
        if price < hma: return {'signal':'sell','confidence':60,'reason':'Price below Hull MA'}
        return {'signal':'neutral','confidence':45,'reason':'HMA neutral'}

    # ---------- Volume ----------
    def _vwap(self, data: Dict) -> Dict:
        highs = np.array(data['highs']); lows = np.array(data['lows']); closes = np.array(data['closes']); volumes = np.array(data['volumes'])
        if volumes.sum() == 0: return {'signal':'neutral','confidence':40,'reason':'No volume for VWAP'}
        typical = (highs + lows + closes) / 3.0
        vwap_val = (typical * volumes).sum() / (volumes.sum() + 1e-9)
        price = closes[-1]
        if price > vwap_val: return {'signal':'buy','confidence':68,'reason':'Above VWAP'}
        if price < vwap_val: return {'signal':'sell','confidence':68,'reason':'Below VWAP'}
        return {'signal':'neutral','confidence':45,'reason':'At VWAP'}

    def _anchored_vwap(self, data: Dict) -> Dict:
        return self._vwap(data)  # simple anchor at start for now

    def _on_balance_volume(self, data: Dict) -> Dict:
        closes = data['closes']; volumes = data['volumes']
        if len(closes) < 3: return {'signal':'neutral','confidence':40,'reason':'Insufficient OBV data'}
        obv = [0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]: obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i-1]: obv.append(obv[-1] - volumes[i])
            else: obv.append(obv[-1])
        slope = obv[-1] - obv[-5] if len(obv) >= 5 else 0
        if slope > 0: return {'signal':'buy','confidence':62,'reason':'OBV rising (accumulation)'}
        if slope < 0: return {'signal':'sell','confidence':62,'reason':'OBV falling (distribution)'}
        return {'signal':'neutral','confidence':45,'reason':'OBV flat'}

    def _chaikin_money_flow(self, data: Dict) -> Dict:
        highs = np.array(data['highs']); lows = np.array(data['lows']); closes = np.array(data['closes']); volumes = np.array(data['volumes'])
        period = 20
        if len(closes) < period: return {'signal':'neutral','confidence':40,'reason':'Insufficient CMF data'}
        mfv = ((closes - lows) - (highs - closes)) / (highs - lows + 1e-9) * volumes
        cmf = pd.Series(mfv).rolling(period).sum().iloc[-1] / (pd.Series(volumes).rolling(period).sum().iloc[-1] + 1e-9)
        if cmf > 0.05: return {'signal':'buy','confidence':70,'reason':'CMF positive (inflow)'}
        if cmf < -0.05: return {'signal':'sell','confidence':70,'reason':'CMF negative (outflow)'}
        return {'signal':'neutral','confidence':45,'reason':'CMF neutral'}

    def _chaikin_oscillator(self, data: Dict) -> Dict:
        highs = np.array(data['highs']); lows = np.array(data['lows']); closes = np.array(data['closes']); volumes = np.array(data['volumes'])
        ad = ((closes - lows) - (highs - closes)) / (highs - lows + 1e-9) * volumes
        if len(ad) < 10: return {'signal':'neutral','confidence':40,'reason':'Insufficient Chaikin data'}
        fast = pd.Series(ad).ewm(span=3, adjust=False).mean().iloc[-1]
        slow = pd.Series(ad).ewm(span=10, adjust=False).mean().iloc[-1]
        val = fast - slow
        if val > 0: return {'signal':'buy','confidence':62,'reason':'Chaikin oscillator positive'}
        if val < 0: return {'signal':'sell','confidence':62,'reason':'Chaikin oscillator negative'}
        return {'signal':'neutral','confidence':45,'reason':'Chaikin neutral'}

    def _volume_spike(self, data: Dict) -> Dict:
        vols = pd.Series(data['volumes'])
        if len(vols) < 20: return {'signal':'neutral','confidence':40,'reason':'Insufficient volume history'}
        avg = vols[-20:].mean()
        if vols.iloc[-1] > 2.0 * avg: return {'signal':'buy','confidence':68,'reason':'Volume spike (participation)'}
        return {'signal':'neutral','confidence':45,'reason':'No big spike'}

    def _vroc(self, data: Dict) -> Dict:
        vols = pd.Series(data['volumes'])
        if len(vols) < 21: return {'signal':'neutral','confidence':40,'reason':'Insufficient VROC data'}
        vroc = (vols.iloc[-1] - vols.iloc[-21]) / (vols.iloc[-21] + 1e-9) * 100
        if vroc > 50: return {'signal':'buy','confidence':60,'reason':'VROC strong positive'}
        if vroc < -50: return {'signal':'sell','confidence':60,'reason':'VROC strong negative'}
        return {'signal':'neutral','confidence':45,'reason':'VROC neutral'}

    # ---------- Volatility ----------
    def _bollinger_bands(self, data: Dict) -> Dict:
        closes = data['closes']
        upper, mid, lower = self.indicators.calculate_bollinger_bands(closes, period=20, std_dev=2)
        price = closes[-1]
        if price < lower[-1]: return {'signal':'buy','confidence':75,'reason':'Price below lower BB (mean reversion)'}
        if price > upper[-1]: return {'signal':'sell','confidence':75,'reason':'Price above upper BB (mean reversion short)'}
        return {'signal':'neutral','confidence':45,'reason':'Within BB'}

    def _keltner_breakout(self, data: Dict) -> Dict:
        return self._keltner_channels(data)

    def _atr_breakout(self, data: Dict) -> Dict:
        atr = self.indicators.calculate_atr(data['highs'], data['lows'], data['closes'], period=14)
        price = data['closes'][-1]; ma = np.mean(data['closes'][-20:])
        if atr <= 0: return {'signal':'neutral','confidence':40,'reason':'ATR insufficient'}
        if price > ma + 1.5 * atr: return {'signal':'buy','confidence':70,'reason':'ATR breakout above band'}
        if price < ma - 1.5 * atr: return {'signal':'sell','confidence':70,'reason':'ATR breakdown below band'}
        return {'signal':'neutral','confidence':45,'reason':'No ATR breakout'}

    def _mass_index(self, data: Dict) -> Dict:
        highs, lows = pd.Series(data['highs']), pd.Series(data['lows'])
        if len(highs) < 25: return {'signal':'neutral','confidence':40,'reason':'Insufficient Mass Index data'}
        hi_lo = highs - lows
        ema1 = hi_lo.ewm(span=9, adjust=False).mean()
        ema2 = ema1.ewm(span=9, adjust=False).mean()
        mass = (ema1 / (ema2 + 1e-9)).rolling(window=25).sum().iloc[-1]
        if mass > 27: return {'signal':'buy','confidence':62,'reason':'Mass Index potential reversal'}
        return {'signal':'neutral','confidence':45,'reason':'Mass Index neutral'}

    def _bollinger_squeeze(self, data: Dict) -> Dict:
        closes = pd.Series(data['closes'])
        if len(closes) < 20: return {'signal':'neutral','confidence':40,'reason':'Insufficient squeeze data'}
        upper, mid, lower = self.indicators.calculate_bollinger_bands(closes.tolist(), period=20, std_dev=2)
        width = (upper[-1] - lower[-1]) / (mid[-1] + 1e-9)
        if width < 0.04: return {'signal':'neutral','confidence':60,'reason':'Bollinger squeeze (low vol) - watch breakout'}
        return {'signal':'neutral','confidence':45,'reason':'No squeeze'}

    # ---------- Price Action ----------
    def _bullish_engulfing(self, data: Dict) -> Dict:
        c = data['chart']; 
        if len(c) < 2: return {'signal':'neutral','confidence':40,'reason':'Need 2 candles'}
        prev, cur = c[-2], c[-1]
        if prev['close'] < prev['open'] and cur['close'] > cur['open'] and cur['close'] >= prev['open'] and cur['open'] <= prev['close']:
            return {'signal':'buy','confidence':78,'reason':'Bullish Engulfing'}
        return {'signal':'neutral','confidence':45,'reason':'No pattern'}

    def _bearish_engulfing(self, data: Dict) -> Dict:
        c = data['chart']; 
        if len(c) < 2: return {'signal':'neutral','confidence':40,'reason':'Need 2 candles'}
        prev, cur = c[-2], c[-1]
        if prev['close'] > prev['open'] and cur['close'] < cur['open'] and cur['open'] >= prev['close'] and cur['close'] <= prev['open']:
            return {'signal':'sell','confidence':78,'reason':'Bearish Engulfing'}
        return {'signal':'neutral','confidence':45,'reason':'No pattern'}

    def _hammer(self, data: Dict) -> Dict:
        c = data['chart'][-1]
        body = abs(c['close'] - c['open']); lower_shadow = min(c['open'], c['close']) - c['low']; upper_shadow = c['high'] - max(c['open'], c['close'])
        if lower_shadow > 2 * body and upper_shadow < body: return {'signal':'buy','confidence':70,'reason':'Hammer bullish reversal'}
        return {'signal':'neutral','confidence':45,'reason':'No hammer'}

    def _shooting_star(self, data: Dict) -> Dict:
        c = data['chart'][-1]
        body = abs(c['close'] - c['open']); upper_shadow = c['high'] - max(c['open'], c['close']); lower_shadow = min(c['open'], c['close']) - c['low']
        if upper_shadow > 2 * body and lower_shadow < body: return {'signal':'sell','confidence':70,'reason':'Shooting Star bearish reversal'}
        return {'signal':'neutral','confidence':45,'reason':'No shooting star'}

    def _doji(self, data: Dict) -> Dict:
        c = data['chart'][-1]; body = abs(c['close'] - c['open']); range_ = c['high'] - c['low']
        if range_ > 0 and (body / range_) < 0.1: return {'signal':'neutral','confidence':60,'reason':'Doji indecision'}
        return {'signal':'neutral','confidence':45,'reason':'No Doji'}

    def _morning_star(self, data: Dict) -> Dict:
        c = data['chart']; 
        if len(c) < 3: return {'signal':'neutral','confidence':40,'reason':'Need 3 candles'}
        a,b,c3 = c[-3], c[-2], c[-1]
        if a['close'] < a['open'] and abs(b['close']-b['open'])<abs(a['close']-a['open']) and c3['close']>c3['open'] and c3['close'] > (a['open']+a['close'])/2:
            return {'signal':'buy','confidence':75,'reason':'Morning Star bullish'}
        return {'signal':'neutral','confidence':45,'reason':'No Morning Star'}

    def _evening_star(self, data: Dict) -> Dict:
        c = data['chart']; 
        if len(c) < 3: return {'signal':'neutral','confidence':40,'reason':'Need 3 candles'}
        a,b,c3 = c[-3], c[-2], c[-1]
        if a['close'] > a['open'] and abs(b['close']-b['open'])<abs(a['close']-a['open']) and c3['close'] < c3['open'] and c3['close'] < (a['open']+a['close'])/2:
            return {'signal':'sell','confidence':75,'reason':'Evening Star bearish'}
        return {'signal':'neutral','confidence':45,'reason':'No Evening Star'}

    def _piercing(self, data: Dict) -> Dict:
        c = data['chart']; 
        if len(c) < 2: return {'signal':'neutral','confidence':40,'reason':'Need 2 candles'}
        prev, cur = c[-2], c[-1]
        if prev['close'] < prev['open'] and cur['open'] < prev['close'] and cur['close'] > prev['open'] and cur['close'] < (prev['open']+prev['close'])/2:
            return {'signal':'buy','confidence':72,'reason':'Piercing pattern bullish'}
        return {'signal':'neutral','confidence':45,'reason':'No piercing'}

    def _dark_cloud_cover(self, data: Dict) -> Dict:
        c = data['chart']; 
        if len(c) < 2: return {'signal':'neutral','confidence':40,'reason':'Need 2 candles'}
        prev, cur = c[-2], c[-1]
        if prev['close'] > prev['open'] and cur['open'] > prev['close'] and cur['close'] < prev['open'] and cur['close'] > (prev['open']+prev['close'])/2:
            return {'signal':'sell','confidence':72,'reason':'Dark Cloud Cover bearish'}
        return {'signal':'neutral','confidence':45,'reason':'No Dark Cloud Cover'}

    def _bullish_harami(self, data: Dict) -> Dict:
        c = data['chart']; 
        if len(c) < 2: return {'signal':'neutral','confidence':40,'reason':'Need 2 candles'}
        p,cur = c[-2], c[-1]
        if p['open'] > p['close'] and cur['open'] > cur['close'] and cur['open'] > p['close'] and cur['close'] < p['open']:
            return {'signal':'buy','confidence':65,'reason':'Bullish Harami'}
        return {'signal':'neutral','confidence':45,'reason':'No Bullish Harami'}

    def _bearish_harami(self, data: Dict) -> Dict:
        c = data['chart']; 
        if len(c) < 2: return {'signal':'neutral','confidence':40,'reason':'Need 2 candles'}
        p,cur = c[-2], c[-1]
        if p['open'] < p['close'] and cur['open'] < cur['close'] and cur['open'] < p['close'] and cur['close'] > p['open']:
            return {'signal':'sell','confidence':65,'reason':'Bearish Harami'}
        return {'signal':'neutral','confidence':45,'reason':'No Bearish Harami'}

    # ---------- Swing ----------
    def _pivot_points(self, data: Dict) -> Dict:
        highs = data['highs'][-10:]; lows = data['lows'][-10:]; closes = data['closes'][-10:]
        if not highs or not lows or not closes: return {'signal':'neutral','confidence':40,'reason':'Insufficient pivot data'}
        pivot = (max(highs) + min(lows) + closes[-1]) / 3.0
        price = closes[-1]
        if price > pivot: return {'signal':'buy','confidence':55,'reason':'Price above pivot'}
        if price < pivot: return {'signal':'sell','confidence':55,'reason':'Price below pivot'}
        return {'signal':'neutral','confidence':45,'reason':'Near pivot'}

    def _fibonacci_retracement(self, data: Dict) -> Dict:
        closes = data['closes']
        if len(closes) < 10: return {'signal':'neutral','confidence':40,'reason':'Insufficient Fib data'}
        high = max(closes[-20:]); low = min(closes[-20:]); price = closes[-1]
        fib38 = low + 0.382*(high-low); fib61 = low + 0.618*(high-low)
        if abs(price - fib38) / (high - low + 1e-9) < 0.01: return {'signal':'buy','confidence':58,'reason':'Near 38.2% retrace support'}
        if abs(price - fib61) / (high - low + 1e-9) < 0.01: return {'signal':'sell','confidence':58,'reason':'Near 61.8% retrace resistance'}
        return {'signal':'neutral','confidence':45,'reason':'Fib not decisive'}

    def _elder_ray(self, data: Dict) -> Dict:
        closes = pd.Series(data['closes']); highs = pd.Series(data['highs']); lows = pd.Series(data['lows'])
        if len(closes) < 20: return {'signal':'neutral','confidence':40,'reason':'Insufficient Elder-Ray data'}
        ema13 = closes.ewm(span=13, adjust=False).mean().iloc[-1]
        bull_power = highs.iloc[-1] - ema13; bear_power = lows.iloc[-1] - ema13
        if bull_power > 0 and bear_power > -0.5 * abs(bull_power): return {'signal':'buy','confidence':60,'reason':'Elder-Ray bullish'}
        if bear_power < 0 and bull_power < 0.5 * abs(bear_power): return {'signal':'sell','confidence':60,'reason':'Elder-Ray bearish'}
        return {'signal':'neutral','confidence':45,'reason':'Elder-Ray neutral'}

    def _pivot_breakout(self, data: Dict) -> Dict:
        closes = data['closes']; pivot_val = (min(data['lows'][-10:]) + max(data['highs'][-10:]) + closes[-1]) / 3.0
        price = closes[-1]
        if price > pivot_val * 1.002: return {'signal':'buy','confidence':68,'reason':'Breakout above pivot'}
        if price < pivot_val * 0.998: return {'signal':'sell','confidence':68,'reason':'Breakdown below pivot'}
        return {'signal':'neutral','confidence':45,'reason':'No pivot breakout'}


