import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

class TechnicalIndicators:
    """Technical indicator calculations"""
    
    def calculate_sma(self, prices: List[float], period: int = 20) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        return sum(prices[-period:]) / period
    
    def calculate_ema(self, prices: List[float], period: int = 20) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return self.calculate_sma(prices, len(prices))
        
        multiplier = 2 / (period + 1)
        ema = self.calculate_sma(prices[:period], period)
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [change if change > 0 else 0 for change in changes[-period:]]
        losses = [-change if change < 0 else 0 for change in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD Calculation"""
        if len(prices) < slow:
            return [0], [0], [0]
        
        # Calculate EMAs
        ema_fast = []
        ema_slow = []
        
        for i in range(len(prices)):
            if i >= fast - 1:
                ema_fast.append(self.calculate_ema(prices[:i+1], fast))
            else:
                ema_fast.append(prices[i])
            
            if i >= slow - 1:
                ema_slow.append(self.calculate_ema(prices[:i+1], slow))
            else:
                ema_slow.append(prices[i])
        
        # MACD line
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(prices))]
        
        # Signal line (EMA of MACD)
        signal_line = []
        for i in range(len(macd_line)):
            if i >= signal - 1:
                signal_line.append(self.calculate_ema(macd_line[:i+1], signal))
            else:
                signal_line.append(macd_line[i])
        
        # Histogram
        histogram = [macd_line[i] - signal_line[i] for i in range(len(macd_line))]
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands"""
        if len(prices) < period:
            price = prices[-1] if prices else 0
            return [price], [price], [price]
        
        upper_band = []
        middle_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if i >= period - 1:
                window = prices[i-period+1:i+1]
                sma = sum(window) / period
                variance = sum((x - sma) ** 2 for x in window) / period
                std = variance ** 0.5
                
                upper_band.append(sma + (std * std_dev))
                middle_band.append(sma)
                lower_band.append(sma - (std * std_dev))
            else:
                price = prices[i]
                upper_band.append(price)
                middle_band.append(price)
                lower_band.append(price)
        
        return upper_band, middle_band, lower_band
    
    def calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[float, float]:
        """Stochastic Oscillator"""
        if len(closes) < period:
            return 50, 50
        
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        d_percent = k_percent * 0.8  # Simplified %D calculation
        
        return k_percent, d_percent
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Average True Range"""
        if len(closes) < 2:
            return 0
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges)
        
        return sum(true_ranges[-period:]) / period
    
    def calculate_all(self, chart_data: List[Dict]) -> Dict:
        """Calculate all technical indicators"""
        if not chart_data:
            return {}
        
        closes = [float(item['close']) for item in chart_data]
        highs = [float(item['high']) for item in chart_data]
        lows = [float(item['low']) for item in chart_data]
        volumes = [int(item['volume']) for item in chart_data]
        
        try:
            indicators = {
                'sma_20': self.calculate_sma(closes, 20),
                'sma_50': self.calculate_sma(closes, 50),
                'ema_12': self.calculate_ema(closes, 12),
                'ema_26': self.calculate_ema(closes, 26),
                'rsi': self.calculate_rsi(closes),
                'atr': self.calculate_atr(highs, lows, closes)
            }
            
            # MACD
            macd_line, signal_line, histogram = self.calculate_macd(closes)
            indicators['macd'] = {
                'macd': macd_line[-1] if macd_line else 0,
                'signal': signal_line[-1] if signal_line else 0,
                'histogram': histogram[-1] if histogram else 0
            }
            
            # Bollinger Bands
            upper, middle, lower = self.calculate_bollinger_bands(closes)
            indicators['bollinger_bands'] = {
                'upper': upper[-1] if upper else 0,
                'middle': middle[-1] if middle else 0,
                'lower': lower[-1] if lower else 0
            }
            
            # Stochastic
            k, d = self.calculate_stochastic(highs, lows, closes)
            indicators['stochastic'] = {
                'k': k,
                'd': d
            }
            
            return indicators
            
        except Exception as e:
            return {'error': f'Indicator calculation failed: {str(e)}'}