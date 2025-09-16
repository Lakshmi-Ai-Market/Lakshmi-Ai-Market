import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class StrategyEngine:
    """Handles all trading strategy calculations and analysis"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.strategies = {
            'momentum': [
                'rsi_momentum', 'macd_crossover', 'stochastic_oscillator',
                'williams_r', 'commodity_channel_index', 'money_flow_index'
            ],
            'reversal': [
                'hammer_pattern', 'doji_pattern', 'bollinger_bands', 'parabolic_sar'
            ],
            'trend': [
                'moving_average_crossover', 'exponential_moving_average',
                'adx_trend', 'ichimoku_cloud', 'aroon_indicator'
            ],
            'scalping': [
                'breakout_strategy', 'gap_analysis', 'average_true_range', 'elder_ray_index'
            ],
            'swing': [
                'pivot_points', 'fibonacci_retracement', 'mass_index'
            ],
            'volume': [
                'volume_weighted_average', 'on_balance_volume', 'chaikin_oscillator'
            ]
        }
    
    def run_analysis(self, market_data: Dict, strategy_type: str = 'all') -> Dict:
        """
        Run comprehensive trading strategy analysis
        
        Args:
            market_data: Market data dictionary
            strategy_type: Type of strategies to run
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            chart_data = market_data['chart']
            if len(chart_data) < 20:
                raise ValueError("Insufficient data for analysis")
            
            # Prepare data arrays
            closes = [float(item['close']) for item in chart_data]
            highs = [float(item['high']) for item in chart_data]
            lows = [float(item['low']) for item in chart_data]
            volumes = [int(item['volume']) for item in chart_data]
            
            analysis_data = {
                'closes': closes,
                'highs': highs,
                'lows': lows,
                'volumes': volumes,
                'candles': chart_data
            }
            
            # Determine which strategies to run
            if strategy_type == 'all':
                strategies_to_run = []
                for category in self.strategies.values():
                    strategies_to_run.extend(category)
            else:
                strategies_to_run = self.strategies.get(strategy_type, [])
            
            # Run strategies
            results = {}
            buy_signals = 0
            sell_signals = 0
            total_confidence = 0
            
            for strategy_name in strategies_to_run:
                try:
                    result = self._run_strategy(strategy_name, analysis_data)
                    results[strategy_name] = result
                    
                    if result['signal'] == 'buy':
                        buy_signals += 1
                    elif result['signal'] == 'sell':
                        sell_signals += 1
                    
                    total_confidence += result['confidence']
                    
                except Exception as e:
                    logger.error(f"Strategy {strategy_name} failed: {str(e)}")
                    results[strategy_name] = {
                        'signal': 'neutral',
                        'confidence': 0,
                        'reason': f'Error: {str(e)}'
                    }
            
            # Calculate consensus
            strategy_count = len(results)
            avg_confidence = total_confidence / strategy_count if strategy_count > 0 else 0
            
            if buy_signals > sell_signals:
                consensus = 'BUY'
                consensus_strength = (buy_signals / strategy_count) * 100
            elif sell_signals > buy_signals:
                consensus = 'SELL'
                consensus_strength = (sell_signals / strategy_count) * 100
            else:
                consensus = 'NEUTRAL'
                consensus_strength = 50
            
            return {
                'strategies': results,
                'consensus': {
                    'signal': consensus,
                    'strength': round(consensus_strength, 2),
                    'confidence': round(avg_confidence, 2)
                },
                'summary': {
                    'total_strategies': strategy_count,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'neutral_signals': strategy_count - buy_signals - sell_signals
                },
                'market_conditions': self._analyze_market_conditions(analysis_data)
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _run_strategy(self, strategy_name: str, data: Dict) -> Dict:
        """Run individual trading strategy"""
        
        if strategy_name == 'rsi_momentum':
            return self._rsi_momentum_strategy(data)
        elif strategy_name == 'macd_crossover':
            return self._macd_crossover_strategy(data)
        elif strategy_name == 'bollinger_bands':
            return self._bollinger_bands_strategy(data)
        elif strategy_name == 'stochastic_oscillator':
            return self._stochastic_strategy(data)
        elif strategy_name == 'moving_average_crossover':
            return self._ma_crossover_strategy(data)
        elif strategy_name == 'volume_weighted_average':
            return self._vwap_strategy(data)
        elif strategy_name == 'breakout_strategy':
            return self._breakout_strategy(data)
        # Add more strategies as needed
        else:
            return {
                'signal': 'neutral',
                'confidence': 50,
                'reason': f'Strategy {strategy_name} not implemented'
            }
    
    def _rsi_momentum_strategy(self, data: Dict) -> Dict:
        """RSI Momentum Strategy"""
        rsi = self.indicators.calculate_rsi(data['closes'])
        rsi_fast = self.indicators.calculate_rsi(data['closes'], period=7)
        
        if rsi < 30 and rsi_fast < 25:
            return {'signal': 'buy', 'confidence': 90, 'reason': 'Strong RSI Oversold Signal'}
        elif rsi > 70 and rsi_fast > 75:
            return {'signal': 'sell', 'confidence': 90, 'reason': 'Strong RSI Overbought Signal'}
        elif rsi < 40:
            return {'signal': 'buy', 'confidence': 65, 'reason': 'RSI Bullish Territory'}
        elif rsi > 60:
            return {'signal': 'sell', 'confidence': 65, 'reason': 'RSI Bearish Territory'}
        else:
            return {'signal': 'neutral', 'confidence': 40, 'reason': 'RSI Neutral Zone'}
    
    def _macd_crossover_strategy(self, data: Dict) -> Dict:
        """MACD Crossover Strategy"""
        macd_line, signal_line, histogram = self.indicators.calculate_macd(data['closes'])
        
        if len(macd_line) < 2:
            return {'signal': 'neutral', 'confidence': 0, 'reason': 'Insufficient MACD data'}
        
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        prev_macd = macd_line[-2]
        prev_signal = signal_line[-2]
        
        if current_macd > current_signal and prev_macd <= prev_signal:
            return {'signal': 'buy', 'confidence': 85, 'reason': 'MACD Bullish Crossover'}
        elif current_macd < current_signal and prev_macd >= prev_signal:
            return {'signal': 'sell', 'confidence': 85, 'reason': 'MACD Bearish Crossover'}
        elif current_macd > current_signal:
            return {'signal': 'buy', 'confidence': 60, 'reason': 'MACD Above Signal'}
        else:
            return {'signal': 'sell', 'confidence': 60, 'reason': 'MACD Below Signal'}
    
    def _bollinger_bands_strategy(self, data: Dict) -> Dict:
        """Bollinger Bands Strategy"""
        upper, middle, lower = self.indicators.calculate_bollinger_bands(data['closes'])
        current_price = data['closes'][-1]
        
        if current_price < lower[-1]:
            return {'signal': 'buy', 'confidence': 75, 'reason': 'Price Below Lower Bollinger Band'}
        elif current_price > upper[-1]:
            return {'signal': 'sell', 'confidence': 75, 'reason': 'Price Above Upper Bollinger Band'}
        else:
            return {'signal': 'neutral', 'confidence': 40, 'reason': 'Price Within Bollinger Bands'}
    
    def generate_ai_suggestions(self, market_data: Dict) -> Dict:
        """Generate AI-powered trading suggestions"""
        try:
            chart_data = market_data['chart']
            closes = [float(item['close']) for item in chart_data]
            volumes = [int(item['volume']) for item in chart_data]
            
            # Calculate key indicators
            rsi = self.indicators.calculate_rsi(closes)
            macd_line, signal_line, _ = self.indicators.calculate_macd(closes)
            upper, middle, lower = self.indicators.calculate_bollinger_bands(closes)
            
            current_price = closes[-1]
            avg_volume = sum(volumes) / len(volumes)
            current_volume = volumes[-1]
            
            suggestions = []
            confidence_factors = []
            
            # RSI Analysis
            if rsi < 30:
                suggestions.append({
                    'type': 'technical',
                    'indicator': 'RSI',
                    'message': f'RSI at {rsi:.2f} indicates oversold conditions. Consider buying on next uptick.',
                    'confidence': 80
                })
                confidence_factors.append(80)
            elif rsi > 70:
                suggestions.append({
                    'type': 'technical',
                    'indicator': 'RSI',
                    'message': f'RSI at {rsi:.2f} shows overbought conditions. Consider profit booking.',
                    'confidence': 80
                })
                confidence_factors.append(80)
            
            # Volume Analysis
            volume_ratio = current_volume / avg_volume
            if volume_ratio > 1.5:
                suggestions.append({
                    'type': 'volume',
                    'indicator': 'Volume',
                    'message': f'Volume is {volume_ratio:.1f}x above average. Strong conviction in current move.',
                    'confidence': 70
                })
                confidence_factors.append(70)
            
            # MACD Analysis
            if len(macd_line) >= 2:
                if macd_line[-1] > signal_line[-1]:
                    suggestions.append({
                        'type': 'momentum',
                        'indicator': 'MACD',
                        'message': 'MACD above signal line indicates bullish momentum.',
                        'confidence': 65
                    })
                    confidence_factors.append(65)
            
            # Bollinger Bands Analysis
            if current_price < lower[-1]:
                suggestions.append({
                    'type': 'mean_reversion',
                    'indicator': 'Bollinger Bands',
                    'message': 'Price below lower Bollinger Band suggests potential bounce.',
                    'confidence': 70
                })
                confidence_factors.append(70)
            
            # Overall confidence
            overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 50
            
            # Risk management suggestion
            suggestions.append({
                'type': 'risk_management',
                'indicator': 'General',
                'message': 'Always use stop-loss orders and position sizing based on risk tolerance.',
                'confidence': 100
            })
            
            return {
                'suggestions': suggestions,
                'overall_confidence': round(overall_confidence, 2),
                'market_sentiment': self._determine_sentiment(rsi, macd_line, signal_line),
                'risk_level': self._assess_risk_level(market_data),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"AI suggestion generation failed: {str(e)}")
            return {
                'suggestions': [{
                    'type': 'error',
                    'message': 'Unable to generate AI suggestions at this time.',
                    'confidence': 0
                }],
                'overall_confidence': 0
            }
    
    def _determine_sentiment(self, rsi: float, macd_line: List, signal_line: List) -> str:
        """Determine overall market sentiment"""
        bullish_factors = 0
        bearish_factors = 0
        
        if rsi < 40:
            bullish_factors += 1
        elif rsi > 60:
            bearish_factors += 1
        
        if len(macd_line) > 0 and len(signal_line) > 0:
            if macd_line[-1] > signal_line[-1]:
                bullish_factors += 1
            else:
                bearish_factors += 1
        
        if bullish_factors > bearish_factors:
            return 'Bullish'
        elif bearish_factors > bullish_factors:
            return 'Bearish'
        else:
            return 'Neutral'
    
    def _assess_risk_level(self, market_data: Dict) -> str:
        """Assess current risk level"""
        try:
            # Calculate volatility
            closes = [float(item['close']) for item in market_data['chart']]
            returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            if volatility > 0.3:
                return 'High'
            elif volatility > 0.15:
                return 'Medium'
            else:
                return 'Low'
        except:
            return 'Medium'
    
    def _analyze_market_conditions(self, data: Dict) -> Dict:
        """Analyze current market conditions"""
        try:
            closes = data['closes']
            volumes = data['volumes']
            
            # Trend analysis
            sma_20 = self.indicators.calculate_sma(closes, 20)
            sma_50 = self.indicators.calculate_sma(closes, 50)
            
            if sma_20 > sma_50:
                trend = 'Uptrend'
            elif sma_20 < sma_50:
                trend = 'Downtrend'
            else:
                trend = 'Sideways'
            
            # Volatility
            atr = self.indicators.calculate_atr(data['highs'], data['lows'], closes)
            volatility = 'High' if atr > np.mean(closes) * 0.02 else 'Normal'
            
            # Volume trend
            avg_volume = np.mean(volumes)
            recent_volume = np.mean(volumes[-5:])
            volume_trend = 'Increasing' if recent_volume > avg_volume * 1.1 else 'Normal'
            
            return {
                'trend': trend,
                'volatility': volatility,
                'volume_trend': volume_trend,
                'support_level': min(closes[-20:]),
                'resistance_level': max(closes[-20:])
            }
            
        except Exception as e:
            logger.error(f"Market condition analysis failed: {str(e)}")
            return {
                'trend': 'Unknown',
                'volatility': 'Unknown',
                'volume_trend': 'Unknown'
            }