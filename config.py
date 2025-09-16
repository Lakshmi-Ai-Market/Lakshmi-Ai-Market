import os
from datetime import timedelta

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    
    # Cache settings
    CACHE_TYPE = 'simple'  # Use 'redis' for production
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'memory://'
    
    # API Keys (optional - Yahoo Finance doesn't require API key)
    ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
    TWELVE_DATA_API_KEY = os.environ.get('TWELVE_DATA_API_KEY')
    IEX_API_KEY = os.environ.get('IEX_API_KEY')
    
    # Data source priorities
    DATA_SOURCES = [
        'yahoo_finance',  # Primary (no API key needed)
        'alpha_vantage',  # Fallback (requires API key)
        'twelve_data',    # Fallback (requires API key)
        'iex_cloud'       # Fallback (requires API key)
    ]
    
    # Trading strategy settings
    DEFAULT_PERIOD = 60  # days
    DEFAULT_INTERVAL = '1d'
    MAX_PERIOD = 365
    
    # Performance settings
    MAX_WORKERS = 4
    REQUEST_TIMEOUT = 30
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    CACHE_TYPE = 'simple'
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL')
    
# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}