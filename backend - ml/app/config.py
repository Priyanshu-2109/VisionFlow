import os
from celery.schedules import crontab

class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS', 'False').lower() == 'true'
    # Redis
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
    REDIS_SSL = os.getenv('REDIS_SSL', 'false').lower() == 'true'
    
    # Celery
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND')
    
    # File Upload
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    
    # CORS
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 60))
    RATE_LIMIT_PER_HOUR = int(os.getenv('RATE_LIMIT_PER_HOUR', 1000))
    
    # Auth Service
    AUTH_SERVICE_URL = os.getenv('AUTH_SERVICE_URL', 'http://localhost:3000/api/auth')
    AUTH_SERVICE_TIMEOUT = int(os.getenv('AUTH_SERVICE_TIMEOUT', 30))
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
    
    # Model paths
    TREND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'static/model_weights/trend_model.pkl')
    RISK_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  'static/model_weights/risk_model.pkl')
    
    # Celery Beat schedule
    CELERYBEAT_SCHEDULE = {
        'collect-technology-trends': {
            'task': 'app.tasks.trend_analysis.collect_trend_data_periodic',
            'schedule': crontab(hour=0, minute=0),  # Daily at midnight
            'args': ('Technology', ['AI', 'machine learning', 'cloud computing', 'cybersecurity'])
        },
        'collect-finance-trends': {
            'task': 'app.tasks.trend_analysis.collect_trend_data_periodic',
            'schedule': crontab(hour=1, minute=0),  # Daily at 1 AM
            'args': ('Finance', ['fintech', 'banking', 'investment', 'cryptocurrency'])
        },
        'collect-retail-trends': {
            'task': 'app.tasks.trend_analysis.collect_trend_data_periodic',
            'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
            'args': ('Retail', ['e-commerce', 'retail', 'consumer', 'shopping'])
        }
    }

class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    REDIS_URL = 'redis://localhost:6379/1'  # Use different Redis DB for tests
    CORS_ORIGINS = ['http://localhost:5173']