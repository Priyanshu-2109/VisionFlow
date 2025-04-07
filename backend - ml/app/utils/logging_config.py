import logging
import os
from logging.handlers import RotatingFileHandler
import time
import json
from flask import request, g, has_request_context

class RequestFormatter(logging.Formatter):
    """Custom formatter to add request info to logs"""
    
    def format(self, record):
        if has_request_context():
            record.url = getattr(request, 'url', None)
            record.remote_addr = getattr(request, 'remote_addr', None)
            record.method = getattr(request, 'method', None)
            record.request_id = getattr(g, 'request_id', 'no-request-id') if hasattr(g, 'request_id') else 'no-request-id'
        else:
            record.url = None
            record.remote_addr = None
            record.method = None
            record.request_id = 'no-request-id'
        
        return super().format(record)

def setup_logging(app):
    """Configure application logging"""
    # Ensure log directory exists
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file handler for error logs
    error_file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    error_file_handler.setLevel(logging.ERROR)
    
    # Set up file handler for info logs
    info_file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'info.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    info_file_handler.setLevel(logging.INFO)
    
    # Set up formatter
    formatter = RequestFormatter(
        '%(asctime)s %(levelname)s [%(request_id)s] %(remote_addr)s '
        '%(method)s %(url)s - %(name)s: %(message)s'
    )
    error_file_handler.setFormatter(formatter)
    info_file_handler.setFormatter(formatter)
    
    # Add handlers to app logger
    app.logger.addHandler(error_file_handler)
    app.logger.addHandler(info_file_handler)
    
    # Set app logger level
    app.logger.setLevel(logging.INFO)
    
    # Configure other loggers
    for logger_name in ['werkzeug', 'sqlalchemy.engine']:
        logger = logging.getLogger(logger_name)
        logger.addHandler(info_file_handler)
    
    # Add request ID middleware
    @app.before_request
    def before_request():
        g.request_id = request.headers.get('X-Request-ID', f"{time.time():.0f}")
    
    # Log all requests
    @app.after_request
    def after_request(response):
        # Skip logging for static files
        if not request.path.startswith('/static'):
            log_data = {
                'remote_addr': request.remote_addr,
                'method': request.method,
                'path': request.path,
                'status': response.status_code,
                'user_agent': request.user_agent.string,
                'request_id': g.request_id,
                'response_time': getattr(g, 'request_time', 0)
            }
            
            # Add user ID if authenticated
            from flask_jwt_extended import get_jwt_identity
            try:
                user_id = get_jwt_identity()
                if user_id:
                    log_data['user_id'] = user_id
            except:
                pass
            
            app.logger.info(f"Request: {json.dumps(log_data)}")
        
        return response
    
    # Add request timing
    @app.before_request
    def start_timer():
        g.start_time = time.time()
    
    @app.after_request
    def log_request_time(response):
        if hasattr(g, 'start_time'):
            g.request_time = time.time() - g.start_time
        return response
