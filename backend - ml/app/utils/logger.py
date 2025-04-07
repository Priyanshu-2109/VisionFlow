import logging
import json
from datetime import datetime
from flask import request, g

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add request context if available
        if hasattr(g, 'request_id'):
            log_data['request_id'] = g.request_id
            
        if request:
            log_data['request'] = {
                'method': request.method,
                'path': request.path,
                'ip': request.remote_addr
            }
            
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
            
        return json.dumps(log_data)

def setup_logger(app):
    """Configure logging for the application"""
    
    # Create logger
    logger = logging.getLogger('app')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    # Add request logging middleware
    @app.before_request
    def log_request_info():
        g.request_id = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
        logger.info('Request received', extra={
            'request_id': g.request_id,
            'method': request.method,
            'path': request.path,
            'ip': request.remote_addr
        })
    
    @app.after_request
    def log_response_info(response):
        logger.info('Response sent', extra={
            'request_id': g.request_id,
            'status_code': response.status_code
        })
        return response
    
    return logger 