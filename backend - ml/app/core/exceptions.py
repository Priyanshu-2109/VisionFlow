class HTTPException(Exception):
    """Base class for HTTP exceptions"""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def to_dict(self):
        """Convert exception to dictionary format"""
        return {
            'error': self.message,
            'status_code': self.status_code
        }

class AppException(HTTPException):
    """Base exception for all application errors"""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)

class ValidationError(AppException):
    """Raised when data validation fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)

class AuthenticationError(AppException):
    """Raised when authentication fails"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)

class AuthorizationError(AppException):
    """Raised when authorization fails"""
    def __init__(self, message: str = "Not authorized"):
        super().__init__(message, status_code=403)

class NotFoundError(AppException):
    """Raised when a resource is not found"""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)

class ConflictError(AppException):
    """Raised when there is a conflict with existing data"""
    def __init__(self, message: str):
        super().__init__(message, status_code=409)

class DatabaseError(AppException):
    """Raised when database operations fail"""
    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message, status_code=500)

class CacheError(AppException):
    """Raised when cache operations fail"""
    def __init__(self, message: str = "Cache operation failed"):
        super().__init__(message, status_code=500)

class ExternalServiceError(AppException):
    """Raised when external service calls fail"""
    def __init__(self, message: str = "External service error"):
        super().__init__(message, status_code=502)

class RateLimitError(AppException):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)

class APIError(Exception):
    """Base class for API errors"""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class BadRequestError(APIError):
    """400 Bad Request"""
    def __init__(self, message: str = "Bad request"):
        super().__init__(message, status_code=400)

class NotFoundError(APIError):
    """404 Not Found"""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)

class ValidationError(APIError):
    """422 Validation Error"""
    def __init__(self, message: str = "Validation error"):
        super().__init__(message, status_code=422)

class DatabaseError(APIError):
    """500 Database Error"""
    def __init__(self, message: str = "Database error"):
        super().__init__(message, status_code=500)

class AuthenticationError(APIError):
    """401 Authentication Error"""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status_code=401)

class AuthorizationError(APIError):
    """403 Authorization Error"""
    def __init__(self, message: str = "Not authorized"):
        super().__init__(message, status_code=403)

class RateLimitError(APIError):
    """429 Rate Limit Error"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)

class ServiceUnavailableError(APIError):
    """503 Service Unavailable"""
    def __init__(self, message: str = "Service unavailable"):
        super().__init__(message, status_code=503)

def handle_http_exception(error):
    """Handle HTTP exceptions and return appropriate response"""
    response = error.to_dict()
    response['status'] = 'error'
    return response, error.status_code

def handle_validation_error(error):
    """Handle validation errors and return appropriate response"""
    response = {
        'status': 'error',
        'error': error.message,
        'status_code': error.status_code
    }
    return response, error.status_code

def handle_authentication_error(error):
    """Handle authentication errors and return appropriate response"""
    response = {
        'status': 'error',
        'error': error.message,
        'status_code': error.status_code
    }
    return response, error.status_code

def handle_authorization_error(error):
    """Handle authorization errors and return appropriate response"""
    response = {
        'status': 'error',
        'error': error.message,
        'status_code': error.status_code
    }
    return response, error.status_code 