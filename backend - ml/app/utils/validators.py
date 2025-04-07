from functools import wraps
from flask import request, jsonify
from marshmallow import Schema, ValidationError, fields, validate
from typing import Type, Dict, Any, Optional, List
import re
import os
from werkzeug.utils import secure_filename

# Allowed file types
ALLOWED_EXTENSIONS = {
    'images': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
    'documents': {'pdf', 'doc', 'docx', 'xls', 'xlsx', 'txt', 'csv'},
    'videos': {'mp4', 'avi', 'mov', 'wmv'},
    'audio': {'mp3', 'wav', 'ogg', 'm4a'}
}

# Maximum file sizes in bytes
MAX_FILE_SIZES = {
    'images': 5 * 1024 * 1024,  # 5MB
    'documents': 10 * 1024 * 1024,  # 10MB
    'videos': 100 * 1024 * 1024,  # 100MB
    'audio': 20 * 1024 * 1024  # 20MB
}

def validate_file_type(file, allowed_types: List[str] = None) -> bool:
    """
    Validate file type
    
    Args:
        file: File object from request
        allowed_types: List of allowed file types (e.g., ['images', 'documents'])
        
    Returns:
        bool: True if file type is valid
        
    Raises:
        ValidationError: If file type is not allowed
    """
    if not file:
        raise ValidationError("No file provided")
        
    filename = secure_filename(file.filename)
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if allowed_types is None:
        allowed_types = list(ALLOWED_EXTENSIONS.keys())
    
    allowed_extensions = set()
    for file_type in allowed_types:
        if file_type in ALLOWED_EXTENSIONS:
            allowed_extensions.update(ALLOWED_EXTENSIONS[file_type])
    
    if extension not in allowed_extensions:
        raise ValidationError(f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}")
    
    return True

def validate_file_size(file, max_size: int = None) -> bool:
    """
    Validate file size
    
    Args:
        file: File object from request
        max_size: Maximum allowed file size in bytes
        
    Returns:
        bool: True if file size is valid
        
    Raises:
        ValidationError: If file size exceeds limit
    """
    if not file:
        raise ValidationError("No file provided")
    
    if max_size is None:
        # Determine file type from extension
        filename = secure_filename(file.filename)
        extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Find the file type category
        for file_type, extensions in ALLOWED_EXTENSIONS.items():
            if extension in extensions:
                max_size = MAX_FILE_SIZES.get(file_type, 5 * 1024 * 1024)  # Default to 5MB
                break
        else:
            max_size = 5 * 1024 * 1024  # Default to 5MB
    
    if file.content_length > max_size:
        size_mb = max_size / (1024 * 1024)
        raise ValidationError(f"File size exceeds {size_mb}MB limit")
    
    return True

def validate_request(schema: Type[Schema]) -> Dict[str, Any]:
    """
    Decorator to validate request data against a schema.
    
    Args:
        schema: Marshmallow schema class to validate against
        
    Returns:
        Dict containing validated data
        
    Raises:
        ValidationError: If validation fails
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Validate query parameters
                query_data = schema().load(request.args)
                
                # Validate request body for POST/PUT requests
                if request.is_json:
                    body_data = schema().load(request.get_json())
                    data = {**query_data, **body_data}
                else:
                    data = query_data
                    
                return f(data, *args, **kwargs)
                
            except ValidationError as e:
                return jsonify({
                    "message": "Validation Error",
                    "errors": e.messages
                }), 400
                
        return decorated_function
    return decorator

def validate_pagination():
    """
    Decorator to validate pagination parameters.
    
    Returns:
        Dict containing validated pagination parameters
        
    Raises:
        ValidationError: If validation fails
    """
    class PaginationSchema(Schema):
        page = fields.Integer(missing=1, validate=validate.Range(min=1))
        per_page = fields.Integer(missing=10, validate=validate.Range(min=1, max=100))
        sort_by = fields.String(missing='created_at')
        sort_order = fields.String(missing='desc', validate=validate.OneOf(['asc', 'desc']))
        
    return validate_request(PaginationSchema)

def validate_search():
    """
    Decorator to validate search parameters.
    
    Returns:
        Dict containing validated search parameters
        
    Raises:
        ValidationError: If validation fails
    """
    class SearchSchema(Schema):
        query = fields.String(required=True, validate=validate.Length(min=2))
        fields = fields.List(fields.String(), missing=[])
        exact_match = fields.Boolean(missing=False)
        
    return validate_request(SearchSchema)

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError("Invalid email format")
    return True

def validate_password(password: str, min_length: int = 8) -> bool:
    """Validate password strength"""
    if len(password) < min_length:
        raise ValidationError(f"Password must be at least {min_length} characters long")
    
    if not re.search(r'[A-Z]', password):
        raise ValidationError("Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        raise ValidationError("Password must contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        raise ValidationError("Password must contain at least one number")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        raise ValidationError("Password must contain at least one special character")
    
    return True

def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    pattern = r'^\+?1?\d{9,15}$'
    if not re.match(pattern, phone):
        raise ValidationError("Invalid phone number format")
    return True

def validate_url(url: str) -> bool:
    """Validate URL format"""
    pattern = r'^https?://(?:[\w-]|(?=%[\da-fA-F]{2}))+[^\s]*$'
    if not re.match(pattern, url):
        raise ValidationError("Invalid URL format")
    return True

def validate_date(date_str: str, format: str = '%Y-%m-%d') -> bool:
    """Validate date format"""
    try:
        from datetime import datetime
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        raise ValidationError(f"Invalid date format. Expected format: {format}")

def validate_required(value: any, field_name: str) -> bool:
    """Validate required field"""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationError(f"{field_name} is required")
    return True

def validate_length(value: str, min_length: Optional[int] = None, max_length: Optional[int] = None, field_name: str = "Value") -> bool:
    """Validate string length"""
    if min_length is not None and len(value) < min_length:
        raise ValidationError(f"{field_name} must be at least {min_length} characters long")
    
    if max_length is not None and len(value) > max_length:
        raise ValidationError(f"{field_name} must not exceed {max_length} characters")
    
    return True

def validate_numeric(value: any, field_name: str = "Value") -> bool:
    """Validate numeric value"""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name} must be numeric")

def validate_range(value: float, min_value: Optional[float] = None, max_value: Optional[float] = None, field_name: str = "Value") -> bool:
    """Validate numeric range"""
    if min_value is not None and value < min_value:
        raise ValidationError(f"{field_name} must be greater than or equal to {min_value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"{field_name} must be less than or equal to {max_value}")
    
    return True 