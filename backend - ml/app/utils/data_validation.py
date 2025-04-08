import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Utility class for validating data inputs
    """
    
    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate a pandas DataFrame for basic quality and usability
        
        Args:
            df: The DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                return False, "Dataset is empty"
            
            # Check minimum size
            if len(df) < 2:
                return False, "Dataset must contain at least 2 rows"
            
            if len(df.columns) < 1:
                return False, "Dataset must contain at least 1 column"
            
            # Check for duplicate column names
            if len(df.columns) != len(set(df.columns)):
                return False, "Dataset contains duplicate column names"
            
            # Check for completely empty columns
            empty_columns = [col for col in df.columns if df[col].isna().all()]
            if empty_columns:
                return False, f"Dataset contains completely empty columns: {', '.join(empty_columns)}"
            
            # Check for excessive missing values (>95% missing)
            high_missing_cols = [col for col in df.columns if df[col].isna().mean() > 0.95]
            if high_missing_cols:
                logger.warning(f"Columns with >95% missing values: {', '.join(high_missing_cols)}")
            
            # Check for columns with only one unique value
            single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
            if single_value_cols:
                logger.warning(f"Columns with only one unique value: {', '.join(single_value_cols)}")
            
            return True, None
            
        except Exception as e:
            return False, f"Error validating dataset: {str(e)}"
    
    @staticmethod
    def validate_json_data(data: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate JSON data against a schema
        
        Args:
            data: The data to validate
            schema: Schema definition
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if data is None
            if data is None:
                return False, "Data is None"
            
            # Check if schema is a dictionary
            if not isinstance(schema, dict):
                return False, "Schema must be a dictionary"
            
            # Check required fields
            for field, field_schema in schema.items():
                if field_schema.get('required', False) and field not in data:
                    return False, f"Required field '{field}' is missing"
            
            # Validate field types and values
            for field, value in data.items():
                if field not in schema:
                    continue  # Skip validation for fields not in schema
                
                field_schema = schema[field]
                field_type = field_schema.get('type')
                
                # Type validation
                if field_type:
                    if field_type == 'string' and not isinstance(value, str):
                        return False, f"Field '{field}' must be a string"
                    elif field_type == 'integer' and not isinstance(value, int):
                        return False, f"Field '{field}' must be an integer"
                    elif field_type == 'number' and not isinstance(value, (int, float)):
                        return False, f"Field '{field}' must be a number"
                    elif field_type == 'boolean' and not isinstance(value, bool):
                        return False, f"Field '{field}' must be a boolean"
                    elif field_type == 'array' and not isinstance(value, list):
                        return False, f"Field '{field}' must be an array"
                    elif field_type == 'object' and not isinstance(value, dict):
                        return False, f"Field '{field}' must be an object"
                
                # Range validation for numbers
                if field_type in ['integer', 'number'] and isinstance(value, (int, float)):
                    if 'minimum' in field_schema and value < field_schema['minimum']:
                        return False, f"Field '{field}' must be >= {field_schema['minimum']}"
                    if 'maximum' in field_schema and value > field_schema['maximum']:
                        return False, f"Field '{field}' must be <= {field_schema['maximum']}"
                
                # Length validation for strings
                if field_type == 'string' and isinstance(value, str):
                    if 'minLength' in field_schema and len(value) < field_schema['minLength']:
                        return False, f"Field '{field}' must have at least {field_schema['minLength']} characters"
                    if 'maxLength' in field_schema and len(value) > field_schema['maxLength']:
                        return False, f"Field '{field}' must have at most {field_schema['maxLength']} characters"
                    if 'pattern' in field_schema and not re.match(field_schema['pattern'], value):
                        return False, f"Field '{field}' does not match required pattern"
                
                # Enumeration validation
                if 'enum' in field_schema and value not in field_schema['enum']:
                    return False, f"Field '{field}' must be one of: {', '.join(map(str, field_schema['enum']))}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error validating JSON data: {str(e)}"
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
        """
        Validate password strength
        
        Args:
            password: The password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'[0-9]', password):
            return False, "Password must contain at least one digit"
        
        return True, None
    
    @staticmethod
    def validate_date_format(date_str: str, format: str = '%Y-%m-%d') -> bool:
        """Validate date string format"""
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_value: Optional[Union[int, float]] = None, 
                             max_value: Optional[Union[int, float]] = None) -> bool:
        """Validate numeric value within range"""
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
        """Validate file extension"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    @staticmethod
    def validate_api_key(api_key: str, service: str) -> bool:
        """
        Validate API key format for different services
        
        Args:
            api_key: The API key to validate
            service: The service name (e.g., 'openai', 'news_api')
            
        Returns:
            Boolean indicating if the key format is valid
        """
        if not api_key:
            return False
            
        if service.lower() == 'openai':
            # OpenAI API keys start with 'sk-' and are 51 characters long
            return api_key.startswith('sk-') and len(api_key) == 51
        elif service.lower() == 'news_api':
            # News API keys are 32 characters long
            return len(api_key) == 32
        elif service.lower() == 'twitter':
            # Twitter API keys are 25 characters long
            return len(api_key) == 25
        else:
            # Generic validation - just check if it's not empty
            return len(api_key) > 0