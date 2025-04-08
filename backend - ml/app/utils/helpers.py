import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import json
import uuid
import os
import re
import io
import base64
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import hashlib
import requests
from flask import current_app, g
import time

logger = logging.getLogger(__name__)

class DataHelper:
    """
    Helper utilities for data manipulation and conversion
    """
    
    @staticmethod
    def dataframe_to_json(df: pd.DataFrame, orient: str = 'records') -> str:
        """Convert DataFrame to JSON string"""
        return df.to_json(orient=orient, date_format='iso')
    
    @staticmethod
    def json_to_dataframe(json_str: str) -> pd.DataFrame:
        """Convert JSON string to DataFrame"""
        try:
            return pd.read_json(json_str)
        except Exception as e:
            logger.error(f"Error converting JSON to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Dict[str, str]:
        """Get column types from DataFrame"""
        column_types = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < 10 and df[col].nunique() / len(df[col]) < 0.05:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'numerical'
            elif pd.api.types.is_datetime64_dtype(df[col]):
                column_types[col] = 'datetime'
            elif df[col].nunique() < 10 or (df[col].nunique() / len(df[col]) < 0.05):
                column_types[col] = 'categorical'
            else:
                # Check if it might be a datetime
                try:
                    pd.to_datetime(df[col])
                    column_types[col] = 'datetime'
                except:
                    # If it has a lot of text, classify as text
                    if df[col].astype(str).str.len().mean() > 50:
                        column_types[col] = 'text'
                    else:
                        column_types[col] = 'categorical'
        
        return column_types
    
    @staticmethod
    def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive summary statistics for a dataset"""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'column_types': DataHelper.get_column_types(df),
            'missing_values': df.isna().sum().to_dict(),
            'numerical_summary': {},
            'categorical_summary': {}
        }
        
        # Get column types
        column_types = DataHelper.get_column_types(df)
        
        # Numerical summary
        for col, col_type in column_types.items():
            if col_type == 'numerical':
                summary['numerical_summary'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q1': float(df[col].quantile(0.25)),
                    'q3': float(df[col].quantile(0.75)),
                    'missing': int(df[col].isna().sum()),
                    'missing_pct': float(df[col].isna().mean() * 100)
                }
        
        # Categorical summary
        for col, col_type in column_types.items():
            if col_type == 'categorical':
                value_counts = df[col].value_counts().head(10).to_dict()
                summary['categorical_summary'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in value_counts.items()},
                    'missing': int(df[col].isna().sum()),
                    'missing_pct': float(df[col].isna().mean() * 100)
                }
        
        return summary
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers in a DataFrame column
        
        Args:
            df: DataFrame
            column: Column name
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean Series where True indicates outlier
        """
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
            
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column {column} is not numeric")
            
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            return (df[column] < lower_bound) | (df[column] > upper_bound)
            
        elif method == 'zscore':
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            return abs(z_scores) > threshold
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    @staticmethod
    def plot_to_base64(plt_figure) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        plt_figure.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(plt_figure)
        return img_str
    
    @staticmethod
    def sample_dataframe(df: pd.DataFrame, n: int = 1000, method: str = 'random') -> pd.DataFrame:
        """
        Sample a DataFrame for visualization or analysis
        
        Args:
            df: DataFrame to sample
            n: Number of samples
            method: Sampling method ('random', 'systematic', 'stratified')
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= n:
            return df
            
        if method == 'random':
            return df.sample(n=n, random_state=42)
            
        elif method == 'systematic':
            step = len(df) // n
            return df.iloc[::step].head(n)
            
        elif method == 'stratified':
            # Try to find a categorical column for stratification
            cat_cols = [col for col, col_type in DataHelper.get_column_types(df).items() 
                       if col_type == 'categorical']
            
            if cat_cols:
                # Use the first categorical column with reasonable number of categories
                strat_col = None
                for col in cat_cols:
                    if df[col].nunique() <= 10:
                        strat_col = col
                        break
                
                if strat_col:
                    return df.groupby(strat_col, group_keys=False).apply(
                        lambda x: x.sample(min(len(x), n // df[strat_col].nunique() + 1))
                    ).head(n)
            
            # Fallback to random sampling
            return df.sample(n=n, random_state=42)
            
        else:
            raise ValueError(f"Unknown sampling method: {method}")

class FileHelper:
    """
    Helper utilities for file operations
    """
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension from filename"""
        return os.path.splitext(filename)[1].lower().lstrip('.')
    
    @staticmethod
    def generate_unique_filename(original_filename: str) -> str:
        """Generate a unique filename preserving the original extension"""
        extension = FileHelper.get_file_extension(original_filename)
        return f"{uuid.uuid4()}.{extension}"
    
    @staticmethod
    def save_uploaded_file(file, upload_folder: str, filename: Optional[str] = None) -> str:
        """
        Save an uploaded file with a unique name
        
        Args:
            file: File object from request.files
            upload_folder: Directory to save the file
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        if filename is None:
            filename = FileHelper.generate_unique_filename(file.filename)
            
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        return file_path
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes"""
        return os.path.getsize(file_path)
    
    @staticmethod
    def get_file_mime_type(file_path: str) -> str:
        """Get file MIME type"""
        import magic
        return magic.from_file(file_path, mime=True)
    
    @staticmethod
    def is_valid_csv(file_path: str) -> bool:
        """Check if file is a valid CSV"""
        try:
            pd.read_csv(file_path, nrows=5)
            return True
        except Exception:
            return False
    
    @staticmethod
    def is_valid_excel(file_path: str) -> bool:
        """Check if file is a valid Excel file"""
        try:
            pd.read_excel(file_path, nrows=5)
            return True
        except Exception:
            return False

class PerformanceMonitor:
    """
    Helper utilities for monitoring and performance
    """
    
    @staticmethod
    def timeit(func: Callable) -> Callable:
        """Decorator to measure function execution time"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
            
            return result
        return wrapper
    
    @staticmethod
    def memory_usage() -> Dict[str, float]:
        """Get current memory usage"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
            'vms': memory_info.vms / (1024 * 1024)   # VMS in MB
        }
    
    @staticmethod
    def log_request_stats() -> None:
        """Log request statistics (to be used in Flask after_request)"""
        if hasattr(g, 'start_time'):
            execution_time = time.time() - g.start_time
            memory = PerformanceMonitor.memory_usage()
            
            logger.info(
                f"Request processed in {execution_time:.4f}s | "
                f"Memory: RSS={memory['rss']:.2f}MB, VMS={memory['vms']:.2f}MB"
            )
    
    @staticmethod
    def calculate_eta(current: int, total: int, elapsed_time: float) -> float:
        """
        Calculate estimated time remaining
        
        Args:
            current: Current progress
            total: Total work to be done
            elapsed_time: Time elapsed so far in seconds
            
        Returns:
            Estimated time remaining in seconds
        """
        if current == 0:
            return float('inf')
            
        progress_fraction = current / total
        estimated_total_time = elapsed_time / progress_fraction
        return estimated_total_time - elapsed_time

class TextHelper:
    """
    Helper utilities for text processing
    """
    
    @staticmethod
    def clean_html(html_text: str) -> str:
        """Remove HTML tags from text"""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', html_text)
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """Extract key terms from text"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            
            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text.lower())
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
            
            # If text is too short, return the filtered words
            if len(filtered_words) < 5:
                return filtered_words[:top_n]
            
            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])
            
            # Get feature names and their TF-IDF scores
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Sort by score and get top terms
            sorted_indices = tfidf_scores.argsort()[::-1]
            top_terms = [feature_names[i] for i in sorted_indices[:top_n]]
            
            return top_terms
            
        except Exception as e:
            logger.warning(f"Error extracting keywords: {str(e)}")
            # Fallback to simple frequency analysis
            from collections import Counter
            words = re.findall(r'\b\w+\b', text.lower())
            return [word for word, _ in Counter(words).most_common(top_n)]
    
    @staticmethod
    def summarize_text(text: str, max_sentences: int = 3) -> str:
        """Generate a simple extractive summary of text"""
        try:
            from nltk.tokenize import sent_tokenize
            from nltk.corpus import stopwords
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # If text is short, return as is
            sentences = sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return text
            
            # Calculate sentence scores using TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate the average TF-IDF score for each sentence
            sentence_scores = [(i, tfidf_matrix[i].mean()) for i in range(len(sentences))]
            
            # Sort by score and get top sentences
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
            
            # Sort by original position to maintain flow
            top_sentences.sort(key=lambda x: x[0])
            
            # Combine sentences
            summary = ' '.join(sentences[i] for i, _ in top_sentences)
            return summary
            
        except Exception as e:
            logger.warning(f"Error summarizing text: {str(e)}")
            # Fallback to first N sentences
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:max_sentences])
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect the language of text"""
        try:
            from langdetect import detect
            return detect(text)
        except Exception as e:
            logger.warning(f"Error detecting language: {str(e)}")
            return 'en'  # Default to English
    
    @staticmethod
    def sentiment_analysis(text: str) -> Dict[str, float]:
        """Perform basic sentiment analysis"""
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
            return sentiment
        except Exception as e:
            logger.warning(f"Error in sentiment analysis: {str(e)}")
            # Return neutral sentiment as fallback
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

class NetworkHelper:
    """
    Helper utilities for network operations
    """
    
    @staticmethod
    def make_request(url: str, method: str = 'GET', params: Dict = None, 
                    headers: Dict = None, data: Any = None, 
                    timeout: int = 30, retries: int = 3) -> Tuple[int, Any]:
        """
        Make HTTP request with retries and error handling
        
        Args:
            url: URL to request
            method: HTTP method
            params: URL parameters
            headers: HTTP headers
            data: Request body
            timeout: Request timeout in seconds
            retries: Number of retries on failure
            
        Returns:
            Tuple of (status_code, response_content)
        """
        attempt = 0
        while attempt < retries:
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    json=data if method in ['POST', 'PUT', 'PATCH'] and isinstance(data, dict) else None,
                    data=data if method in ['POST', 'PUT', 'PATCH'] and not isinstance(data, dict) else None,
                    timeout=timeout
                )
                
                # Try to parse JSON response
                try:
                    content = response.json()
                except:
                    content = response.text
                
                return response.status_code, content
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request to {url} timed out (attempt {attempt+1}/{retries})")
                attempt += 1
                if attempt < retries:
                    time.sleep(1)  # Wait before retrying
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request to {url} failed: {str(e)}")
                return 0, {'error': str(e)}
        
        return 0, {'error': 'Request failed after multiple attempts'}
    
    @staticmethod
    def download_file(url: str, local_path: str, chunk_size: int = 8192) -> bool:
        """
        Download file from URL to local path
        
        Args:
            url: URL to download
            local_path: Local path to save file
            chunk_size: Download chunk size
            
        Returns:
            Boolean indicating success
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file from {url}: {str(e)}")
            return False
    
    @staticmethod
    def get_ip_info(ip_address: str) -> Dict[str, Any]:
        """Get information about an IP address"""
        try:
            response = requests.get(f"https://ipinfo.io/{ip_address}/json", timeout=5)
            return response.json()
        except Exception as e:
            logger.error(f"Error getting IP info: {str(e)}")
            return {}

def format_datetime(dt: datetime, format: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format datetime object to string
    
    Args:
        dt: datetime object to format
        format: strftime format string
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format)

def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename using UUID
    
    Args:
        original_filename: Original filename with extension
        
    Returns:
        Unique filename with original extension
    """
    ext = os.path.splitext(original_filename)[1]
    return f"{uuid.uuid4()}{ext}"

def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    Calculate SHA-256 hash of a file
    
    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read
        
    Returns:
        SHA-256 hash of the file
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (including the dot)
    """
    return os.path.splitext(filename)[1]

def ensure_directory(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path to ensure
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)

def format_file_size(size_bytes: int) -> str:
    """
    Format file size to human readable string
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Human readable file size (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"