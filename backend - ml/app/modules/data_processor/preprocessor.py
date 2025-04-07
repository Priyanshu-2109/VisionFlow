# app/modules/data_processor/preprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing module for business datasets
    """
    
    def __init__(self):
        self.dataset = None
        self.original_dataset = None
        self.column_types = {}
        self.preprocessing_steps = []
        self.numerical_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.text_columns = []
        self.target_column = None
        self.preprocessing_pipeline = None
        
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset from various file formats"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                self.dataset = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.dataset = pd.read_excel(file_path)
            elif file_extension == '.json':
                self.dataset = pd.read_json(file_path)
            elif file_extension == '.xml':
                self.dataset = pd.read_xml(file_path)
            elif file_extension == '.parquet':
                self.dataset = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            self.original_dataset = self.dataset.copy()
            logger.info(f"Successfully loaded dataset with shape: {self.dataset.shape}")
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def infer_column_types(self) -> Dict[str, str]:
        """Automatically infer column types from the dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
            
        self.column_types = {}
        
        for column in self.dataset.columns:
            if pd.api.types.is_numeric_dtype(self.dataset[column]):
                if self.dataset[column].nunique() < 10 and self.dataset[column].nunique() / len(self.dataset[column]) < 0.05:
                    self.column_types[column] = 'categorical'
                    self.categorical_columns.append(column)
                else:
                    self.column_types[column] = 'numerical'
                    self.numerical_columns.append(column)
            elif pd.api.types.is_datetime64_dtype(self.dataset[column]):
                self.column_types[column] = 'datetime'
                self.datetime_columns.append(column)
            elif self.dataset[column].nunique() < 10 or (self.dataset[column].nunique() / len(self.dataset[column]) < 0.05):
                self.column_types[column] = 'categorical'
                self.categorical_columns.append(column)
            else:
                # Check if it might be a datetime
                try:
                    pd.to_datetime(self.dataset[column])
                    self.column_types[column] = 'datetime'
                    self.datetime_columns.append(column)
                except:
                    # If it has a lot of text, classify as text
                    if self.dataset[column].astype(str).str.len().mean() > 50:
                        self.column_types[column] = 'text'
                        self.text_columns.append(column)
                    else:
                        self.column_types[column] = 'categorical'
                        self.categorical_columns.append(column)
        
        self.preprocessing_steps.append({
            'step': 'infer_column_types',
            'result': self.column_types
        })
        
        return self.column_types
    
    def handle_missing_values(self, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """Handle missing values with various strategies"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
            
        if not self.column_types:
            self.infer_column_types()
            
        # Default strategies
        if strategy is None:
            strategy = {
                'numerical': 'mean',
                'categorical': 'most_frequent',
                'datetime': 'drop',
                'text': 'empty_string'
            }
        
        missing_stats_before = self.dataset.isna().sum().to_dict()
        
        # Handle numerical columns
        for col in self.numerical_columns:
            if strategy.get('numerical') == 'mean':
                self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mean())
            elif strategy.get('numerical') == 'median':
                self.dataset[col] = self.dataset[col].fillna(self.dataset[col].median())
            elif strategy.get('numerical') == 'zero':
                self.dataset[col] = self.dataset[col].fillna(0)
                
        # Handle categorical columns
        for col in self.categorical_columns:
            if strategy.get('categorical') == 'most_frequent':
                self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mode()[0] if not self.dataset[col].mode().empty else "Unknown")
            elif strategy.get('categorical') == 'new_category':
                self.dataset[col] = self.dataset[col].fillna("Missing")
                
        # Handle datetime columns
        for col in self.datetime_columns:
            if strategy.get('datetime') == 'drop':
                self.dataset = self.dataset.dropna(subset=[col])
            elif strategy.get('datetime') == 'mode':
                self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mode()[0] if not self.dataset[col].mode().empty else pd.NaT)
                
        # Handle text columns
        for col in self.text_columns:
            if strategy.get('text') == 'empty_string':
                self.dataset[col] = self.dataset[col].fillna("")
                
        missing_stats_after = self.dataset.isna().sum().to_dict()
        
        self.preprocessing_steps.append({
            'step': 'handle_missing_values',
            'strategy': strategy,
            'missing_before': missing_stats_before,
            'missing_after': missing_stats_after
        })
        
        return self.dataset
    
    def handle_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Detect and handle outliers in numerical columns"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
            
        if not self.column_types:
            self.infer_column_types()
            
        outliers_stats = {}
        
        for col in self.numerical_columns:
            outliers_mask = np.zeros(len(self.dataset), dtype=bool)
            
            if method == 'iqr':
                Q1 = self.dataset[col].quantile(0.25)
                Q3 = self.dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers_mask = (self.dataset[col] < lower_bound) | (self.dataset[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = (self.dataset[col] - self.dataset[col].mean()) / self.dataset[col].std()
                outliers_mask = abs(z_scores) > threshold
                
            # Record outlier stats
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                outliers_stats[col] = {
                    'count': int(outliers_count),
                    'percentage': float(outliers_count / len(self.dataset) * 100)
                }
                
                # Replace outliers with boundary values
                self.dataset.loc[self.dataset[col] < lower_bound, col] = lower_bound if method == 'iqr' else None
                self.dataset.loc[self.dataset[col] > upper_bound, col] = upper_bound if method == 'iqr' else None
        
        self.preprocessing_steps.append({
            'step': 'handle_outliers',
            'method': method,
            'threshold': threshold,
            'outliers_detected': outliers_stats
        })
        
        return self.dataset
    
    def encode_categorical_features(self, method: str = 'onehot', max_categories: int = 10) -> pd.DataFrame:
        """Encode categorical features using various methods"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
            
        if not self.column_types:
            self.infer_column_types()
            
        encoding_stats = {}
        
        for col in self.categorical_columns:
            n_categories = self.dataset[col].nunique()
            encoding_stats[col] = {'original_categories': n_categories}
            
            if method == 'onehot' and n_categories <= max_categories:
                # One-hot encoding
                dummies = pd.get_dummies(self.dataset[col], prefix=col, drop_first=False)
                self.dataset = pd.concat([self.dataset, dummies], axis=1)
                self.dataset = self.dataset.drop(col, axis=1)
                encoding_stats[col]['method'] = 'onehot'
                encoding_stats[col]['new_columns'] = dummies.columns.tolist()
                
            elif method == 'label' or (method == 'onehot' and n_categories > max_categories):
                # Label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                self.dataset[col] = le.fit_transform(self.dataset[col].astype(str))
                encoding_stats[col]['method'] = 'label'
                encoding_stats[col]['mapping'] = dict(zip(le.classes_, le.transform(le.classes_)))
                
        self.preprocessing_steps.append({
            'step': 'encode_categorical_features',
            'method': method,
            'max_categories': max_categories,
            'encoding_stats': encoding_stats
        })
        
        return self.dataset
    
    def normalize_numerical_features(self, method: str = 'minmax') -> pd.DataFrame:
        """Normalize numerical features using various methods"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
            
        if not self.column_types:
            self.infer_column_types()
            
        normalization_stats = {}
        
        if method == 'minmax':
            scaler = MinMaxScaler()
            if self.numerical_columns:
                self.dataset[self.numerical_columns] = scaler.fit_transform(self.dataset[self.numerical_columns])
                normalization_stats['min'] = scaler.data_min_.tolist()
                normalization_stats['max'] = scaler.data_max_.tolist()
                
        elif method == 'standard':
            scaler = StandardScaler()
            if self.numerical_columns:
                self.dataset[self.numerical_columns] = scaler.fit_transform(self.dataset[self.numerical_columns])
                normalization_stats['mean'] = scaler.mean_.tolist()
                normalization_stats['std'] = scaler.scale_.tolist()
                
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            if self.numerical_columns:
                self.dataset[self.numerical_columns] = scaler.fit_transform(self.dataset[self.numerical_columns])
                normalization_stats['center'] = scaler.center_.tolist()
                normalization_stats['scale'] = scaler.scale_.tolist()
                
        self.preprocessing_steps.append({
            'step': 'normalize_numerical_features',
            'method': method,
            'columns': self.numerical_columns,
            'stats': normalization_stats
        })
        
        return self.dataset
    
    def process_datetime_features(self) -> pd.DataFrame:
        """Extract useful features from datetime columns"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
            
        if not self.column_types:
            self.infer_column_types()
            
        datetime_processing_stats = {}
        
        for col in self.datetime_columns:
            # Ensure column is datetime type
            try:
                self.dataset[col] = pd.to_datetime(self.dataset[col])
                
                # Extract useful components
                self.dataset[f'{col}_year'] = self.dataset[col].dt.year
                self.dataset[f'{col}_month'] = self.dataset[col].dt.month
                self.dataset[f'{col}_day'] = self.dataset[col].dt.day
                self.dataset[f'{col}_weekday'] = self.dataset[col].dt.weekday
                self.dataset[f'{col}_quarter'] = self.dataset[col].dt.quarter
                
                # Add to numerical columns
                self.numerical_columns.extend([
                    f'{col}_year', f'{col}_month', f'{col}_day', 
                    f'{col}_weekday', f'{col}_quarter'
                ])
                
                datetime_processing_stats[col] = {
                    'new_features': [
                        f'{col}_year', f'{col}_month', f'{col}_day', 
                        f'{col}_weekday', f'{col}_quarter'
                    ]
                }
                
            except Exception as e:
                logger.warning(f"Could not process datetime column {col}: {str(e)}")
                datetime_processing_stats[col] = {'error': str(e)}
                
        self.preprocessing_steps.append({
            'step': 'process_datetime_features',
            'stats': datetime_processing_stats
        })
        
        return self.dataset
    
    def create_preprocessing_pipeline(self) -> None:
        """Create a scikit-learn pipeline for preprocessing"""
        if not self.column_types:
            self.infer_column_types()
            
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_columns),
                ('cat', categorical_transformer, self.categorical_columns)
            ],
            remainder='passthrough'
        )
        
        self.preprocessing_pipeline = preprocessor
        
        self.preprocessing_steps.append({
            'step': 'create_preprocessing_pipeline',
            'pipeline_components': {
                'numerical_transformer': 'SimpleImputer + StandardScaler',
                'categorical_transformer': 'SimpleImputer + OneHotEncoder'
            }
        })
    
    def save_pipeline(self, path: str) -> None:
        """Save the preprocessing pipeline for future use"""
        if self.preprocessing_pipeline is None:
            self.create_preprocessing_pipeline()
            
        joblib.dump(self.preprocessing_pipeline, path)
        logger.info(f"Preprocessing pipeline saved to {path}")
        
        self.preprocessing_steps.append({
            'step': 'save_pipeline',
            'path': path
        })
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
            
        numerical_summary = {}
        for col in self.numerical_columns:
            numerical_summary[col] = {
                'mean': float(self.dataset[col].mean()),
                'median': float(self.dataset[col].median()),
                'std': float(self.dataset[col].std()),
                'min': float(self.dataset[col].min()),
                'max': float(self.dataset[col].max()),
                'missing': int(self.dataset[col].isna().sum())
            }
            
        categorical_summary = {}
        for col in self.categorical_columns:
            categorical_summary[col] = {
                'unique_values': int(self.dataset[col].nunique()),
                'top_categories': self.dataset[col].value_counts().head(5).to_dict(),
                'missing': int(self.dataset[col].isna().sum())
            }
            
        return {
            'shape': self.dataset.shape,
            'columns': list(self.dataset.columns),
            'column_types': self.column_types,
            'numerical_summary': numerical_summary,
            'categorical_summary': categorical_summary,
            'preprocessing_steps': self.preprocessing_steps
        }
    
    def auto_preprocess(self) -> pd.DataFrame:
        """Perform automatic preprocessing with sensible defaults"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
            
        # Step 1: Infer column types
        self.infer_column_types()
        
        # Step 2: Handle missing values
        self.handle_missing_values()
        
        # Step 3: Process datetime features
        self.process_datetime_features()
        
        # Step 4: Handle outliers
        self.handle_outliers()
        
        # Step 5: Encode categorical features
        self.encode_categorical_features()
        
        # Step 6: Normalize numerical features
        self.normalize_numerical_features()
        
        # Step 7: Create preprocessing pipeline
        self.create_preprocessing_pipeline()
        
        return self.dataset