# app/modules/data_processor/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for business datasets
    """
    
    def __init__(self, dataset: pd.DataFrame = None):
        self.dataset = dataset
        self.numerical_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.engineering_steps = []
        
    def set_dataset(self, dataset: pd.DataFrame) -> None:
        """Set the dataset to work with"""
        self.dataset = dataset
        
    def set_column_types(self, numerical: List[str], categorical: List[str], datetime: List[str]) -> None:
        """Set column types manually"""
        self.numerical_columns = numerical
        self.categorical_columns = categorical
        self.datetime_columns = datetime
        
    def infer_column_types(self) -> Dict[str, List[str]]:
        """Infer column types from dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        self.numerical_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
        for col in self.dataset.columns:
            if pd.api.types.is_numeric_dtype(self.dataset[col]):
                if self.dataset[col].nunique() < 10 and self.dataset[col].nunique() / len(self.dataset[col]) < 0.05:
                    self.categorical_columns.append(col)
                else:
                    self.numerical_columns.append(col)
            elif pd.api.types.is_datetime64_dtype(self.dataset[col]):
                self.datetime_columns.append(col)
            elif self.dataset[col].nunique() < 10 or (self.dataset[col].nunique() / len(self.dataset[col]) < 0.05):
                self.categorical_columns.append(col)
            else:
                # Check if it might be a datetime
                try:
                    pd.to_datetime(self.dataset[col])
                    self.datetime_columns.append(col)
                except:
                    self.categorical_columns.append(col)
        
        return {
            'numerical': self.numerical_columns,
            'categorical': self.categorical_columns,
            'datetime': self.datetime_columns
        }
    def create_interaction_features(self, interaction_pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """Create interaction features between numerical columns"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if not self.numerical_columns:
            self.infer_column_types()
            
        # If no specific pairs provided, create interactions between all numerical columns
        if interaction_pairs is None:
            interaction_pairs = []
            for i, col1 in enumerate(self.numerical_columns):
                for col2 in self.numerical_columns[i+1:]:
                    interaction_pairs.append((col1, col2))
        
        interactions_created = []
        
        for col1, col2 in interaction_pairs:
            if col1 in self.dataset.columns and col2 in self.dataset.columns:
                # Multiplication interaction
                interaction_name = f"{col1}_x_{col2}"
                self.dataset[interaction_name] = self.dataset[col1] * self.dataset[col2]
                interactions_created.append({"type": "multiplication", "name": interaction_name, "columns": [col1, col2]})
                
                # Division interactions (with error handling)
                if (self.dataset[col2] != 0).all():
                    div_name = f"{col1}_div_{col2}"
                    self.dataset[div_name] = self.dataset[col1] / self.dataset[col2]
                    interactions_created.append({"type": "division", "name": div_name, "columns": [col1, col2]})
                
                if (self.dataset[col1] != 0).all():
                    div_name = f"{col2}_div_{col1}"
                    self.dataset[div_name] = self.dataset[col2] / self.dataset[col1]
                    interactions_created.append({"type": "division", "name": div_name, "columns": [col2, col1]})
                    
                # Addition
                add_name = f"{col1}_plus_{col2}"
                self.dataset[add_name] = self.dataset[col1] + self.dataset[col2]
                interactions_created.append({"type": "addition", "name": add_name, "columns": [col1, col2]})
                
                # Subtraction
                sub_name = f"{col1}_minus_{col2}"
                self.dataset[sub_name] = self.dataset[col1] - self.dataset[col2]
                interactions_created.append({"type": "subtraction", "name": sub_name, "columns": [col1, col2]})
        
        self.engineering_steps.append({
            'step': 'create_interaction_features',
            'interactions_created': interactions_created
        })
        
        # Add new columns to numerical columns list
        for interaction in interactions_created:
            self.numerical_columns.append(interaction["name"])
            
        return self.dataset
    
    def create_polynomial_features(self, columns: List[str] = None, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for numerical columns"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if not self.numerical_columns:
            self.infer_column_types()
            
        if columns is None:
            columns = self.numerical_columns
            
        polynomials_created = []
        
        for col in columns:
            if col in self.dataset.columns:
                for d in range(2, degree + 1):
                    poly_name = f"{col}_pow{d}"
                    self.dataset[poly_name] = np.power(self.dataset[col], d)
                    polynomials_created.append({"column": col, "degree": d, "name": poly_name})
                    self.numerical_columns.append(poly_name)
        
        self.engineering_steps.append({
            'step': 'create_polynomial_features',
            'degree': degree,
            'polynomials_created': polynomials_created
        })
        
        return self.dataset
    
    def create_lag_features(self, columns: List[str], time_column: str, lag_periods: List[int]) -> pd.DataFrame:
        """Create lag features for time series data"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
        
        if time_column not in self.dataset.columns:
            raise ValueError(f"Time column {time_column} not found in dataset")
            
        # Ensure the dataset is sorted by time
        self.dataset = self.dataset.sort_values(by=time_column)
        
        lag_features_created = []
        
        for col in columns:
            if col in self.dataset.columns:
                for lag in lag_periods:
                    lag_name = f"{col}_lag{lag}"
                    self.dataset[lag_name] = self.dataset[col].shift(lag)
                    lag_features_created.append({"column": col, "lag": lag, "name": lag_name})
                    self.numerical_columns.append(lag_name)
        
        self.engineering_steps.append({
            'step': 'create_lag_features',
            'time_column': time_column,
            'lag_periods': lag_periods,
            'lag_features_created': lag_features_created
        })
        
        return self.dataset
    
    def create_rolling_features(self, columns: List[str], time_column: str, 
                               windows: List[int], functions: List[str]) -> pd.DataFrame:
        """Create rolling window features for time series data"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
        
        if time_column not in self.dataset.columns:
            raise ValueError(f"Time column {time_column} not found in dataset")
            
        # Ensure the dataset is sorted by time
        self.dataset = self.dataset.sort_values(by=time_column)
        
        rolling_features_created = []
        
        for col in columns:
            if col in self.dataset.columns:
                for window in windows:
                    for func in functions:
                        feature_name = f"{col}_{func}_{window}"
                        
                        if func == 'mean':
                            self.dataset[feature_name] = self.dataset[col].rolling(window=window).mean()
                        elif func == 'std':
                            self.dataset[feature_name] = self.dataset[col].rolling(window=window).std()
                        elif func == 'min':
                            self.dataset[feature_name] = self.dataset[col].rolling(window=window).min()
                        elif func == 'max':
                            self.dataset[feature_name] = self.dataset[col].rolling(window=window).max()
                        elif func == 'sum':
                            self.dataset[feature_name] = self.dataset[col].rolling(window=window).sum()
                        
                        rolling_features_created.append({
                            "column": col, 
                            "window": window, 
                            "function": func,
                            "name": feature_name
                        })
                        
                        self.numerical_columns.append(feature_name)
        
        self.engineering_steps.append({
            'step': 'create_rolling_features',
            'time_column': time_column,
            'windows': windows,
            'functions': functions,
            'rolling_features_created': rolling_features_created
        })
        
        return self.dataset
    
    def create_date_features(self, date_columns: List[str] = None) -> pd.DataFrame:
        """Extract comprehensive date features from datetime columns"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if not self.datetime_columns and not date_columns:
            self.infer_column_types()
            
        if date_columns is None:
            date_columns = self.datetime_columns
            
        date_features_created = []
        
        for col in date_columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(self.dataset[col]):
                try:
                    self.dataset[col] = pd.to_datetime(self.dataset[col])
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {str(e)}")
                    continue
            
            # Basic date components
            self.dataset[f"{col}_year"] = self.dataset[col].dt.year
            self.dataset[f"{col}_month"] = self.dataset[col].dt.month
            self.dataset[f"{col}_day"] = self.dataset[col].dt.day
            self.dataset[f"{col}_weekday"] = self.dataset[col].dt.weekday
            self.dataset[f"{col}_quarter"] = self.dataset[col].dt.quarter
            self.dataset[f"{col}_week"] = self.dataset[col].dt.isocalendar().week
            
            # Add these to numerical columns
            basic_features = [
                f"{col}_year", f"{col}_month", f"{col}_day", 
                f"{col}_weekday", f"{col}_quarter", f"{col}_week"
            ]
            self.numerical_columns.extend(basic_features)
            
            # Is weekend
            self.dataset[f"{col}_is_weekend"] = self.dataset[col].dt.weekday.isin([5, 6]).astype(int)
            self.numerical_columns.append(f"{col}_is_weekend")
            
            # Is month start/end
            self.dataset[f"{col}_is_month_start"] = self.dataset[col].dt.is_month_start.astype(int)
            self.dataset[f"{col}_is_month_end"] = self.dataset[col].dt.is_month_end.astype(int)
            self.numerical_columns.extend([f"{col}_is_month_start", f"{col}_is_month_end"])
            
            # Is quarter start/end
            self.dataset[f"{col}_is_quarter_start"] = self.dataset[col].dt.is_quarter_start.astype(int)
            self.dataset[f"{col}_is_quarter_end"] = self.dataset[col].dt.is_quarter_end.astype(int)
            self.numerical_columns.extend([f"{col}_is_quarter_start", f"{col}_is_quarter_end"])
            
            # Season (Northern Hemisphere)
            def get_season(month):
                if month in [12, 1, 2]:
                    return 0  # Winter
                elif month in [3, 4, 5]:
                    return 1  # Spring
                elif month in [6, 7, 8]:
                    return 2  # Summer
                else:
                    return 3  # Fall
            
            self.dataset[f"{col}_season"] = self.dataset[col].dt.month.apply(get_season)
            self.numerical_columns.append(f"{col}_season")
            
            # Record created features
            date_features_created.append({
                "column": col,
                "features": [
                    f"{col}_year", f"{col}_month", f"{col}_day", 
                    f"{col}_weekday", f"{col}_quarter", f"{col}_week",
                    f"{col}_is_weekend", f"{col}_is_month_start", f"{col}_is_month_end",
                    f"{col}_is_quarter_start", f"{col}_is_quarter_end", f"{col}_season"
                ]
            })
        
        self.engineering_steps.append({
            'step': 'create_date_features',
            'date_features_created': date_features_created
        })
        
        return self.dataset
    
    def create_aggregation_features(self, group_columns: List[str], agg_columns: List[str], 
                                   agg_functions: List[str]) -> pd.DataFrame:
        """Create aggregation features based on categorical columns"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if not (self.numerical_columns and self.categorical_columns):
            self.infer_column_types()
            
        agg_features_created = []
        
        for group_col in group_columns:
            if group_col not in self.dataset.columns:
                continue
                
            for agg_col in agg_columns:
                if agg_col not in self.dataset.columns:
                    continue
                    
                # Calculate aggregations
                agg_dict = {}
                for func in agg_functions:
                    agg_dict[func] = self.dataset.groupby(group_col)[agg_col].transform(func)
                    feature_name = f"{group_col}_{agg_col}_{func}"
                    self.dataset[feature_name] = agg_dict[func]
                    self.numerical_columns.append(feature_name)
                    agg_features_created.append({
                        "group_column": group_col,
                        "agg_column": agg_col,
                        "function": func,
                        "name": feature_name
                    })
        
        self.engineering_steps.append({
            'step': 'create_aggregation_features',
            'group_columns': group_columns,
            'agg_columns': agg_columns,
            'agg_functions': agg_functions,
            'agg_features_created': agg_features_created
        })
        
        return self.dataset
    
    def create_ratio_features(self, numerator_columns: List[str], denominator_columns: List[str]) -> pd.DataFrame:
        """Create ratio features between numerical columns"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if not self.numerical_columns:
            self.infer_column_types()
            
        ratio_features_created = []
        
        for num_col in numerator_columns:
            if num_col not in self.dataset.columns:
                continue
                
            for denom_col in denominator_columns:
                if denom_col not in self.dataset.columns or num_col == denom_col:
                    continue
                    
                # Create ratio feature (with handling for zero division)
                ratio_name = f"{num_col}_div_{denom_col}"
                self.dataset[ratio_name] = self.dataset[num_col] / self.dataset[denom_col].replace(0, np.nan)
                
                # Replace infinities with NaN
                self.dataset[ratio_name].replace([np.inf, -np.inf], np.nan, inplace=True)
                
                self.numerical_columns.append(ratio_name)
                ratio_features_created.append({
                    "numerator": num_col,
                    "denominator": denom_col,
                    "name": ratio_name
                })
        
        self.engineering_steps.append({
            'step': 'create_ratio_features',
            'ratio_features_created': ratio_features_created
        })
        
        return self.dataset
    
    def reduce_dimensions(self, n_components: int = 3, method: str = 'pca') -> pd.DataFrame:
        """Reduce dimensions of numerical features using PCA or other methods"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if not self.numerical_columns:
            self.infer_column_types()
            
        # Select only numerical columns with no missing values
        valid_cols = [col for col in self.numerical_columns 
                     if col in self.dataset.columns and not self.dataset[col].isna().any()]
        
        if len(valid_cols) < 2:
            logger.warning("Not enough valid numerical columns for dimension reduction")
            return self.dataset
            
        X = self.dataset[valid_cols]
        
        if method == 'pca':
            reducer = PCA(n_components=min(n_components, len(valid_cols)))
            reduced_data = reducer.fit_transform(X)
            
            # Add PCA components to dataset
            for i in range(reduced_data.shape[1]):
                component_name = f"pca_component_{i+1}"
                self.dataset[component_name] = reduced_data[:, i]
                self.numerical_columns.append(component_name)
                
            reduction_stats = {
                'method': 'pca',
                'n_components': reduced_data.shape[1],
                'explained_variance_ratio': reducer.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(reducer.explained_variance_ratio_).tolist(),
                'components_added': [f"pca_component_{i+1}" for i in range(reduced_data.shape[1])]
            }
            
        self.engineering_steps.append({
            'step': 'reduce_dimensions',
            'method': method,
            'original_columns': valid_cols,
            'stats': reduction_stats
        })
        
        return self.dataset
    
    def select_features(self, target_column: str, n_features: int = 10, method: str = 'mutual_info') -> List[str]:
        """Select most important features based on relationship with target"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if target_column not in self.dataset.columns:
            raise ValueError(f"Target column {target_column} not found in dataset")
            
        if not self.numerical_columns:
            self.infer_column_types()
            
        # Select only numerical columns with no missing values
        valid_cols = [col for col in self.numerical_columns 
                     if col in self.dataset.columns and col != target_column 
                     and not self.dataset[col].isna().any()]
        
        if len(valid_cols) < 1:
            logger.warning("Not enough valid numerical columns for feature selection")
            return []
            
        X = self.dataset[valid_cols]
        y = self.dataset[target_column]
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=min(n_features, len(valid_cols)))
        else:  # Default to f_regression
            selector = SelectKBest(f_regression, k=min(n_features, len(valid_cols)))
            
        selector.fit(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [valid_cols[i] for i in selected_indices]
        
        # Get scores for each feature
        feature_scores = list(zip(valid_cols, selector.scores_))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.engineering_steps.append({
            'step': 'select_features',
            'method': method,
            'target_column': target_column,
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'feature_scores': [(col, float(score)) for col, score in feature_scores]
        })
        
        return selected_features
    
    def auto_engineer_features(self, target_column: str = None) -> pd.DataFrame:
        """Automatically apply feature engineering techniques based on dataset characteristics"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        # Step 1: Infer column types
        self.infer_column_types()
        
        # Step 2: Create polynomial features for numerical columns (limited to most important)
        if len(self.numerical_columns) > 0:
            top_numerical = self.numerical_columns[:min(5, len(self.numerical_columns))]
            self.create_polynomial_features(columns=top_numerical, degree=2)
        
        # Step 3: Create interaction features between top numerical columns
        if len(self.numerical_columns) >= 2:
            top_numerical = self.numerical_columns[:min(5, len(self.numerical_columns))]
            interaction_pairs = []
            for i, col1 in enumerate(top_numerical):
                for col2 in top_numerical[i+1:]:
                    interaction_pairs.append((col1, col2))
            self.create_interaction_features(interaction_pairs)
        
        # Step 4: Create date features if datetime columns exist
        if len(self.datetime_columns) > 0:
            self.create_date_features()
        
        # Step 5: Create aggregation features if categorical columns exist
        if len(self.categorical_columns) > 0 and len(self.numerical_columns) > 0:
            top_categorical = self.categorical_columns[:min(3, len(self.categorical_columns))]
            top_numerical = self.numerical_columns[:min(3, len(self.numerical_columns))]
            self.create_aggregation_features(
                group_columns=top_categorical,
                agg_columns=top_numerical,
                agg_functions=['mean', 'std', 'min', 'max']
            )
        
        # Step 6: If target column provided, select most important features
        selected_features = None
        if target_column is not None and target_column in self.dataset.columns:
            selected_features = self.select_features(target_column, n_features=20)
        
        return self.dataset, selected_features