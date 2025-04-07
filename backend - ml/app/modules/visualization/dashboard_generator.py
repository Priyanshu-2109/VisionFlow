# app/modules/visualization/dashboard_generator.py
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class DashboardGenerator:
    """
    Comprehensive dashboard generator for business datasets
    """
    
    def __init__(self, dataset: pd.DataFrame = None):
        self.dataset = dataset
        self.numerical_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.text_columns = []
        self.charts = []
        self.insights = []
        self.color_scheme = 'blues'  # Default color scheme
        
    def set_dataset(self, dataset: pd.DataFrame) -> None:
        """Set the dataset to visualize"""
        self.dataset = dataset
        
    def set_column_types(self, numerical: List[str], categorical: List[str], 
                        datetime: List[str], text: List[str] = None) -> None:
        """Set column types manually"""
        self.numerical_columns = numerical
        self.categorical_columns = categorical
        self.datetime_columns = datetime
        self.text_columns = text or []
        
    def infer_column_types(self) -> Dict[str, List[str]]:
        """Infer column types from dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        self.numerical_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.text_columns = []
        
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
                    # If it has a lot of text, classify as text
                    if self.dataset[col].astype(str).str.len().mean() > 50:
                        self.text_columns.append(col)
                    else:
                        self.categorical_columns.append(col)
        
        return {
            'numerical': self.numerical_columns,
            'categorical': self.categorical_columns,
            'datetime': self.datetime_columns,
            'text': self.text_columns
        }
    
    def set_color_scheme(self, scheme: str) -> None:
        """Set color scheme for visualizations"""
        valid_schemes = ['blues', 'greens', 'reds', 'purples', 'oranges', 
                         'viridis', 'plasma', 'inferno', 'magma', 'cividis']
        
        if scheme.lower() in valid_schemes:
            self.color_scheme = scheme.lower()
        else:
            logger.warning(f"Invalid color scheme: {scheme}. Using default 'blues'.")
            self.color_scheme = 'blues'
    
    def create_histogram(self, column: str, bins: int = 30, title: str = None) -> Dict[str, Any]:
        """Create histogram for numerical column"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if column not in self.dataset.columns:
            raise ValueError(f"Column {column} not found in dataset")
            
        # Create Histogram
        fig = px.histogram(
            self.dataset, 
            x=column, 
            nbins=bins,
            title=title or f"Distribution of {column}",
            color_discrete_sequence=px.colors.sequential.Blues,
            opacity=0.8,
            marginal="box"
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            bargap=0.05,
            template="plotly_white"
        )
        
        # Add KDE (Density) Line
        data = self.dataset[column].dropna().values
        kde_fig = ff.create_distplot([data], [column], show_hist=False, show_rug=False)

        for trace in kde_fig.data:
            trace.update(line=dict(color='red', width=2))
            fig.add_trace(trace)
        
        # Calculate basic statistics
        stats = {
            'mean': float(self.dataset[column].mean()),
            'median': float(self.dataset[column].median()),
            'std': float(self.dataset[column].std()),
            'min': float(self.dataset[column].min()),
            'max': float(self.dataset[column].max()),
            'skewness': float(self.dataset[column].skew()),
            'kurtosis': float(self.dataset[column].kurtosis())
        }
        
        chart_data = {
            'type': 'histogram',
            'column': column,
            'title': title or f"Distribution of {column}",
            'plotly_figure': fig.to_json(),
            'stats': stats
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def create_bar_chart(self, column: str, top_n: int = 10, title: str = None, 
                         horizontal: bool = False) -> Dict[str, Any]:
        """Create bar chart for categorical column"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if column not in self.dataset.columns:
            raise ValueError(f"Column {column} not found in dataset")
            
        # Get value counts and take top N
        value_counts = self.dataset[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        
        if len(value_counts) > top_n:
            top_values = value_counts.head(top_n-1)
            other_count = value_counts.iloc[top_n-1:]['count'].sum()
            other_row = pd.DataFrame({column: ['Other'], 'count': [other_count]})
            value_counts = pd.concat([top_values, other_row], ignore_index=True)
        
        # Create bar chart
        if horizontal:
            fig = px.bar(
                value_counts, 
                y=column, 
                x='count',
                title=title or f"Distribution of {column}",
                color_discrete_sequence=px.colors.sequential.Blues,
                orientation='h'
            )
        else:
            fig = px.bar(
                value_counts, 
                x=column, 
                y='count',
                title=title or f"Distribution of {column}",
                color_discrete_sequence=px.colors.sequential.Blues
            )
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title="" if horizontal else column,
            yaxis_title="Count" if horizontal else "Count"
        )
        
        # Calculate percentage for each category
        total = value_counts['count'].sum()
        percentages = {row[column]: float(row['count'] / total * 100) 
                      for _, row in value_counts.iterrows()}
        
        chart_data = {
            'type': 'bar_chart',
            'column': column,
            'title': title or f"Distribution of {column}",
            'plotly_figure': fig.to_json(),
            'value_counts': value_counts.to_dict(orient='records'),
            'percentages': percentages
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def create_pie_chart(self, column: str, top_n: int = 7, title: str = None) -> Dict[str, Any]:
        """Create pie chart for categorical column"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if column not in self.dataset.columns:
            raise ValueError(f"Column {column} not found in dataset")
            
        # Get value counts and take top N
        value_counts = self.dataset[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        
        if len(value_counts) > top_n:
            top_values = value_counts.head(top_n-1)
            other_count = value_counts.iloc[top_n-1:]['count'].sum()
            other_row = pd.DataFrame({column: ['Other'], 'count': [other_count]})
            value_counts = pd.concat([top_values, other_row], ignore_index=True)
        
        # Create pie chart
        fig = px.pie(
            value_counts, 
            names=column, 
            values='count',
            title=title or f"Distribution of {column}",
            color_discrete_sequence=px.colors.sequential.Blues,
            hole=0.3
        )
        
        fig.update_layout(
            template="plotly_white"
        )
        
        # Calculate percentage for each category
        total = value_counts['count'].sum()
        percentages = {row[column]: float(row['count'] / total * 100) 
                      for _, row in value_counts.iterrows()}
        
        chart_data = {
            'type': 'pie_chart',
            'column': column,
            'title': title or f"Distribution of {column}",
            'plotly_figure': fig.to_json(),
            'value_counts': value_counts.to_dict(orient='records'),
            'percentages': percentages
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def create_scatter_plot(self, x_column: str, y_column: str, color_column: str = None,
                           size_column: str = None, title: str = None) -> Dict[str, Any]:
        """Create scatter plot between two numerical columns"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if x_column not in self.dataset.columns or y_column not in self.dataset.columns:
            raise ValueError(f"Columns {x_column} or {y_column} not found in dataset")
            
        # Create scatter plot
        fig = px.scatter(
            self.dataset,
            x=x_column,
            y=y_column,
            color=color_column,
            size=size_column,
            title=title or f"{y_column} vs {x_column}",
            opacity=0.7,
            color_continuous_scale=self.color_scheme if color_column and pd.api.types.is_numeric_dtype(self.dataset[color_column]) else None
        )
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title=x_column,
            yaxis_title=y_column
        )
        
        # Add trend line
        if pd.api.types.is_numeric_dtype(self.dataset[x_column]) and pd.api.types.is_numeric_dtype(self.dataset[y_column]):
            valid_mask = ~(self.dataset[[x_column, y_column]].isna().any(axis=1))
            if valid_mask.sum() > 1:  # Need at least 2 points for a trend line
                x = self.dataset.loc[valid_mask, x_column]
                y = self.dataset.loc[valid_mask, y_column]
                
                # Calculate correlation
                correlation = float(x.corr(y))
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.poly1d(np.polyfit(x, y, 1))(x),
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='red', width=2, dash='dash')
                    )
                )
            else:
                correlation = None
        else:
            correlation = None
        
        chart_data = {
            'type': 'scatter_plot',
            'x_column': x_column,
            'y_column': y_column,
            'color_column': color_column,
            'size_column': size_column,
            'title': title or f"{y_column} vs {x_column}",
            'plotly_figure': fig.to_json(),
            'correlation': correlation
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def create_line_chart(self, x_column: str, y_columns: List[str], title: str = None) -> Dict[str, Any]:
        """Create line chart for time series or sequential data"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if x_column not in self.dataset.columns:
            raise ValueError(f"Column {x_column} not found in dataset")
            
        for col in y_columns:
            if col not in self.dataset.columns:
                raise ValueError(f"Column {col} not found in dataset")
        
        # Sort by x column if it's datetime
        df = self.dataset.copy()
        if pd.api.types.is_datetime64_dtype(df[x_column]):
            df = df.sort_values(by=x_column)
        
        # Create line chart
        fig = go.Figure()
        
        for col in y_columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x_column],
                    y=df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(width=2),
                    marker=dict(size=5)
                )
            )
        
        fig.update_layout(
            title=title or f"Trend of {', '.join(y_columns)} over {x_column}",
            xaxis_title=x_column,
            yaxis_title="Value",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Calculate trend statistics
        trend_stats = {}
        if pd.api.types.is_datetime64_dtype(df[x_column]) or pd.api.types.is_numeric_dtype(df[x_column]):
            for col in y_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Calculate growth
                    first_valid = df[col].first_valid_index()
                    last_valid = df[col].last_valid_index()
                    
                    if first_valid is not None and last_valid is not None:
                        start_value = df.loc[first_valid, col]
                        end_value = df.loc[last_valid, col]
                        
                        if start_value != 0:
                            growth_pct = (end_value - start_value) / abs(start_value) * 100
                        else:
                            growth_pct = float('inf') if end_value > 0 else (0 if end_value == 0 else float('-inf'))
                        
                        trend_stats[col] = {
                            'start_value': float(start_value),
                            'end_value': float(end_value),
                            'absolute_change': float(end_value - start_value),
                            'percentage_change': float(growth_pct)
                        }
        
        chart_data = {
            'type': 'line_chart',
            'x_column': x_column,
            'y_columns': y_columns,
            'title': title or f"Trend of {', '.join(y_columns)} over {x_column}",
            'plotly_figure': fig.to_json(),
            'trend_stats': trend_stats
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def create_box_plot(self, x_column: str, y_column: str, title: str = None) -> Dict[str, Any]:
        """Create box plot to show distribution across categories"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if x_column not in self.dataset.columns or y_column not in self.dataset.columns:
            raise ValueError(f"Columns {x_column} or {y_column} not found in dataset")
        
        # Create box plot
        fig = px.box(
            self.dataset,
            x=x_column,
            y=y_column,
            title=title or f"Distribution of {y_column} by {x_column}",
            color=x_column,
            color_discrete_sequence=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title=x_column,
            yaxis_title=y_column,
            showlegend=False
        )
        
        # Calculate statistics for each group
        group_stats = {}
        for group in self.dataset[x_column].unique():
            group_data = self.dataset[self.dataset[x_column] == group][y_column]
            group_stats[str(group)] = {
                'mean': float(group_data.mean()),
                'median': float(group_data.median()),
                'std': float(group_data.std()),
                'min': float(group_data.min()),
                'max': float(group_data.max()),
                'count': int(len(group_data))
            }
        
        chart_data = {
            'type': 'box_plot',
            'x_column': x_column,
            'y_column': y_column,
            'title': title or f"Distribution of {y_column} by {x_column}",
            'plotly_figure': fig.to_json(),
            'group_stats': group_stats
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def create_heatmap(self, columns: List[str] = None, title: str = None) -> Dict[str, Any]:
        """Create correlation heatmap between numerical columns"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if not self.numerical_columns and not columns:
            self.infer_column_types()
            
        # Use specified columns or default to numerical columns
        if columns is None:
            columns = self.numerical_columns
            
        # Filter to only include columns in the dataset
        columns = [col for col in columns if col in self.dataset.columns]
        
        if len(columns) < 2:
            raise ValueError("Need at least 2 numerical columns for a heatmap")
            
        # Calculate correlation matrix
        corr_matrix = self.dataset[columns].corr().round(2)
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale=self.color_scheme,
            title=title or "Correlation Heatmap"
        )
        
        fig.update_layout(
            template="plotly_white"
        )
        
        # Find highest correlations (excluding self-correlations)
        corr_pairs = []
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                corr_pairs.append({
                    'column1': columns[i],
                    'column2': columns[j],
                    'correlation': float(corr_matrix.iloc[i, j])
                })
        
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        chart_data = {
            'type': 'heatmap',
            'columns': columns,
            'title': title or "Correlation Heatmap",
            'plotly_figure': fig.to_json(),
            'correlation_matrix': corr_matrix.to_dict(),
            'top_correlations': corr_pairs[:5]
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def create_stacked_bar(self, x_column: str, y_column: str, color_column: str,
                          title: str = None) -> Dict[str, Any]:
        """Create stacked bar chart to show composition across categories"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if (x_column not in self.dataset.columns or 
            y_column not in self.dataset.columns or
            color_column not in self.dataset.columns):
            raise ValueError(f"One or more columns not found in dataset")
        
        # Aggregate data
        df_agg = self.dataset.groupby([x_column, color_column])[y_column].sum().reset_index()
        
        # Create stacked bar chart
        fig = px.bar(
            df_agg,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title or f"{y_column} by {x_column} and {color_column}",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title=x_column,
            yaxis_title=y_column,
            legend_title=color_column
        )
        
        # Calculate percentages within each x-category
        pct_data = []
        for x_val in df_agg[x_column].unique():
            group_data = df_agg[df_agg[x_column] == x_val]
            total = group_data[y_column].sum()
            
            for _, row in group_data.iterrows():
                pct_data.append({
                    x_column: row[x_column],
                    color_column: row[color_column],
                    y_column: float(row[y_column]),
                    'percentage': float(row[y_column] / total * 100) if total > 0 else 0
                })
        
        chart_data = {
            'type': 'stacked_bar',
            'x_column': x_column,
            'y_column': y_column,
            'color_column': color_column,
            'title': title or f"{y_column} by {x_column} and {color_column}",
            'plotly_figure': fig.to_json(),
            'percentage_data': pct_data
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def create_bubble_chart(self, x_column: str, y_column: str, size_column: str,
                           color_column: str = None, title: str = None) -> Dict[str, Any]:
        """Create bubble chart with size and optional color dimensions"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if (x_column not in self.dataset.columns or 
            y_column not in self.dataset.columns or
            size_column not in self.dataset.columns):
            raise ValueError(f"One or more columns not found in dataset")
        
        # Create bubble chart
        fig = px.scatter(
            self.dataset,
            x=x_column,
            y=y_column,
            size=size_column,
            color=color_column,
            title=title or f"{y_column} vs {x_column} (size: {size_column})",
            opacity=0.7,
            color_continuous_scale=self.color_scheme if color_column and pd.api.types.is_numeric_dtype(self.dataset[color_column]) else None,
            size_max=50
        )
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title=x_column,
            yaxis_title=y_column
        )
        
        chart_data = {
            'type': 'bubble_chart',
            'x_column': x_column,
            'y_column': y_column,
            'size_column': size_column,
            'color_column': color_column,
            'title': title or f"{y_column} vs {x_column} (size: {size_column})",
            'plotly_figure': fig.to_json()
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def create_sunburst_chart(self, path_columns: List[str], value_column: str = None,
                             title: str = None) -> Dict[str, Any]:
        """Create sunburst chart to visualize hierarchical data"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        for col in path_columns:
            if col not in self.dataset.columns:
                raise ValueError(f"Column {col} not found in dataset")
                
        if value_column and value_column not in self.dataset.columns:
            raise ValueError(f"Value column {value_column} not found in dataset")
        
        # Create sunburst chart
        fig = px.sunburst(
            self.dataset,
            path=path_columns,
            values=value_column,
            title=title or f"Hierarchical View of {', '.join(path_columns)}",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            template="plotly_white"
        )
        
        chart_data = {
            'type': 'sunburst_chart',
            'path_columns': path_columns,
            'value_column': value_column,
            'title': title or f"Hierarchical View of {', '.join(path_columns)}",
            'plotly_figure': fig.to_json()
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def create_treemap(self, path_columns: List[str], value_column: str,
                      title: str = None) -> Dict[str, Any]:
        """Create treemap to visualize hierarchical data with area proportional to values"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        for col in path_columns:
            if col not in self.dataset.columns:
                raise ValueError(f"Column {col} not found in dataset")
                
        if value_column not in self.dataset.columns:
            raise ValueError(f"Value column {value_column} not found in dataset")
        
        # Create treemap chart
        fig = px.treemap(
            self.dataset,
            path=path_columns,
            values=value_column,
            title=title or f"Treemap of {value_column} by {', '.join(path_columns)}",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            template="plotly_white"
        )
        
        chart_data = {
            'type': 'treemap',
            'path_columns': path_columns,
            'value_column': value_column,
            'title': title or f"Treemap of {value_column} by {', '.join(path_columns)}",
            'plotly_figure': fig.to_json()
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics for the dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if not (self.numerical_columns or self.categorical_columns):
            self.infer_column_types()
            
        summary = {
            'dataset_shape': self.dataset.shape,
            'column_types': {
                'numerical': len(self.numerical_columns),
                'categorical': len(self.categorical_columns),
                'datetime': len(self.datetime_columns),
                'text': len(self.text_columns)
            },
            'missing_values': self.dataset.isna().sum().to_dict(),
            'numerical_summary': {},
            'categorical_summary': {}
        }
        
        # Numerical summary
        for col in self.numerical_columns:
            if col in self.dataset.columns:
                summary['numerical_summary'][col] = {
                    'mean': float(self.dataset[col].mean()),
                    'median': float(self.dataset[col].median()),
                    'std': float(self.dataset[col].std()),
                    'min': float(self.dataset[col].min()),
                    'max': float(self.dataset[col].max()),
                    'missing': int(self.dataset[col].isna().sum()),
                    'missing_pct': float(self.dataset[col].isna().mean() * 100)
                }
        
        # Categorical summary
        for col in self.categorical_columns:
            if col in self.dataset.columns:
                value_counts = self.dataset[col].value_counts()
                top_values = value_counts.head(5).to_dict()
                summary['categorical_summary'][col] = {
                    'unique_values': int(self.dataset[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in top_values.items()},
                    'missing': int(self.dataset[col].isna().sum()),
                    'missing_pct': float(self.dataset[col].isna().mean() * 100)
                }
        
        return summary
    
    def generate_insights(self) -> List[Dict[str, Any]]:
        """Generate automated insights from the dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        if not (self.numerical_columns or self.categorical_columns):
            self.infer_column_types()
            
        insights = []
        
        # 1. Missing value patterns
        missing_cols = {col: self.dataset[col].isna().mean() * 100 
                       for col in self.dataset.columns if self.dataset[col].isna().any()}
        
        if missing_cols:
            high_missing = {col: pct for col, pct in missing_cols.items() if pct > 20}
            if high_missing:
                insights.append({
                    'type': 'missing_values',
                    'title': 'High Missing Values',
                    'description': f"Several columns have a high percentage of missing values: {', '.join([f'{col} ({pct:.1f}%)' for col, pct in high_missing.items()])}. Consider imputation or removal strategies.",
                    'columns': list(high_missing.keys()),
                    'severity': 'high' if any(pct > 50 for pct in high_missing.values()) else 'medium'
                })
        
        # 2. High correlations
        if len(self.numerical_columns) >= 2:
            corr_matrix = self.dataset[self.numerical_columns].corr()
            high_corr_pairs = []
            
            for i in range(len(self.numerical_columns)):
                for j in range(i+1, len(self.numerical_columns)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.8:
                        high_corr_pairs.append({
                            'column1': self.numerical_columns[i],
                            'column2': self.numerical_columns[j],
                            'correlation': float(corr)
                        })
            
            if high_corr_pairs:
                insights.append({
                    'type': 'high_correlation',
                    'title': 'High Correlations Detected',
                    'description': f"Strong correlations found between variables, suggesting potential multicollinearity. Top pair: {high_corr_pairs[0]['column1']} and {high_corr_pairs[0]['column2']} (r={high_corr_pairs[0]['correlation']:.2f})",
                    'correlation_pairs': high_corr_pairs,
                    'severity': 'medium'
                })
        
        # 3. Outlier detection
        outlier_cols = {}
        for col in self.numerical_columns:
            if col in self.dataset.columns:
                Q1 = self.dataset[col].quantile(0.25)
                Q3 = self.dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.dataset[col] < lower_bound) | (self.dataset[col] > upper_bound)).sum()
                outlier_pct = outliers / len(self.dataset) * 100
                
                if outlier_pct > 5:
                    outlier_cols[col] = float(outlier_pct)
        
        if outlier_cols:
            insights.append({
                'type': 'outliers',
                'title': 'Significant Outliers Detected',
                'description': f"Several columns contain significant outliers: {', '.join([f'{col} ({pct:.1f}%)' for col, pct in outlier_cols.items()])}. These may affect statistical analyses and model performance.",
                'columns': list(outlier_cols.keys()),
                'severity': 'medium'
            })
        
        # 4. Skewed distributions
        skewed_cols = {}
        for col in self.numerical_columns:
            if col in self.dataset.columns:
                skewness = self.dataset[col].skew()
                if abs(skewness) > 1.5:
                    skewed_cols[col] = float(skewness)
        
        if skewed_cols:
            insights.append({
                'type': 'skewed_distribution',
                'title': 'Highly Skewed Distributions',
                'description': f"Several numerical columns have highly skewed distributions: {', '.join([f'{col} (skew={skew:.1f})' for col, skew in skewed_cols.items()])}. Consider transformations for analyses that assume normality.",
                'columns': list(skewed_cols.keys()),
                'severity': 'low'
            })
        
        # 5. Imbalanced categories
        imbalanced_cols = {}
        for col in self.categorical_columns:
            if col in self.dataset.columns:
                value_counts = self.dataset[col].value_counts(normalize=True)
                if value_counts.iloc[0] > 0.8:  # If the most common category is over 80%
                    imbalanced_cols[col] = float(value_counts.iloc[0] * 100)
        
        if imbalanced_cols:
            insights.append({
                'type': 'imbalanced_categories',
                'title': 'Imbalanced Categorical Variables',
                'description': f"Several categorical columns are heavily imbalanced: {', '.join([f'{col} (top category: {pct:.1f}%)' for col, pct in imbalanced_cols.items()])}. This may affect model training and evaluation.",
                'columns': list(imbalanced_cols.keys()),
                'severity': 'medium'
            })
        
        # 6. Constant or near-constant columns
        constant_cols = []
        for col in self.dataset.columns:
            if self.dataset[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            insights.append({
                'type': 'constant_columns',
                'title': 'Constant Columns Detected',
                'description': f"The following columns have only one unique value: {', '.join(constant_cols)}. These provide no information and should be removed.",
                'columns': constant_cols,
                'severity': 'high'
            })
        
        # Store insights
        self.insights = insights
        return insights
    def auto_generate_dashboard(self) -> Dict[str, Any]:
        """Automatically generate a comprehensive dashboard based on dataset characteristics"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        # 1. Infer column types
        self.infer_column_types()
        
        # 2. Generate summary statistics
        summary = self.generate_summary_statistics()
        
        # 3. Generate insights
        insights = self.generate_insights()
        
        # 4. Generate visualizations based on data types
        # 4.1 Numerical columns visualizations
        for col in self.numerical_columns[:5]:  # Limit to first 5 numerical columns
            self.create_histogram(col)
        
        # 4.2 Categorical columns visualizations
        for col in self.categorical_columns[:5]:  # Limit to first 5 categorical columns
            if self.dataset[col].nunique() <= 10:
                self.create_pie_chart(col)
            else:
                self.create_bar_chart(col, horizontal=True)
        
        # 4.3 Correlation heatmap for numerical columns
        if len(self.numerical_columns) >= 2:
            self.create_heatmap()
        
        # 4.4 Time series visualizations if datetime columns exist
        if self.datetime_columns:
            time_col = self.datetime_columns[0]
            # Find top 2 numerical columns for time series visualization
            num_cols_for_time = self.numerical_columns[:2] if len(self.numerical_columns) >= 2 else self.numerical_columns
            if num_cols_for_time:
                self.create_line_chart(time_col, num_cols_for_time)
        
        # 4.5 Scatter plots between top numerical columns
        if len(self.numerical_columns) >= 2:
            self.create_scatter_plot(self.numerical_columns[0], self.numerical_columns[1])
            
            # Add color dimension if categorical column exists
            if self.categorical_columns:
                self.create_scatter_plot(
                    self.numerical_columns[0], 
                    self.numerical_columns[1],
                    color_column=self.categorical_columns[0]
                )
        
        # 4.6 Box plots for numerical vs categorical
        if self.numerical_columns and self.categorical_columns:
            for cat_col in self.categorical_columns[:2]:  # Limit to first 2 categorical columns
                if self.dataset[cat_col].nunique() <= 10:  # Only if not too many categories
                    self.create_box_plot(cat_col, self.numerical_columns[0])
        
        # 4.7 Create hierarchical visualization if multiple categorical columns
        if len(self.categorical_columns) >= 2 and self.numerical_columns:
            path_cols = self.categorical_columns[:3]  # Use up to 3 categories for hierarchy
            self.create_treemap(path_cols, self.numerical_columns[0])
        
        # 5. Compile dashboard data
        dashboard = {
            'title': 'Automated Data Dashboard',
            'summary': summary,
            'insights': insights,
            'charts': self.charts,
            'dataset_preview': self.dataset.head(10).to_dict(orient='records'),
            'column_types': {
                'numerical': self.numerical_columns,
                'categorical': self.categorical_columns,
                'datetime': self.datetime_columns,
                'text': self.text_columns
            }
        }
        
        return dashboard
    
    def generate_dashboard_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Generate a customized dashboard based on natural language prompt"""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset first.")
            
        # Ensure column types are inferred
        if not (self.numerical_columns or self.categorical_columns):
            self.infer_column_types()
            
        # Basic prompt parsing (in a real system, this would use a more sophisticated NLP approach)
        prompt = prompt.lower()
        
        # Extract column mentions from prompt
        mentioned_columns = []
        for col in self.dataset.columns:
            if col.lower() in prompt:
                mentioned_columns.append(col)
        
        # If no columns specifically mentioned, use auto-generation
        if not mentioned_columns:
            return self.auto_generate_dashboard()
        
        # Create visualizations based on prompt keywords and mentioned columns
        charts = []
        
        # Time series / trends
        if any(kw in prompt for kw in ['trend', 'time', 'series', 'over time', 'historical']):
            time_cols = [col for col in self.datetime_columns if col in mentioned_columns]
            if not time_cols and self.datetime_columns:
                time_cols = [self.datetime_columns[0]]  # Use first datetime column if none mentioned
                
            if time_cols:
                num_cols = [col for col in self.numerical_columns if col in mentioned_columns]
                if not num_cols:
                    num_cols = self.numerical_columns[:2]  # Use first two numerical columns if none mentioned
                    
                if num_cols:
                    charts.append(self.create_line_chart(time_cols[0], num_cols))
        
        # Distributions
        if any(kw in prompt for kw in ['distribution', 'histogram', 'spread']):
            num_cols = [col for col in self.numerical_columns if col in mentioned_columns]
            if not num_cols:
                num_cols = self.numerical_columns[:3]  # Use first three numerical columns if none mentioned
                
            for col in num_cols:
                charts.append(self.create_histogram(col))
        
        # Categories / proportions
        if any(kw in prompt for kw in ['category', 'proportion', 'percentage', 'breakdown']):
            cat_cols = [col for col in self.categorical_columns if col in mentioned_columns]
            if not cat_cols:
                cat_cols = self.categorical_columns[:3]  # Use first three categorical columns if none mentioned
                
            for col in cat_cols:
                if self.dataset[col].nunique() <= 7:
                    charts.append(self.create_pie_chart(col))
                else:
                    charts.append(self.create_bar_chart(col))
        
        # Relationships / correlations
        if any(kw in prompt for kw in ['relationship', 'correlation', 'compare', 'versus', 'vs']):
            num_cols = [col for col in self.numerical_columns if col in mentioned_columns]
            
            if len(num_cols) >= 2:
                charts.append(self.create_scatter_plot(num_cols[0], num_cols[1]))
                charts.append(self.create_heatmap(num_cols))
            elif len(self.numerical_columns) >= 2:
                charts.append(self.create_scatter_plot(self.numerical_columns[0], self.numerical_columns[1]))
                charts.append(self.create_heatmap(self.numerical_columns[:5]))
        
        # Hierarchical / nested data
        if any(kw in prompt for kw in ['hierarchy', 'nested', 'drill down', 'breakdown by']):
            cat_cols = [col for col in self.categorical_columns if col in mentioned_columns]
            num_cols = [col for col in self.numerical_columns if col in mentioned_columns]
            
            if len(cat_cols) >= 2 and num_cols:
                charts.append(self.create_treemap(cat_cols[:3], num_cols[0]))
            elif len(self.categorical_columns) >= 2 and self.numerical_columns:
                charts.append(self.create_treemap(self.categorical_columns[:3], self.numerical_columns[0]))
        
        # If no specific charts were created, fall back to auto-generation
        if not charts:
            return self.auto_generate_dashboard()
        
        # Generate summary statistics
        summary = self.generate_summary_statistics()
        
        # Generate insights
        insights = self.generate_insights()
        
        # Compile dashboard data
        dashboard = {
            'title': 'Custom Dashboard: ' + prompt[:50] + ('...' if len(prompt) > 50 else ''),
            'prompt': prompt,
            'summary': summary,
            'insights': insights,
            'charts': self.charts,  # Use all charts including those created before
            'dataset_preview': self.dataset.head(10).to_dict(orient='records'),
            'column_types': {
                'numerical': self.numerical_columns,
                'categorical': self.categorical_columns,
                'datetime': self.datetime_columns,
                'text': self.text_columns
            }
        }
        
        return dashboard