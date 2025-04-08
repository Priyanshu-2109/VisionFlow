from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from app.models.analytics import AnalysisType, VisualizationType
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class AnalyticsService:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.kmeans = KMeans()
        self.linear_regression = LinearRegression()
        self.random_forest = RandomForestRegressor()
        self.prophet = Prophet()

    def perform_descriptive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform descriptive statistical analysis on the data."""
        try:
            results = {
                "summary_statistics": data.describe().to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "correlation_matrix": data.corr().to_dict(),
                "data_types": data.dtypes.astype(str).to_dict()
            }
            return results
        except Exception as e:
            logger.error(f"Error in descriptive analysis: {str(e)}")
            raise

    def perform_predictive_analysis(self, data: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Perform predictive analysis using machine learning models."""
        try:
            # Prepare data
            X = data.drop(columns=[target])
            y = data[target]
            
            # Split data
            train_size = int(len(data) * 0.8)
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_test = X[train_size:]
            y_test = y[train_size:]

            # Train models
            self.linear_regression.fit(X_train, y_train)
            self.random_forest.fit(X_train, y_train)

            # Make predictions
            lr_pred = self.linear_regression.predict(X_test)
            rf_pred = self.random_forest.predict(X_test)

            # Calculate metrics
            results = {
                "linear_regression": {
                    "predictions": lr_pred.tolist(),
                    "coefficients": self.linear_regression.coef_.tolist(),
                    "intercept": float(self.linear_regression.intercept_)
                },
                "random_forest": {
                    "predictions": rf_pred.tolist(),
                    "feature_importance": dict(zip(X.columns, self.random_forest.feature_importances_))
                }
            }
            return results
        except Exception as e:
            logger.error(f"Error in predictive analysis: {str(e)}")
            raise

    def perform_prescriptive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform prescriptive analysis to recommend actions."""
        try:
            # Identify anomalies
            z_scores = stats.zscore(data)
            anomalies = np.abs(z_scores) > 3

            # Perform clustering
            scaled_data = self.scaler.fit_transform(data)
            self.kmeans.fit(scaled_data)
            clusters = self.kmeans.labels_

            # Perform PCA
            pca_result = self.pca.fit_transform(scaled_data)

            results = {
                "anomalies": anomalies.tolist(),
                "clusters": clusters.tolist(),
                "pca": {
                    "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                    "components": self.pca.components_.tolist()
                }
            }
            return results
        except Exception as e:
            logger.error(f"Error in prescriptive analysis: {str(e)}")
            raise

    def perform_diagnostic_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform diagnostic analysis to understand relationships and patterns."""
        try:
            # Calculate correlations
            correlations = data.corr()

            # Perform statistical tests
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            statistical_tests = {}
            for col1 in numeric_cols:
                for col2 in numeric_cols:
                    if col1 != col2:
                        correlation, p_value = stats.pearsonr(data[col1], data[col2])
                        statistical_tests[f"{col1}_{col2}"] = {
                            "correlation": correlation,
                            "p_value": p_value
                        }

            results = {
                "correlations": correlations.to_dict(),
                "statistical_tests": statistical_tests
            }
            return results
        except Exception as e:
            logger.error(f"Error in diagnostic analysis: {str(e)}")
            raise

    def create_visualization(
        self,
        data: pd.DataFrame,
        viz_type: VisualizationType,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create visualizations based on the data and configuration."""
        try:
            if config is None:
                config = {}

            if viz_type == VisualizationType.LINE:
                fig = px.line(data, **config)
            elif viz_type == VisualizationType.BAR:
                fig = px.bar(data, **config)
            elif viz_type == VisualizationType.PIE:
                fig = px.pie(data, **config)
            elif viz_type == VisualizationType.SCATTER:
                fig = px.scatter(data, **config)
            elif viz_type == VisualizationType.HEATMAP:
                fig = px.imshow(data.corr(), **config)
            elif viz_type == VisualizationType.BOX:
                fig = px.box(data, **config)
            elif viz_type == VisualizationType.VIOLIN:
                fig = px.violin(data, **config)
            elif viz_type == VisualizationType.RADAR:
                fig = go.Figure()
                for col in data.columns:
                    fig.add_trace(go.Scatterpolar(
                        r=data[col],
                        theta=data.index,
                        name=col
                    ))
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")

            return {
                "data": fig.to_dict(),
                "layout": fig.layout.to_dict()
            }
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            raise

    def perform_time_series_analysis(self, data: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """Perform time series analysis using Prophet."""
        try:
            # Prepare data for Prophet
            prophet_data = data.rename(columns={date_col: 'ds'})
            
            # Fit Prophet model
            self.prophet.fit(prophet_data)
            
            # Make future predictions
            future = self.prophet.make_future_dataframe(periods=30)
            forecast = self.prophet.predict(future)

            results = {
                "forecast": forecast.to_dict(),
                "trend": self.prophet.trend.to_dict(),
                "seasonality": self.prophet.seasonality.to_dict()
            }
            return results
        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            raise

    def perform_text_analysis(self, text_data: List[str]) -> Dict[str, Any]:
        """Perform text analysis on a list of text documents."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(text_data, columns=['text'])
            
            # Basic text analysis
            results = {
                "word_counts": df['text'].str.split().str.len().describe().to_dict(),
                "character_counts": df['text'].str.len().describe().to_dict(),
                "unique_words": len(set(' '.join(text_data).split())),
                "average_word_length": df['text'].str.split().apply(lambda x: np.mean([len(w) for w in x])).mean()
            }
            return results
        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            raise 