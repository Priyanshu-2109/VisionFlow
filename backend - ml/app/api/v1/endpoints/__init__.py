from app.api.v1.endpoints.analytics import bp as analytics_bp
from app.api.v1.endpoints.reports import bp as reports_bp
from app.api.v1.endpoints.dashboards import bp as dashboards_bp
from app.api.v1.endpoints.data_sources import bp as data_sources_bp
from app.api.v1.endpoints.visualizations import bp as visualizations_bp

__all__ = [
    'analytics_bp',
    'reports_bp',
    'dashboards_bp',
    'data_sources_bp',
    'visualizations_bp'
] 