from flask import Blueprint

# Create API v1 blueprint
bp = Blueprint('api_v1', __name__)

# Import routes after blueprint creation to avoid circular imports
from app.api.v1 import routes

# Import and include other routers
from app.api.v1.endpoints import (
    analytics,
    reports,
    dashboards,
    data_sources,
    visualizations
)

# Register routers
bp.register_blueprint(analytics.bp, url_prefix='/analytics')
bp.register_blueprint(reports.bp, url_prefix='/reports')
bp.register_blueprint(dashboards.bp, url_prefix='/dashboards')
bp.register_blueprint(data_sources.bp, url_prefix='/data-sources')
bp.register_blueprint(visualizations.bp, url_prefix='/visualizations') 