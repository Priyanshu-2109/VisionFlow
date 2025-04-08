from flask import Blueprint, jsonify, render_template
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from marshmallow import Schema, fields
from app.config import Config
from flask_swagger_ui import get_swaggerui_blueprint
import yaml
import os

docs_bp = Blueprint('docs', __name__)

# Create APISpec instance
spec = APISpec(
    title="Business Intelligence Platform API",
    version="1.0.0",
    openapi_version="3.0.2",
    plugins=[FlaskPlugin(), MarshmallowPlugin()],
)

# Security scheme
spec.components.security_scheme(
    "bearerAuth",
    {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }
)

# Common response schemas
class ErrorSchema(Schema):
    message = fields.Str(required=True)
    errors = fields.Dict(keys=fields.Str(), values=fields.List(fields.Str()), allow_none=True)

class PaginationSchema(Schema):
    page = fields.Int(required=True)
    per_page = fields.Int(required=True)
    total = fields.Int(required=True)
    pages = fields.Int(required=True)

# Swagger UI configuration
SWAGGER_URL = '/docs'
API_URL = '/static/swagger.yaml'

def register_docs(app):
    """Register API documentation endpoints"""
    
    @docs_bp.route('/docs')
    def get_docs():
        """Get OpenAPI documentation"""
        return jsonify(spec.to_dict())
        
    @docs_bp.route('/docs/swagger')
    def get_swagger():
        """Get Swagger UI"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Documentation</title>
            <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4/swagger-ui.css">
            <script src="https://unpkg.com/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
        </head>
        <body>
            <div id="swagger-ui"></div>
            <script>
                window.onload = function() {{
                    SwaggerUIBundle({{
                        url: "/docs",
                        dom_id: '#swagger-ui',
                        deepLinking: true,
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIBundle.SwaggerUIStandalonePreset
                        ],
                    }})
                }}
            </script>
        </body>
        </html>
        """
        
    @docs_bp.route('/docs/redoc')
    def get_redoc():
        """Get ReDoc UI"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Documentation</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                }}
            </style>
        </head>
        <body>
            <redoc spec-url="/docs"></redoc>
            <script src="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js"></script>
        </body>
        </html>
        """
        
    app.register_blueprint(docs_bp, url_prefix='/api')

def add_docs(app):
    """Add Swagger documentation to the app"""
    # Register Swagger UI blueprint
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "Business Intelligence Platform API"
        }
    )
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

def setup_documentation():
    """Setup API documentation"""
    # Load OpenAPI specification
    spec_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'swagger.yaml')
    try:
        with open(spec_path, 'r') as f:
            spec = yaml.safe_load(f)
        return jsonify(spec)
    except Exception as e:
        return jsonify({
            'error': 'Failed to load API documentation',
            'message': str(e)
        }), 500

@docs_bp.route('/docs')
def view_docs():
    """View API documentation"""
    return render_template('docs.html') 