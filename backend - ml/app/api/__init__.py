from flask import Blueprint, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create main API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize CORS
CORS(api_bp, resources={r"/*": {"origins": "*"}})

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Health check endpoint
@api_bp.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    })

# Import routes after blueprint creation to avoid circular imports
from app.api import routes

# Register version 1 blueprint
from app.api.v1 import bp as api_v1_bp
api_bp.register_blueprint(api_v1_bp, url_prefix='/v1')
