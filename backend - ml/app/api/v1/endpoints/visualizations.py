from flask import Blueprint, jsonify

bp = Blueprint('visualizations', __name__)

@bp.route('/', methods=['GET'])
def get_visualizations():
    """Get visualizations overview"""
    return jsonify({
        "message": "Visualizations endpoint"
    }), 200 