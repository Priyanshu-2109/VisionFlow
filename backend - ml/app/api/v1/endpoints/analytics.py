from flask import Blueprint, jsonify

bp = Blueprint('analytics', __name__)

@bp.route('/', methods=['GET'])
def get_analytics():
    """Get analytics overview"""
    return jsonify({
        "message": "Analytics endpoint"
    }), 200 