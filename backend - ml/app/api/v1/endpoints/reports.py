from flask import Blueprint, jsonify

bp = Blueprint('reports', __name__)

@bp.route('/', methods=['GET'])
def get_reports():
    """Get reports overview"""
    return jsonify({
        "message": "Reports endpoint"
    }), 200 