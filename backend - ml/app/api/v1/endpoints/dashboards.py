from flask import Blueprint, jsonify

bp = Blueprint('dashboards', __name__)

@bp.route('/', methods=['GET'])
def get_dashboards():
    """Get dashboards overview"""
    return jsonify({
        "message": "Dashboards endpoint"
    }), 200 