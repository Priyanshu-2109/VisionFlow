from flask import Blueprint, jsonify

bp = Blueprint('data_sources', __name__)

@bp.route('/', methods=['GET'])
def get_data_sources():
    """Get data sources overview"""
    return jsonify({
        "message": "Data sources endpoint"
    }), 200 