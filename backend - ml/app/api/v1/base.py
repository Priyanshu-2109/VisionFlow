from flask import Blueprint, jsonify, request
from typing import Any, Dict, List, Optional
from app.services.base import BaseService
from app.middleware.auth import auth_required

class BaseAPI:
    """Base API class with common CRUD endpoints"""
    
    def __init__(self, service: BaseService, blueprint_name: str, url_prefix: str):
        self.service = service
        self.blueprint = Blueprint(blueprint_name, __name__, url_prefix=url_prefix)
        self.register_routes()

    def register_routes(self):
        """Register API routes"""
        @self.blueprint.route('/', methods=['GET'])
        @auth_required
        def get_all():
            items = self.service.get_all()
            return jsonify(items)

        @self.blueprint.route('/<int:id>', methods=['GET'])
        @auth_required
        def get_by_id(id: int):
            item = self.service.get(id)
            if not item:
                return jsonify({"message": "Item not found"}), 404
            return jsonify(item)

        @self.blueprint.route('/', methods=['POST'])
        @auth_required
        def create():
            data = request.get_json()
            try:
                item = self.service.create(data)
                return jsonify(item), 201
            except ValueError as e:
                return jsonify({"message": str(e)}), 400

        @self.blueprint.route('/<int:id>', methods=['PUT'])
        @auth_required
        def update(id: int):
            data = request.get_json()
            try:
                item = self.service.update(id, data)
                if not item:
                    return jsonify({"message": "Item not found"}), 404
                return jsonify(item)
            except ValueError as e:
                return jsonify({"message": str(e)}), 400

        @self.blueprint.route('/<int:id>', methods=['DELETE'])
        @auth_required
        def delete(id: int):
            if self.service.delete(id):
                return '', 204
            return jsonify({"message": "Item not found"}), 404

        @self.blueprint.route('/filter', methods=['GET'])
        @auth_required
        def filter():
            filters = request.args.to_dict()
            items = self.service.filter(**filters)
            return jsonify(items) 