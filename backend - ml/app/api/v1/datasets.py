from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models.dataset import Dataset
from app.api.deps import get_current_user
from app.schemas.dataset import DatasetSchema, DatasetCreateSchema
from app.utils.validators import validate_request
from app.utils.error_handlers import handle_errors

datasets_bp = Blueprint('datasets', __name__)
dataset_schema = DatasetSchema()
datasets_schema = DatasetSchema(many=True)

@datasets_bp.route('/', methods=['GET'])
@jwt_required()
@handle_errors
def get_datasets():
    """Get all datasets for the current user"""
    current_user_id = get_jwt_identity()
    datasets = Dataset.query.filter_by(user_id=current_user_id).all()
    return jsonify(datasets_schema.dump(datasets)), 200

@datasets_bp.route('/<int:dataset_id>', methods=['GET'])
@jwt_required()
@handle_errors
def get_dataset(dataset_id):
    """Get a specific dataset by ID"""
    current_user_id = get_jwt_identity()
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user_id).first_or_404()
    return jsonify(dataset_schema.dump(dataset)), 200

@datasets_bp.route('/', methods=['POST'])
@jwt_required()
@validate_request(DatasetCreateSchema)
@handle_errors
def create_dataset():
    """Create a new dataset"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    dataset = Dataset(**data, user_id=current_user_id)
    dataset.save()
    return jsonify(dataset_schema.dump(dataset)), 201

@datasets_bp.route('/<int:dataset_id>', methods=['PUT'])
@jwt_required()
@validate_request(DatasetCreateSchema)
@handle_errors
def update_dataset(dataset_id):
    """Update an existing dataset"""
    current_user_id = get_jwt_identity()
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user_id).first_or_404()
    data = request.get_json()
    dataset.update(**data)
    return jsonify(dataset_schema.dump(dataset)), 200

@datasets_bp.route('/<int:dataset_id>', methods=['DELETE'])
@jwt_required()
@handle_errors
def delete_dataset(dataset_id):
    """Delete a dataset"""
    current_user_id = get_jwt_identity()
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user_id).first_or_404()
    dataset.delete()
    return '', 204 