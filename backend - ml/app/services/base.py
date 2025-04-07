from typing import Any, Dict, List, Optional, Type, TypeVar
from app.models.base import BaseModel
from app.schemas.base import BaseSchema

ModelType = TypeVar("ModelType", bound=BaseModel)
SchemaType = TypeVar("SchemaType", bound=BaseSchema)

class BaseService:
    """Base service class with common CRUD operations"""
    
    def __init__(self, model: Type[ModelType], schema: Type[SchemaType]):
        self.model = model
        self.schema = schema

    def get(self, id: int) -> Optional[Dict[str, Any]]:
        """Get a single item by ID"""
        item = self.model.get_by_id(id)
        return self.schema().dump(item) if item else None

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all items"""
        items = self.model.get_all()
        return self.schema(many=True).dump(items)

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new item"""
        schema = self.schema()
        validated_data = schema.load(data)
        item = self.model(**validated_data)
        item.save()
        return schema.dump(item)

    def update(self, id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing item"""
        item = self.model.get_by_id(id)
        if not item:
            return None
            
        schema = self.schema()
        validated_data = schema.load(data, partial=True)
        for key, value in validated_data.items():
            setattr(item, key, value)
            
        item.save()
        return schema.dump(item)

    def delete(self, id: int) -> bool:
        """Delete an item"""
        item = self.model.get_by_id(id)
        if not item:
            return False
            
        item.delete()
        return True

    def filter(self, **kwargs) -> List[Dict[str, Any]]:
        """Filter items by attributes"""
        items = self.model.filter(**kwargs)
        return self.schema(many=True).dump(items) 