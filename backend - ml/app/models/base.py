from datetime import datetime
from typing import Any
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.sql import func
from app import db

Base = declarative_base()

class BaseModel(db.Model):
    """Base model class with common functionality"""
    __abstract__ = True
    
    id: Any
    __name__: str
    
    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()
    
    # Common columns
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = db.Column(db.DateTime(timezone=True), nullable=True)
    
    def to_dict(self) -> dict:
        """Convert model to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update(self, **kwargs) -> None:
        """Update model attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def create(cls, **kwargs):
        """Create new model instance"""
        instance = cls(**kwargs)
        return instance
    
    def save(self):
        """Save the model to the database"""
        db.session.add(self)
        db.session.commit()
        return self
    
    def delete(self):
        """Delete the model from the database"""
        db.session.delete(self)
        db.session.commit()
    
    @classmethod
    def get_by_id(cls, id):
        """Get a model instance by ID"""
        return cls.query.get(id)
    
    @classmethod
    def get_all(cls):
        """Get all model instances"""
        return cls.query.all()
    
    @classmethod
    def filter(cls, **kwargs):
        """Filter models by attributes"""
        return cls.query.filter_by(**kwargs).all()
    
    @classmethod
    def count(cls):
        """Count total models"""
        return cls.query.count()
    
    @classmethod
    def exists(cls, **kwargs):
        """Check if model exists"""
        return cls.query.filter_by(**kwargs).first() is not None
    
    def soft_delete(self) -> None:
        """Soft delete the record"""
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore a soft-deleted record"""
        self.deleted_at = None 