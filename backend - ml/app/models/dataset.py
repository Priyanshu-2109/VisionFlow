# app/models/dataset.py
from app.db.session import Base
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey
from datetime import datetime
import uuid
import json

class Dataset(Base):
    __tablename__ = 'datasets'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(120))
    description = Column(Text, nullable=True)
    file_path = Column(String(255))
    file_type = Column(String(20))
    size_bytes = Column(Integer)
    columns = Column(Text)  # JSON string of column names and types
    row_count = Column(Integer)
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default='pending')
    preprocessing_steps = Column(Text, nullable=True)  # JSON string of preprocessing steps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(String(36), ForeignKey('users.id'))
    
    def set_columns(self, columns_dict):
        self.columns = json.dumps(columns_dict)
        
    def get_columns(self):
        return json.loads(self.columns) if self.columns else {}
    
    def set_preprocessing_steps(self, steps):
        self.preprocessing_steps = json.dumps(steps)
        
    def get_preprocessing_steps(self):
        return json.loads(self.preprocessing_steps) if self.preprocessing_steps else []
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'file_type': self.file_type,
            'size_bytes': self.size_bytes,
            'columns': self.get_columns(),
            'row_count': self.row_count,
            'is_processed': self.is_processed,
            'processing_status': self.processing_status,
            'preprocessing_steps': self.get_preprocessing_steps(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'user_id': self.user_id
        }