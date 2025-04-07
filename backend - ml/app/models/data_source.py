from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, Enum, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.session import Base
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
from datetime import datetime
import enum

class DataSourceType(str, enum.Enum):
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    SQL = "sql"
    API = "api"
    MONGODB = "mongodb"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    S3 = "s3"
    GOOGLE_SHEETS = "google_sheets"
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"
    ZENDESK = "zendesk"
    CUSTOM = "custom"

class DataSourceStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PROCESSING = "processing"

class DataSource(Base):
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    type = Column(Enum(DataSourceType))
    connection_params = Column(JSON)  # Connection parameters
    schema = Column(JSON)  # Data schema
    status = Column(Enum(DataSourceStatus), default=DataSourceStatus.INACTIVE)
    last_sync = Column(DateTime(timezone=True))
    sync_frequency = Column(Integer)  # in minutes
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    owner = relationship("User", back_populates="data_sources")
    analyses = relationship("Analysis", back_populates="data_source")
    datasets = relationship("Dataset", back_populates="data_source")

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"))
    query = Column(Text)  # SQL query or data transformation logic
    schema = Column(JSON)  # Dataset schema
    preview_data = Column(JSON)  # Sample data for preview
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    data_source = relationship("DataSource", back_populates="datasets")
    creator = relationship("User", back_populates="datasets")

# Pydantic models for API
class DataSourceBase(BaseModel):
    name: str
    description: Optional[str] = None
    type: DataSourceType
    connection_params: dict
    schema: Optional[dict] = None
    sync_frequency: Optional[int] = None
    is_active: bool = True

class DataSourceCreate(DataSourceBase):
    pass

class DataSourceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[DataSourceType] = None
    connection_params: Optional[dict] = None
    schema: Optional[dict] = None
    sync_frequency: Optional[int] = None
    is_active: Optional[bool] = None

class DataSourceInDB(DataSourceBase):
    id: int
    status: DataSourceStatus
    last_sync: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]
    created_by: int

    class Config:
        from_attributes = True

class DataSource(DataSourceInDB):
    pass

class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    data_source_id: int
    query: Optional[str] = None
    schema: Optional[dict] = None
    preview_data: Optional[dict] = None

class DatasetCreate(DatasetBase):
    pass

class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    query: Optional[str] = None
    schema: Optional[dict] = None
    preview_data: Optional[dict] = None

class DatasetInDB(DatasetBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]
    created_by: int

    class Config:
        from_attributes = True

class Dataset(DatasetInDB):
    pass 