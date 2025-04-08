from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class DataSourceType(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"

class DataSourceStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    SYNCING = "syncing"

class DataSourceBase(BaseModel):
    name: str
    description: Optional[str] = None
    type: DataSourceType
    status: DataSourceStatus = DataSourceStatus.INACTIVE
    connection_details: Dict[str, Any]
    schema: Optional[Dict[str, Any]] = None
    is_public: bool = False
    refresh_interval: Optional[int] = None  # in minutes
    last_sync: Optional[datetime] = None
    next_sync: Optional[datetime] = None
    error_message: Optional[str] = None
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    owner_id: Optional[int] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class DataSourceCreate(DataSourceBase):
    pass

class DataSourceUpdate(DataSourceBase):
    name: Optional[str] = None
    type: Optional[DataSourceType] = None
    connection_details: Optional[Dict[str, Any]] = None

class DataSourceInDBBase(DataSourceBase):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class DataSource(DataSourceInDBBase):
    pass

class DataSourceInDB(DataSourceInDBBase):
    pass

class DataSourceStats(BaseModel):
    id: int
    data_source_id: int
    total_rows: int
    total_size_bytes: int
    last_sync_duration: Optional[float]  # in seconds
    sync_success_rate: float
    error_count: int
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime

class DataSourceSync(BaseModel):
    id: int
    data_source_id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]  # in seconds
    rows_processed: Optional[int]
    rows_failed: Optional[int]
    error_message: Optional[str]
    metadata: Dict[str, Any] = {}

class DataSourceValidation(BaseModel):
    id: int
    data_source_id: int
    validation_type: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]  # in seconds
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class DataSourceSchema(BaseModel):
    id: int
    data_source_id: int
    column_name: str
    data_type: str
    is_nullable: bool
    default_value: Optional[Any]
    constraints: List[Dict[str, Any]] = []
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class DataSourceTransform(BaseModel):
    id: int
    data_source_id: int
    name: str
    description: Optional[str] = None
    transform_type: str
    parameters: Dict[str, Any]
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

class DataSourceSchedule(BaseModel):
    id: int
    data_source_id: int
    cron_expression: str
    is_active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class DataSourceAccess(BaseModel):
    id: int
    data_source_id: int
    user_id: int
    permission: str
    granted_by: int
    granted_at: datetime
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class DataSourceAuditLog(BaseModel):
    id: int
    data_source_id: int
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime
    created_by: Optional[int]

class DataSourceSample(BaseModel):
    id: int
    data_source_id: int
    data: List[Dict[str, Any]]
    row_count: int
    created_at: datetime
    created_by: Optional[int]

class DataSourceMetadata(BaseModel):
    id: int
    data_source_id: int
    key: str
    value: Any
    created_at: datetime
    updated_at: datetime
    created_by: Optional[int] 