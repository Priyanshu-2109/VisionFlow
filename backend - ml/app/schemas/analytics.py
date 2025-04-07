from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class AnalyticsType(str, Enum):
    DESCRIPTIVE = "descriptive"
    PREDICTIVE = "predictive"
    DIAGNOSTIC = "diagnostic"
    PRESCRIPTIVE = "prescriptive"

class AnalyticsStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    ERROR = "error"

class AnalyticsBase(BaseModel):
    name: str
    description: Optional[str] = None
    type: AnalyticsType
    status: AnalyticsStatus = AnalyticsStatus.DRAFT
    data_source_id: Optional[int] = None
    owner_id: Optional[int] = None
    is_public: bool = False
    parameters: Dict[str, Any] = {}
    results: Optional[Dict[str, Any]] = None
    refresh_interval: Optional[int] = None  # in minutes
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None  # in seconds
    row_count: Optional[int] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class AnalyticsCreate(AnalyticsBase):
    pass

class AnalyticsUpdate(AnalyticsBase):
    name: Optional[str] = None
    type: Optional[AnalyticsType] = None
    parameters: Optional[Dict[str, Any]] = None

class AnalyticsInDBBase(AnalyticsBase):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class Analytics(AnalyticsInDBBase):
    pass

class AnalyticsInDB(AnalyticsInDBBase):
    pass

class AnalyticsStats(BaseModel):
    id: int
    analytics_id: int
    total_runs: int
    success_rate: float
    avg_execution_time: float
    last_run_duration: Optional[float]
    error_count: int
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime

class AnalyticsRun(BaseModel):
    id: int
    analytics_id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]  # in seconds
    rows_processed: Optional[int]
    error_message: Optional[str]
    results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}

class AnalyticsValidation(BaseModel):
    id: int
    analytics_id: int
    validation_type: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]  # in seconds
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class AnalyticsSchedule(BaseModel):
    id: int
    analytics_id: int
    cron_expression: str
    is_active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class AnalyticsAccess(BaseModel):
    id: int
    analytics_id: int
    user_id: int
    permission: str
    granted_by: int
    granted_at: datetime
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class AnalyticsAuditLog(BaseModel):
    id: int
    analytics_id: int
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime
    created_by: Optional[int]

class AnalyticsResult(BaseModel):
    id: int
    analytics_id: int
    run_id: int
    data: Dict[str, Any]
    visualization: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    created_by: Optional[int]

class AnalyticsVisualization(BaseModel):
    id: int
    analytics_id: int
    name: str
    type: str
    config: Dict[str, Any]
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[int]

class AnalyticsMetric(BaseModel):
    id: int
    analytics_id: int
    name: str
    description: Optional[str] = None
    type: str
    calculation: str
    parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[int]

class AnalyticsAlert(BaseModel):
    id: int
    analytics_id: int
    name: str
    description: Optional[str] = None
    condition: str
    threshold: float
    operator: str
    is_active: bool = True
    notification_channels: List[str] = []
    created_at: datetime
    updated_at: datetime
    created_by: Optional[int]

class AnalyticsExport(BaseModel):
    id: int
    analytics_id: int
    format: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    file_path: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    created_by: Optional[int] 