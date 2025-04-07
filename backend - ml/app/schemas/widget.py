from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class WidgetType(str, Enum):
    CHART = "chart"
    TABLE = "table"
    KPI = "kpi"
    MAP = "map"
    TEXT = "text"
    FILTER = "filter"

class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"
    CANDLESTICK = "candlestick"
    RADAR = "radar"

class WidgetStatus(str, Enum):
    ACTIVE = "active"
    ERROR = "error"
    LOADING = "loading"

class WidgetBase(BaseModel):
    name: str
    description: Optional[str] = None
    type: WidgetType
    status: WidgetStatus = WidgetStatus.ACTIVE
    dashboard_id: int
    data_source_id: Optional[int] = None
    analytics_id: Optional[int] = None
    layout: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
    data: Optional[Dict[str, Any]] = None
    refresh_interval: Optional[int] = None  # in minutes
    last_refresh: Optional[datetime] = None
    next_refresh: Optional[datetime] = None
    error_message: Optional[str] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class WidgetCreate(WidgetBase):
    pass

class WidgetUpdate(WidgetBase):
    name: Optional[str] = None
    type: Optional[WidgetType] = None
    layout: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None

class WidgetInDBBase(WidgetBase):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class Widget(WidgetInDBBase):
    pass

class WidgetInDB(WidgetInDBBase):
    pass

class WidgetStats(BaseModel):
    id: int
    widget_id: int
    total_refreshes: int
    success_rate: float
    avg_load_time: float
    last_refresh_duration: Optional[float]
    error_count: int
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime

class WidgetRefresh(BaseModel):
    id: int
    widget_id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]  # in seconds
    error_message: Optional[str]
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}

class WidgetValidation(BaseModel):
    id: int
    widget_id: int
    validation_type: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]  # in seconds
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class WidgetSchedule(BaseModel):
    id: int
    widget_id: int
    cron_expression: str
    is_active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class WidgetAccess(BaseModel):
    id: int
    widget_id: int
    user_id: int
    permission: str
    granted_by: int
    granted_at: datetime
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class WidgetAuditLog(BaseModel):
    id: int
    widget_id: int
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime
    created_by: Optional[int]

class WidgetExport(BaseModel):
    id: int
    widget_id: int
    format: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    file_path: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    created_by: Optional[int]

class WidgetSnapshot(BaseModel):
    id: int
    widget_id: int
    name: str
    description: Optional[str] = None
    data: Dict[str, Any]
    created_at: datetime
    created_by: Optional[int]

class WidgetComment(BaseModel):
    id: int
    widget_id: int
    user_id: int
    content: str
    parent_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    created_by: Optional[int]

class WidgetAnnotation(BaseModel):
    id: int
    widget_id: int
    user_id: int
    type: str
    content: str
    position: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[int]

class WidgetBookmark(BaseModel):
    id: int
    widget_id: int
    user_id: int
    name: str
    description: Optional[str] = None
    filters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[int]

class WidgetTemplate(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    type: WidgetType
    layout: Dict[str, Any]
    config: Dict[str, Any]
    is_public: bool = False
    created_at: datetime
    updated_at: datetime
    created_by: Optional[int]

class WidgetVersion(BaseModel):
    id: int
    widget_id: int
    version: int
    changes: Dict[str, Any]
    created_at: datetime
    created_by: Optional[int] 