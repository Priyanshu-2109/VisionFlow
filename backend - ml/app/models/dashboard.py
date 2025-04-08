from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, Enum, Boolean, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.session import Base
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import enum

class DashboardType(str, enum.Enum):
    ANALYTICS = "analytics"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    CUSTOM = "custom"

class ReportType(str, enum.Enum):
    PERFORMANCE = "performance"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    CUSTOM = "custom"

# Association table for dashboard sharing
dashboard_shares = Table(
    'dashboard_shares',
    Base.metadata,
    Column('dashboard_id', Integer, ForeignKey('dashboards.id')),
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('permission', String)  # read, write, admin
)

class Dashboard(Base):
    __tablename__ = "dashboards"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    type = Column(Enum(DashboardType))
    layout = Column(JSON)  # Dashboard layout configuration
    filters = Column(JSON)  # Global dashboard filters
    is_public = Column(Boolean, default=False)
    is_template = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    owner = relationship("User", back_populates="dashboards")
    shared_with = relationship("User", secondary=dashboard_shares, back_populates="shared_dashboards")
    reports = relationship("Report", back_populates="dashboard")
    widgets = relationship("Widget", back_populates="dashboard")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    type = Column(Enum(ReportType))
    dashboard_id = Column(Integer, ForeignKey("dashboards.id"))
    content = Column(JSON)  # Report content and configuration
    schedule = Column(JSON)  # Report schedule configuration
    recipients = Column(JSON)  # Report recipients
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    dashboard = relationship("Dashboard", back_populates="reports")
    creator = relationship("User", back_populates="reports")
    executions = relationship("ReportExecution", back_populates="report")

class Widget(Base):
    __tablename__ = "widgets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    dashboard_id = Column(Integer, ForeignKey("dashboards.id"))
    type = Column(String)  # chart, table, metric, etc.
    config = Column(JSON)  # Widget configuration
    data = Column(JSON)  # Widget data
    position = Column(JSON)  # Widget position in dashboard
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    dashboard = relationship("Dashboard", back_populates="widgets")
    creator = relationship("User", back_populates="widgets")

class ReportExecution(Base):
    __tablename__ = "report_executions"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    status = Column(String)  # success, failed, running
    result = Column(JSON)  # Execution result
    error = Column(Text)  # Error message if failed
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    report = relationship("Report", back_populates="executions")

# Pydantic models for API
class DashboardBase(BaseModel):
    name: str
    description: Optional[str] = None
    type: DashboardType
    layout: Optional[dict] = None
    filters: Optional[dict] = None
    is_public: bool = False
    is_template: bool = False

class DashboardCreate(DashboardBase):
    pass

class DashboardUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[DashboardType] = None
    layout: Optional[dict] = None
    filters: Optional[dict] = None
    is_public: Optional[bool] = None
    is_template: Optional[bool] = None

class DashboardInDB(DashboardBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]
    created_by: int

    class Config:
        from_attributes = True

class Dashboard(DashboardInDB):
    pass

class ReportBase(BaseModel):
    name: str
    description: Optional[str] = None
    type: ReportType
    dashboard_id: int
    content: Optional[dict] = None
    schedule: Optional[dict] = None
    recipients: Optional[list] = None

class ReportCreate(ReportBase):
    pass

class ReportUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[ReportType] = None
    content: Optional[dict] = None
    schedule: Optional[dict] = None
    recipients: Optional[list] = None

class ReportInDB(ReportBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]
    created_by: int

    class Config:
        from_attributes = True

class Report(ReportInDB):
    pass

class WidgetBase(BaseModel):
    name: str
    description: Optional[str] = None
    dashboard_id: int
    type: str
    config: Optional[dict] = None
    data: Optional[dict] = None
    position: Optional[dict] = None

class WidgetCreate(WidgetBase):
    pass

class WidgetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    config: Optional[dict] = None
    data: Optional[dict] = None
    position: Optional[dict] = None

class WidgetInDB(WidgetBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]
    created_by: int

    class Config:
        from_attributes = True

class Widget(WidgetInDB):
    pass 