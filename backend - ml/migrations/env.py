import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from flask import current_app

from alembic import context

# Import all models here
from app.models.user import User
from app.models.data_source import DataSource
from app.models.analytics import Analytics
from app.models.dashboard import Dashboard
from app.models.widget import Widget
from app.models.alert import Alert
from app.models.api_key import APIKey
from app.models.audit_log import AuditLog
from app.models.notification import Notification
from app.models.user_preference import UserPreference
from app.models.data_source_sync import DataSourceSync
from app.models.analytics_run import AnalyticsRun
from app.models.dashboard_share import DashboardShare
from app.models.widget_refresh import WidgetRefresh
from app.models.alert_rule import AlertRule
from app.models.api_key_usage import APIKeyUsage
from app.models.audit_log_detail import AuditLogDetail
from app.models.notification_preference import NotificationPreference
from app.models.user_session import UserSession
from app.models.data_source_validation import DataSourceValidation
from app.models.analytics_schedule import AnalyticsSchedule
from app.models.dashboard_version import DashboardVersion
from app.models.widget_template import WidgetTemplate
from app.models.alert_history import AlertHistory
from app.models.api_key_permission import APIKeyPermission
from app.models.audit_log_metadata import AuditLogMetadata
from app.models.notification_template import NotificationTemplate
from app.models.user_role import UserRole
from app.models.data_source_transform import DataSourceTransform
from app.models.analytics_visualization import AnalyticsVisualization
from app.models.dashboard_snapshot import DashboardSnapshot
from app.models.widget_comment import WidgetComment
from app.models.alert_condition import AlertCondition
from app.models.api_key_audit import APIKeyAudit
from app.models.audit_log_category import AuditLogCategory
from app.models.notification_channel import NotificationChannel
from app.models.user_permission import UserPermission
from app.models.data_source_schema import DataSourceSchema
from app.models.analytics_metric import AnalyticsMetric
from app.models.dashboard_layout import DashboardLayout
from app.models.widget_annotation import WidgetAnnotation
from app.models.alert_action import AlertAction
from app.models.api_key_scope import APIKeyScope
from app.models.audit_log_level import AuditLogLevel
from app.models.notification_status import NotificationStatus
from app.models.user_status import UserStatus
from app.models.data_source_type import DataSourceType
from app.models.analytics_type import AnalyticsType
from app.models.dashboard_type import DashboardType
from app.models.widget_type import WidgetType
from app.models.alert_type import AlertType
from app.models.api_key_type import APIKeyType
from app.models.audit_log_type import AuditLogType
from app.models.notification_type import NotificationType
from app.models.user_type import UserType
from app.models.data_source_status import DataSourceStatus
from app.models.analytics_status import AnalyticsStatus
from app.models.dashboard_status import DashboardStatus
from app.models.widget_status import WidgetStatus
from app.models.alert_status import AlertStatus
from app.models.api_key_status import APIKeyStatus
from app.models.audit_log_status import AuditLogStatus
from app.models.notification_priority import NotificationPriority
from app.models.user_activity import UserActivity
from app.models.data_source_activity import DataSourceActivity
from app.models.analytics_activity import AnalyticsActivity
from app.models.dashboard_activity import DashboardActivity
from app.models.widget_activity import WidgetActivity
from app.models.alert_activity import AlertActivity
from app.models.api_key_activity import APIKeyActivity
from app.models.audit_log_activity import AuditLogActivity
from app.models.notification_activity import NotificationActivity
from app.models.user_session_activity import UserSessionActivity
from app.models.data_source_sync_activity import DataSourceSyncActivity
from app.models.analytics_run_activity import AnalyticsRunActivity
from app.models.dashboard_share_activity import DashboardShareActivity
from app.models.widget_refresh_activity import WidgetRefreshActivity
from app.models.alert_rule_activity import AlertRuleActivity
from app.models.api_key_usage_activity import APIKeyUsageActivity
from app.models.audit_log_detail_activity import AuditLogDetailActivity
from app.models.notification_preference_activity import NotificationPreferenceActivity
from app.models.user_preference_activity import UserPreferenceActivity
from app.models.data_source_validation_activity import DataSourceValidationActivity
from app.models.analytics_schedule_activity import AnalyticsScheduleActivity
from app.models.dashboard_version_activity import DashboardVersionActivity
from app.models.widget_template_activity import WidgetTemplateActivity
from app.models.alert_history_activity import AlertHistoryActivity
from app.models.api_key_permission_activity import APIKeyPermissionActivity
from app.models.audit_log_metadata_activity import AuditLogMetadataActivity
from app.models.notification_template_activity import NotificationTemplateActivity
from app.models.user_role_activity import UserRoleActivity
from app.models.data_source_transform_activity import DataSourceTransformActivity
from app.models.analytics_visualization_activity import AnalyticsVisualizationActivity
from app.models.dashboard_snapshot_activity import DashboardSnapshotActivity
from app.models.widget_comment_activity import WidgetCommentActivity
from app.models.alert_condition_activity import AlertConditionActivity
from app.models.api_key_audit_activity import APIKeyAuditActivity
from app.models.audit_log_category_activity import AuditLogCategoryActivity
from app.models.notification_channel_activity import NotificationChannelActivity
from app.models.user_permission_activity import UserPermissionActivity
from app.models.data_source_schema_activity import DataSourceSchemaActivity
from app.models.analytics_metric_activity import AnalyticsMetricActivity
from app.models.dashboard_layout_activity import DashboardLayoutActivity
from app.models.widget_annotation_activity import WidgetAnnotationActivity
from app.models.alert_action_activity import AlertActionActivity
from app.models.api_key_scope_activity import APIKeyScopeActivity
from app.models.audit_log_level_activity import AuditLogLevelActivity
from app.models.notification_status_activity import NotificationStatusActivity
from app.models.user_status_activity import UserStatusActivity
from app.models.data_source_type_activity import DataSourceTypeActivity
from app.models.analytics_type_activity import AnalyticsTypeActivity
from app.models.dashboard_type_activity import DashboardTypeActivity
from app.models.widget_type_activity import WidgetTypeActivity
from app.models.alert_type_activity import AlertTypeActivity
from app.models.api_key_type_activity import APIKeyTypeActivity
from app.models.audit_log_type_activity import AuditLogTypeActivity
from app.models.notification_type_activity import NotificationTypeActivity
from app.models.user_type_activity import UserTypeActivity
from app.models.data_source_status_activity import DataSourceStatusActivity
from app.models.analytics_status_activity import AnalyticsStatusActivity
from app.models.dashboard_status_activity import DashboardStatusActivity
from app.models.widget_status_activity import WidgetStatusActivity
from app.models.alert_status_activity import AlertStatusActivity
from app.models.api_key_status_activity import APIKeyStatusActivity
from app.models.audit_log_status_activity import AuditLogStatusActivity
from app.models.notification_priority_activity import NotificationPriorityActivity

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger('alembic.env')


def get_engine():
    try:
        # this works with Flask-SQLAlchemy<3 and Alchemical
        return current_app.extensions['migrate'].db.get_engine()
    except TypeError:
        # this works with Flask-SQLAlchemy>=3
        return current_app.extensions['migrate'].db.engine


def get_engine_url():
    try:
        return get_engine().url.render_as_string(hide_password=False).replace(
            '%', '%%')
    except AttributeError:
        return str(get_engine().url).replace('%', '%%')


# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_metadata():
    if hasattr(target_db, 'metadatas'):
        return target_db.metadatas[None]
    return target_db.metadata


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
