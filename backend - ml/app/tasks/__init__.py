# app/tasks/__init__.py
from celery import Celery
from app.tasks.trend_analysis import (
    collect_trend_data_periodic,
    analyze_trends,
    generate_trend_report
)
from app.tasks.market_simulation import (
    run_market_simulation,
    generate_simulation_report
)
from app.tasks.data_processing import (
    process_dataset,
    clean_data,
    validate_data
)
from app.tasks.notifications import (
    send_notification,
    send_email_notification
)
from app.tasks.reports import (
    generate_daily_report,
    generate_weekly_report,
    generate_monthly_report
)

__all__ = [
    'collect_trend_data_periodic',
    'analyze_trends',
    'generate_trend_report',
    'run_market_simulation',
    'generate_simulation_report',
    'process_dataset',
    'clean_data',
    'validate_data',
    'send_notification',
    'send_email_notification',
    'generate_daily_report',
    'generate_weekly_report',
    'generate_monthly_report'
]

def make_celery(app):
    """Create and configure Celery instance"""
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery