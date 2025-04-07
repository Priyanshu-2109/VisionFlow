from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from app.models.user import User
from app.models.dataset import Dataset
from app.models.report import Report
from app.extensions import db
from app.utils.cache import cache

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/analytics', methods=['GET'])
@cache.cached(timeout=300)  # Cache for 5 minutes
def get_analytics():
    time_range = request.args.get('timeRange', '7d')
    
    # Calculate date range
    end_date = datetime.utcnow()
    if time_range == '24h':
        start_date = end_date - timedelta(days=1)
    elif time_range == '7d':
        start_date = end_date - timedelta(days=7)
    elif time_range == '30d':
        start_date = end_date - timedelta(days=30)
    else:  # 90d
        start_date = end_date - timedelta(days=90)

    # Get total counts
    total_users = User.query.count()
    active_datasets = Dataset.query.filter_by(status='active').count()
    total_reports = Report.query.count()

    # Calculate growth percentages
    previous_period_start = start_date - (end_date - start_date)
    
    previous_users = User.query.filter(User.created_at < previous_period_start).count()
    previous_datasets = Dataset.query.filter(
        Dataset.created_at < previous_period_start,
        Dataset.status == 'active'
    ).count()
    previous_reports = Report.query.filter(Report.created_at < previous_period_start).count()

    user_growth = ((total_users - previous_users) / previous_users * 100) if previous_users > 0 else 0
    dataset_growth = ((active_datasets - previous_datasets) / previous_datasets * 100) if previous_datasets > 0 else 0
    report_growth = ((total_reports - previous_reports) / previous_reports * 100) if previous_reports > 0 else 0

    # Get trends data
    trends = {
        'users': get_trend_data(User, 'created_at', start_date, end_date),
        'datasets': get_trend_data(Dataset, 'created_at', start_date, end_date),
        'reports': get_trend_data(Report, 'created_at', start_date, end_date)
    }

    # Get recent activity
    recent_activity = get_recent_activity()

    return jsonify({
        'totalUsers': total_users,
        'activeDatasets': active_datasets,
        'totalReports': total_reports,
        'userGrowth': round(user_growth, 2),
        'datasetGrowth': round(dataset_growth, 2),
        'reportGrowth': round(report_growth, 2),
        'trends': trends,
        'recentActivity': recent_activity
    })

def get_trend_data(model, date_field, start_date, end_date):
    """Get daily counts for a model within a date range."""
    daily_counts = db.session.query(
        func.date(getattr(model, date_field)).label('date'),
        func.count().label('count')
    ).filter(
        getattr(model, date_field).between(start_date, end_date)
    ).group_by(
        func.date(getattr(model, date_field))
    ).all()

    return [
        {
            'date': str(count.date),
            'count': count.count
        }
        for count in daily_counts
    ]

def get_recent_activity():
    """Get recent activity across all models."""
    activities = []
    
    # Get recent user registrations
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    for user in recent_users:
        activities.append({
            'type': 'user',
            'description': f'New user registered: {user.email}',
            'timestamp': user.created_at,
            'icon': 'UserIcon'
        })

    # Get recent dataset uploads
    recent_datasets = Dataset.query.order_by(Dataset.created_at.desc()).limit(5).all()
    for dataset in recent_datasets:
        activities.append({
            'type': 'dataset',
            'description': f'New dataset uploaded: {dataset.name}',
            'timestamp': dataset.created_at,
            'icon': 'DocumentTextIcon'
        })

    # Get recent report generations
    recent_reports = Report.query.order_by(Report.created_at.desc()).limit(5).all()
    for report in recent_reports:
        activities.append({
            'type': 'report',
            'description': f'New report generated: {report.title}',
            'timestamp': report.created_at,
            'icon': 'ChartBarIcon'
        })

    # Sort all activities by timestamp and take the most recent 10
    activities.sort(key=lambda x: x['timestamp'], reverse=True)
    return activities[:10] 