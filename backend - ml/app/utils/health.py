from flask import Blueprint, jsonify
from app.core.database import SessionLocal
from app.core.cache import cache
import psutil
import os

health_bp = Blueprint('health', __name__)

def check_database():
    """Check database connection"""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception:
        return False

def check_redis():
    """Check Redis connection"""
    try:
        cache.redis_client.ping()
        return True
    except Exception:
        return False

def check_disk_space():
    """Check disk space"""
    try:
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent
        }
    except Exception:
        return None

def check_memory():
    """Check memory usage"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
    except Exception:
        return None

def check_cpu():
    """Check CPU usage"""
    try:
        return psutil.cpu_percent(interval=1)
    except Exception:
        return None

@health_bp.route('/health')
def check_health():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy',
        'services': {
            'database': check_database(),
            'redis': check_redis()
        },
        'system': {
            'disk_space': check_disk_space(),
            'memory': check_memory(),
            'cpu': check_cpu()
        }
    }
    
    # Check if any service is unhealthy
    if not all(health_status['services'].values()):
        health_status['status'] = 'unhealthy'
        return jsonify(health_status), 503
    
    return jsonify(health_status), 200

@health_bp.route('/readiness')
def check_readiness():
    """Readiness check endpoint"""
    readiness_status = {
        'status': 'ready',
        'services': {
            'database': check_database(),
            'redis': check_redis()
        }
    }
    
    # Check if any service is not ready
    if not all(readiness_status['services'].values()):
        readiness_status['status'] = 'not_ready'
        return jsonify(readiness_status), 503
    
    return jsonify(readiness_status), 200

@health_bp.route('/liveness')
def check_liveness():
    """Liveness check endpoint"""
    liveness_status = {
        'status': 'alive',
        'system': {
            'disk_space': check_disk_space(),
            'memory': check_memory(),
            'cpu': check_cpu()
        }
    }
    
    # Check if system resources are critically low
    if liveness_status['system']['disk_space'] and liveness_status['system']['disk_space']['percent'] > 90:
        liveness_status['status'] = 'critical'
        return jsonify(liveness_status), 503
    
    return jsonify(liveness_status), 200 