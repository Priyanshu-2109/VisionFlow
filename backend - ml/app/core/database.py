from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    echo=settings.DB_ECHO
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
Base = declarative_base()

def setup_database():
    """Setup database connection and create tables"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"Error setting up database: {e}")
        raise

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
def init_db():
    from app.models.base import Base
    Base.metadata.create_all(bind=engine)

# Drop database
def drop_db():
    from app.models.base import Base
    Base.metadata.drop_all(bind=engine)

# Database health check
def check_db_connection():
    try:
        # Try to connect to the database
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False

# Add event listeners for connection pool management
from sqlalchemy import event
from sqlalchemy.engine import Engine
import logging

logger = logging.getLogger(__name__)

@event.listens_for(Engine, "connect")
def connect(dbapi_connection, connection_record):
    logger.info("Database connection established")

@event.listens_for(Engine, "checkout")
def checkout(dbapi_connection, connection_record, connection_proxy):
    logger.debug("Database connection checked out from pool")

@event.listens_for(Engine, "checkin")
def checkin(dbapi_connection, connection_record):
    logger.debug("Database connection returned to pool")

# Add connection pool statistics
def get_pool_stats():
    """Get database connection pool statistics"""
    return {
        "pool_size": engine.pool.size(),
        "checked_out": engine.pool.checkedout(),
        "overflow": engine.pool.overflow(),
        "checkedin": engine.pool.checkedin(),
    }

# Add utility functions for transaction management
from contextlib import contextmanager
from typing import Generator

@contextmanager
def get_db() -> Generator:
    """Get database session with automatic rollback on exception"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

@contextmanager
def transaction():
    """
    Transaction context manager.
    Automatically rolls back transaction on exception.
    """
    with get_db() as db:
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise 