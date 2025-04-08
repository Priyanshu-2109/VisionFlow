# app/tasks/data_processing.py
from app import celery, db
from app.models.dataset import Dataset
from app.modules.data_processor.preprocessor import DataPreprocessor
import pandas as pd
import time
import logging

logger = logging.getLogger(__name__)

@celery.task(bind=True, max_retries=3)
def process_dataset_async(self, dataset_id, preprocessing_steps):
    """Process dataset asynchronously"""
    try:
        # Get dataset
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            logger.error(f"Dataset {dataset_id} not found")
            return {'status': 'error', 'message': 'Dataset not found'}
        
        # Update status
        dataset.processing_status = 'processing'
        db.session.commit()
        
        # Load dataset
        preprocessor = DataPreprocessor()
        df = preprocessor.load_dataset(dataset.file_path)
        
        # Apply each preprocessing step
        for step in preprocessing_steps:
            step_type = step.get('type')
            params = step.get('params', {})
            
            if step_type == 'handle_missing_values':
                preprocessor.handle_missing_values(params)
            elif step_type == 'handle_outliers':
                preprocessor.handle_outliers(params.get('method'), params.get('threshold'))
            elif step_type == 'encode_categorical_features':
                preprocessor.encode_categorical_features(params.get('method'), params.get('max_categories'))
            elif step_type == 'normalize_numerical_features':
                preprocessor.normalize_numerical_features(params.get('method'))
            elif step_type == 'process_datetime_features':
                preprocessor.process_datetime_features()
            
            # Simulate some processing time for demo purposes
            time.sleep(1)
        
        # Update dataset with preprocessing steps
        dataset.is_processed = True
        dataset.processing_status = 'completed'
        dataset.set_preprocessing_steps(preprocessor.preprocessing_steps)
        
        db.session.commit()
        
        return {
            'status': 'success',
            'dataset_id': dataset_id,
            'preprocessing_summary': preprocessor.get_data_summary()
        }
        
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_id}: {str(e)}")
        
        # Update dataset with error status
        try:
            dataset = Dataset.query.get(dataset_id)
            if dataset:
                dataset.processing_status = 'error'
                db.session.commit()
        except:
            pass
        
        # Retry task
        self.retry(exc=e, countdown=60)  # Retry after 1 minute