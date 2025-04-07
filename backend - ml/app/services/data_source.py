from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pymongo
import redis
import elasticsearch
import boto3
from google.oauth2 import service_account
from googleapiclient.discovery import build
import requests
from app.models.data_source import DataSourceType, DataSourceStatus
from app.core.config import settings
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DataSourceService:
    def __init__(self):
        self.connections = {}

    def connect(self, data_source_type: DataSourceType, connection_params: Dict[str, Any]) -> Any:
        """Establish connection to the data source."""
        try:
            if data_source_type == DataSourceType.SQL:
                engine = create_engine(connection_params["connection_string"])
                self.connections[data_source_type] = engine
                return engine
            elif data_source_type == DataSourceType.MONGODB:
                client = pymongo.MongoClient(connection_params["connection_string"])
                self.connections[data_source_type] = client
                return client
            elif data_source_type == DataSourceType.REDIS:
                client = redis.Redis(
                    host=connection_params["host"],
                    port=connection_params["port"],
                    db=connection_params["db"]
                )
                self.connections[data_source_type] = client
                return client
            elif data_source_type == DataSourceType.ELASTICSEARCH:
                client = elasticsearch.Elasticsearch([connection_params["connection_string"]])
                self.connections[data_source_type] = client
                return client
            elif data_source_type == DataSourceType.S3:
                client = boto3.client(
                    's3',
                    aws_access_key_id=connection_params["access_key"],
                    aws_secret_access_key=connection_params["secret_key"],
                    region_name=connection_params["region"]
                )
                self.connections[data_source_type] = client
                return client
            elif data_source_type == DataSourceType.GOOGLE_SHEETS:
                credentials = service_account.Credentials.from_service_account_file(
                    connection_params["credentials_file"],
                    scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
                )
                service = build('sheets', 'v4', credentials=credentials)
                self.connections[data_source_type] = service
                return service
            else:
                raise ValueError(f"Unsupported data source type: {data_source_type}")
        except Exception as e:
            logger.error(f"Error connecting to data source: {str(e)}")
            raise

    def read_csv(self, file_path: str) -> pd.DataFrame:
        """Read data from a CSV file."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise

    def read_excel(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Read data from an Excel file."""
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise

    def read_json(self, file_path: str) -> pd.DataFrame:
        """Read data from a JSON file."""
        try:
            return pd.read_json(file_path)
        except Exception as e:
            logger.error(f"Error reading JSON file: {str(e)}")
            raise

    def read_sql(self, query: str, connection_params: Dict[str, Any]) -> pd.DataFrame:
        """Read data from a SQL database."""
        try:
            engine = self.connect(DataSourceType.SQL, connection_params)
            return pd.read_sql(query, engine)
        except Exception as e:
            logger.error(f"Error reading from SQL database: {str(e)}")
            raise

    def read_mongodb(self, connection_params: Dict[str, Any], query: Dict[str, Any]) -> pd.DataFrame:
        """Read data from MongoDB."""
        try:
            client = self.connect(DataSourceType.MONGODB, connection_params)
            db = client[connection_params["database"]]
            collection = db[connection_params["collection"]]
            cursor = collection.find(query)
            return pd.DataFrame(list(cursor))
        except Exception as e:
            logger.error(f"Error reading from MongoDB: {str(e)}")
            raise

    def read_redis(self, connection_params: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        """Read data from Redis."""
        try:
            client = self.connect(DataSourceType.REDIS, connection_params)
            return {key: client.get(key) for key in keys}
        except Exception as e:
            logger.error(f"Error reading from Redis: {str(e)}")
            raise

    def read_elasticsearch(self, connection_params: Dict[str, Any], query: Dict[str, Any]) -> pd.DataFrame:
        """Read data from Elasticsearch."""
        try:
            client = self.connect(DataSourceType.ELASTICSEARCH, connection_params)
            response = client.search(
                index=connection_params["index"],
                body=query
            )
            return pd.DataFrame([hit["_source"] for hit in response["hits"]["hits"]])
        except Exception as e:
            logger.error(f"Error reading from Elasticsearch: {str(e)}")
            raise

    def read_s3(self, connection_params: Dict[str, Any], file_path: str) -> pd.DataFrame:
        """Read data from S3."""
        try:
            client = self.connect(DataSourceType.S3, connection_params)
            response = client.get_object(
                Bucket=connection_params["bucket"],
                Key=file_path
            )
            if file_path.endswith('.csv'):
                return pd.read_csv(response['Body'])
            elif file_path.endswith('.xlsx'):
                return pd.read_excel(response['Body'])
            elif file_path.endswith('.json'):
                return pd.read_json(response['Body'])
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        except Exception as e:
            logger.error(f"Error reading from S3: {str(e)}")
            raise

    def read_google_sheets(self, connection_params: Dict[str, Any], range_name: str) -> pd.DataFrame:
        """Read data from Google Sheets."""
        try:
            service = self.connect(DataSourceType.GOOGLE_SHEETS, connection_params)
            sheet = service.spreadsheets()
            result = sheet.values().get(
                spreadsheetId=connection_params["spreadsheet_id"],
                range=range_name
            ).execute()
            values = result.get('values', [])
            if not values:
                return pd.DataFrame()
            return pd.DataFrame(values[1:], columns=values[0])
        except Exception as e:
            logger.error(f"Error reading from Google Sheets: {str(e)}")
            raise

    def read_api(self, url: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Read data from an API endpoint."""
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except Exception as e:
            logger.error(f"Error reading from API: {str(e)}")
            raise

    def validate_data_source(self, data_source_type: DataSourceType, connection_params: Dict[str, Any]) -> bool:
        """Validate the data source connection."""
        try:
            if data_source_type == DataSourceType.SQL:
                engine = self.connect(DataSourceType.SQL, connection_params)
                engine.connect()
                return True
            elif data_source_type == DataSourceType.MONGODB:
                client = self.connect(DataSourceType.MONGODB, connection_params)
                client.server_info()
                return True
            elif data_source_type == DataSourceType.REDIS:
                client = self.connect(DataSourceType.REDIS, connection_params)
                client.ping()
                return True
            elif data_source_type == DataSourceType.ELASTICSEARCH:
                client = self.connect(DataSourceType.ELASTICSEARCH, connection_params)
                client.ping()
                return True
            elif data_source_type == DataSourceType.S3:
                client = self.connect(DataSourceType.S3, connection_params)
                client.list_buckets()
                return True
            elif data_source_type == DataSourceType.GOOGLE_SHEETS:
                service = self.connect(DataSourceType.GOOGLE_SHEETS, connection_params)
                service.spreadsheets().get(
                    spreadsheetId=connection_params["spreadsheet_id"]
                ).execute()
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error validating data source: {str(e)}")
            return False

    def get_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get the schema of the data."""
        try:
            return {
                "columns": data.columns.tolist(),
                "dtypes": data.dtypes.astype(str).to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "unique_values": data.nunique().to_dict()
            }
        except Exception as e:
            logger.error(f"Error getting schema: {str(e)}")
            raise 