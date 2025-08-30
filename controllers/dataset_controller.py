import pandas as pd
import uuid
from datetime import datetime
from typing import Dict, Any, List
from fastapi import UploadFile, HTTPException
from database.firebase_db import firestore_client
from logger.bs_logger import bs_logger


class DatasetController:
    def __init__(self):
        self.db = firestore_client
        self.datasets_collection = "datasets"
        self.applications_subcollection = "applications"

    async def upload_dataset(self, file: UploadFile) -> Dict[str, Any]:
        """
        Upload and process CSV dataset file, storing dataset metadata and individual applications
        """
        try:
            # Validate file type
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="File must be a CSV")
            
            # Generate unique dataset ID
            dataset_id = str(uuid.uuid4())
            
            # Read CSV file
            contents = await file.read()
            
            # Parse CSV using pandas
            try:
                df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
            
            # Get basic dataset information
            num_applications = len(df)
            columns = df.columns.tolist()
            
            # Dataset metadata
            dataset_metadata = {
                "id": dataset_id,
                "filename": file.filename,
                "file_size": len(contents),
                "content_type": file.content_type,
                "columns": columns,
                "num_columns": len(columns),
                "num_applications": num_applications,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Store dataset metadata in Firebase
            dataset_ref = self.db.collection(self.datasets_collection).document(dataset_id)
            dataset_ref.set(dataset_metadata)
            
            # Store individual applications in subcollection
            applications_ref = dataset_ref.collection(self.applications_subcollection)
            
            # Process each row and store as individual application
            for index, row in df.iterrows():
                application_id = str(uuid.uuid4())
                application_data = {
                    "id": application_id,
                    "row_index": index,
                    "data": row.to_dict(),
                    "created_at": datetime.utcnow()
                }
                applications_ref.document(application_id).set(application_data)
            
            bs_logger.info(f"Dataset {dataset_id} uploaded successfully with {num_applications} applications")
            
            return {
                "dataset_id": dataset_id,
                "message": "Dataset uploaded successfully",
                "metadata": dataset_metadata
            }
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error(f"Error uploading dataset: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get dataset metadata by ID
        """
        try:
            dataset_ref = self.db.collection(self.datasets_collection).document(dataset_id)
            dataset_doc = dataset_ref.get()
            
            if not dataset_doc.exists:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            return dataset_doc.to_dict()
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error(f"Error retrieving dataset {dataset_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def get_datasets(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get list of all datasets with pagination
        """
        try:
            query = self.db.collection(self.datasets_collection).order_by("created_at", direction="DESCENDING")
            
            if offset > 0:
                # Get the document to start after for pagination
                docs = query.limit(offset).get()
                if docs:
                    query = query.start_after(docs[-1])
            
            datasets = query.limit(limit).get()
            
            return [doc.to_dict() for doc in datasets]
            
        except Exception as e:
            bs_logger.error(f"Error retrieving datasets: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def get_applications(self, dataset_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get applications from a specific dataset with pagination
        """
        try:
            dataset_ref = self.db.collection(self.datasets_collection).document(dataset_id)
            
            # Check if dataset exists
            if not dataset_ref.get().exists:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            applications_ref = dataset_ref.collection(self.applications_subcollection)
            query = applications_ref.order_by("row_index")
            
            if offset > 0:
                docs = query.limit(offset).get()
                if docs:
                    query = query.start_after(docs[-1])
            
            applications = query.limit(limit).get()
            
            return [doc.to_dict() for doc in applications]
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error(f"Error retrieving applications for dataset {dataset_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def delete_dataset(self, dataset_id: str) -> Dict[str, str]:
        """
        Delete a dataset and all its applications
        """
        try:
            dataset_ref = self.db.collection(self.datasets_collection).document(dataset_id)
            
            # Check if dataset exists
            if not dataset_ref.get().exists:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            # Delete all applications in subcollection
            applications_ref = dataset_ref.collection(self.applications_subcollection)
            applications = applications_ref.get()
            
            for app_doc in applications:
                app_doc.reference.delete()
            
            # Delete the dataset document
            dataset_ref.delete()
            
            bs_logger.info(f"Dataset {dataset_id} deleted successfully")
            
            return {"message": f"Dataset {dataset_id} deleted successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")